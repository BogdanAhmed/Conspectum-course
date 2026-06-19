import asyncio
from collections import Counter
import dataclasses
import json
import logging
import os
import pathlib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from threading import Lock
import typing

from faster_whisper import WhisperModel
import openai

try:
    from faster_whisper import BatchedInferencePipeline
except ImportError:  # Older faster-whisper versions and tests may not provide it.
    BatchedInferencePipeline = None

from .gpu import check_ctranslate2_cuda
from .gpu import choose_primary_gpu
from .gpu import configure_nvidia_runtime_paths
from .gpu import CTranslate2CudaStatus
from .gpu import format_gpu_info
from .gpu import NvidiaGpuInfo
from .gpu import recommend_whisper_batch_size
from .gpu import recommend_whisper_compute_type
from .gpu import run_nvidia_smi
from .logger import Logger

LANGUAGE_DETECTION_MS = 12_000
DEFAULT_AI_MODEL_NAME = "openai/gpt-oss-120b:free"
DEFAULT_AI_MODEL_FALLBACKS = (
    "openai/gpt-oss-20b:free",
    "z-ai/glm-4.5-air:free",
    "moonshotai/kimi-k2.6:free",
    "openrouter/free",
)
DEFAULT_WHISPER_MODEL_SIZE = "large-v3"
DEFAULT_WHISPER_DEVICE = "auto"
DEFAULT_WHISPER_COMPUTE_TYPE = "auto"
DEFAULT_WHISPER_BATCH_SIZE = 8
DEFAULT_WHISPER_BEAM_SIZE = 1
DEFAULT_WHISPER_BEST_OF = 1
DEFAULT_AUDIO_PREPROCESS_FILTER = "highpass=f=80,lowpass=f=8000,loudnorm=I=-16:TP=-1.5:LRA=11"
DEFAULT_SUMMARY_TRANSCRIPT_CONTEXT_CHARS = 24000
DEFAULT_POSTPROCESS_TRANSCRIPT_CONTEXT_CHARS = 30000
WHISPER_MODEL_FALLBACKS = ("large-v3", "medium", "small", "base")
CUDA_COMPUTE_TYPE_FALLBACKS = ("float16", "int8", "auto")
CPU_COMPUTE_TYPE_FALLBACKS = ("int8",)
FALSE_ENV_VALUES = {"0", "false", "no", "off"}
TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
SUPPORTED_WHISPER_DEVICES = {"auto", "cuda", "cpu"}
AI_RESPONSE_PARSE_ERROR_NAMES = {
    "APIResponseValidationError",
    "DecodingError",
    "JSONDecodeError",
}
AI_RESPONSE_PARSE_ERROR_MARKERS = (
    "expecting value",
    "failed to decode json",
    "invalid json",
    "jsondecodeerror",
    "response is not valid json",
)
CUDA_LIBRARY_ERROR_MARKERS = (
    "cublas",
    "cudnn",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "cuda driver",
    "cuda runtime",
)
UNSUPPORTED_COMPUTE_TYPE_MARKERS = (
    "compute type",
    "do not support efficient",
    "not support efficient",
    "unsupported compute",
)
AI_PROVIDER_LOGGER = logging.getLogger("conspectum.ai")
TRANSCRIPTION_LOGGER = logging.getLogger("conspectum.whisper")

LANGUAGE_NAMES = {
    "ru": "Russian",
    "en": "English",
}

DETAIL_LEVEL_PROMPTS = {
    "brief": (
        "Keep the overview very compact: 2-4 sentences only. Mention the topic, 2-3 core ideas, and the final takeaway. "
        "Do not list secondary examples or detailed derivations."
    ),
    "standard": (
        "Provide a balanced overview with the main topic, the most important concepts, and the key conclusion."
    ),
    "detailed": (
        "Provide a richer overview that mentions the learning goals, major definitions, formulas, examples, and conclusions."
    ),
}


@dataclasses.dataclass(frozen=True)
class TranscriptionModelSettings:
    model_name: str
    device: str
    compute_type: str
    cpu_threads: int
    num_workers: int


@dataclasses.dataclass(frozen=True)
class TranscriptionLoadContext:
    gpu: NvidiaGpuInfo | None = None
    ctranslate2: CTranslate2CudaStatus | None = None
    initial_cpu_fallback_reason: str | None = None


@dataclasses.dataclass(frozen=True)
class TranscriptionRuntimeInfo:
    requested_settings: TranscriptionModelSettings
    active_settings: TranscriptionModelSettings
    model_name: str
    gpu_name: str | None = None
    gpu_vram_mb: int | None = None
    fallback_reason: str | None = None


_TRANSCRIPTION_MODEL: WhisperModel | None = None
_TRANSCRIPTION_MODEL_NAME: str | None = None
_TRANSCRIPTION_MODEL_SETTINGS: TranscriptionModelSettings | None = None
_TRANSCRIPTION_RUNTIME_INFO: TranscriptionRuntimeInfo | None = None
_TRANSCRIPTION_BATCHED_PIPELINE: typing.Any | None = None
_TRANSCRIPTION_BATCHED_MODEL: WhisperModel | None = None
_TRANSCRIPTION_MODEL_LOCK = Lock()

TRANSCRIPTION_LANGUAGE_PROMPTS = {
    "ru": ("Русскоязычная лекция или разговор. Имена, аббревиатуры и специальные термины сохраняются как звучат."),
    "en": "English lecture or conversation. Names, abbreviations, and technical terms are preserved as spoken.",
}
TRANSCRIPTION_PROMPT_ECHO_MARKERS = (
    "transcribe it in english without translating it",
    "english lecture or conversation",
    "русскоязычная лекция или разговор",
    "распознавай речь на русском языке",
)
REFUSAL_MARKERS = (
    "i'm sorry, but i cannot assist",
    "i’m sorry, but i cannot assist",
    "i cannot assist with that request",
    "i can't assist with that request",
)

RUSSIAN_RETRY_SOURCE_LANGUAGES = {"uk", "be"}
TRANSCRIPTION_TO_OUTPUT_LANGUAGE = {
    "en": "en",
    "ru": "ru",
    "uk": "ru",
    "be": "ru",
}

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".oga",
    ".opus",
    ".flac",
    ".aac",
    ".mp4",
    ".m4b",
    ".webm",
}

SUPPORTED_AUDIO_MIME_TYPES = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/webm": ".webm",
    "video/webm": ".webm",
}


@dataclasses.dataclass
class Summary:
    title: str
    abstract: str


@dataclasses.dataclass
class TranscriptionResult:
    text: str
    language: str | None = None


def guess_audio_suffix(filename: str | None = None, mime_type: str | None = None) -> str:
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix in SUPPORTED_AUDIO_EXTENSIONS:
            return suffix

    if mime_type:
        normalized_mime = mime_type.lower().strip()
        if normalized_mime in SUPPORTED_AUDIO_MIME_TYPES:
            return SUPPORTED_AUDIO_MIME_TYPES[normalized_mime]

    return ".wav"


def is_supported_audio(filename: str | None = None, mime_type: str | None = None) -> bool:
    if filename and Path(filename).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
        return True

    if mime_type and mime_type.lower().strip() in SUPPORTED_AUDIO_MIME_TYPES:
        return True

    return False


def read_int_env(name: str, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    configured = os.environ.get(name)
    if configured is None:
        value = default
    else:
        try:
            value = int(configured.strip())
        except ValueError:
            value = default

    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def env_flag_enabled(name: str, default: bool = True) -> bool:
    configured = os.environ.get(name)
    if configured is None:
        return default
    return configured.strip().lower() not in FALSE_ENV_VALUES


def env_flag_explicitly_enabled(name: str, default: bool = False) -> bool:
    configured = os.environ.get(name)
    if configured is None:
        return default
    return configured.strip().lower() in TRUE_ENV_VALUES


def get_string_env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def parse_model_list(value: str) -> list[str]:
    return [model.strip() for model in re.split(r"[,;\s]+", value) if model.strip()]


def get_ai_model_name() -> str:
    return get_string_env("MODEL_NAME", DEFAULT_AI_MODEL_NAME)


def get_ai_model_fallbacks() -> list[str]:
    configured = os.environ.get("AI_MODEL_FALLBACKS", "").strip()
    if configured:
        return parse_model_list(configured)
    return list(DEFAULT_AI_MODEL_FALLBACKS)


def get_ai_model_candidates(primary_model: str | None = None) -> list[str]:
    candidates = [primary_model.strip() if primary_model else get_ai_model_name()]
    candidates.extend(get_ai_model_fallbacks())

    unique_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)
    return unique_candidates


def should_try_next_ai_model(exc: Exception) -> bool:
    exc_name = exc.__class__.__name__
    raw_message = str(exc).lower()
    if isinstance(exc, json.JSONDecodeError):
        return True
    if exc_name in AI_RESPONSE_PARSE_ERROR_NAMES:
        return True
    if any(marker in raw_message for marker in AI_RESPONSE_PARSE_ERROR_MARKERS):
        return True
    if exc_name in {"RateLimitError", "APITimeoutError", "APIConnectionError"}:
        return True
    if exc_name != "APIStatusError":
        return False

    status_code = getattr(exc, "status_code", None)
    if status_code in {402, 408, 409, 429, 500, 502, 503, 504}:
        return True
    return any(
        marker in raw_message
        for marker in (
            "credits",
            "can only afford",
            "rate-limited",
            "temporarily rate",
            "provider returned error",
            "prompt tokens limit exceeded",
        )
    )


async def create_chat_completion(
    ai: openai.AsyncOpenAI,
    *,
    messages: list[dict[str, str]],
    model: str | None = None,
    max_model_attempts: int | None = None,
    **kwargs: typing.Any,
) -> typing.Any:
    candidates = get_ai_model_candidates(model)
    if max_model_attempts is not None:
        candidates = candidates[: max(1, max_model_attempts)]
    last_error: Exception | None = None

    for index, candidate in enumerate(candidates):
        try:
            return await ai.chat.completions.create(
                model=candidate,
                messages=messages,
                **kwargs,
            )
        except Exception as exc:
            last_error = exc
            if index >= len(candidates) - 1 or not should_try_next_ai_model(exc):
                raise
            next_candidate = candidates[index + 1]
            AI_PROVIDER_LOGGER.warning(
                "AI model %s failed with %s. Trying fallback model %s.",
                candidate,
                exc.__class__.__name__,
                next_candidate,
            )

    assert last_error is not None
    raise last_error


def normalize_whisper_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized in SUPPORTED_WHISPER_DEVICES:
        return normalized
    TRANSCRIPTION_LOGGER.warning(
        "Unsupported WHISPER_DEVICE=%r. Falling back to %s.",
        device,
        DEFAULT_WHISPER_DEVICE,
    )
    return DEFAULT_WHISPER_DEVICE


def get_transcription_model_settings() -> TranscriptionModelSettings:
    return TranscriptionModelSettings(
        model_name=get_string_env("WHISPER_MODEL_SIZE", DEFAULT_WHISPER_MODEL_SIZE),
        device=normalize_whisper_device(get_string_env("WHISPER_DEVICE", DEFAULT_WHISPER_DEVICE)),
        compute_type=get_string_env("WHISPER_COMPUTE_TYPE", DEFAULT_WHISPER_COMPUTE_TYPE).lower(),
        cpu_threads=read_int_env("WHISPER_CPU_THREADS", 0, min_value=0),
        num_workers=read_int_env("WHISPER_NUM_WORKERS", 1, min_value=1),
    )


def resolve_compute_type(
    requested_compute_type: str,
    device: str,
    gpu: NvidiaGpuInfo | None = None,
) -> str:
    if requested_compute_type == "auto":
        return recommend_whisper_compute_type(device, gpu)
    return requested_compute_type


def make_attempt_settings(
    requested_settings: TranscriptionModelSettings,
    *,
    device: str,
    compute_type: str,
    gpu: NvidiaGpuInfo | None = None,
) -> TranscriptionModelSettings:
    return dataclasses.replace(
        requested_settings,
        device=device,
        compute_type=resolve_compute_type(compute_type, device, gpu),
    )


def create_transcription_model(model_name: str, settings: TranscriptionModelSettings) -> WhisperModel:
    if settings.device == "cuda":
        configure_nvidia_runtime_paths()
    return WhisperModel(
        model_name,
        device=settings.device,
        compute_type=settings.compute_type,
        cpu_threads=settings.cpu_threads,
        num_workers=settings.num_workers,
    )


def build_transcription_load_context(settings: TranscriptionModelSettings) -> TranscriptionLoadContext:
    if settings.device == "cpu":
        TRANSCRIPTION_LOGGER.info("Whisper device forced to CPU by WHISPER_DEVICE=cpu.")
        return TranscriptionLoadContext()

    nvidia_smi = run_nvidia_smi()
    gpu = choose_primary_gpu(nvidia_smi)
    if gpu is not None:
        TRANSCRIPTION_LOGGER.info("NVIDIA GPU detected for Whisper: %s.", format_gpu_info(gpu))
    elif settings.device == "auto":
        reason = nvidia_smi.error or "nvidia-smi did not report an NVIDIA GPU."
        TRANSCRIPTION_LOGGER.warning("WHISPER_DEVICE=auto selected CPU fallback: %s", reason)
        return TranscriptionLoadContext(initial_cpu_fallback_reason=reason)
    else:
        reason = nvidia_smi.error or "nvidia-smi did not report an NVIDIA GPU."
        TRANSCRIPTION_LOGGER.warning(
            "WHISPER_DEVICE=cuda was requested, but GPU discovery was not clean: %s. Trying CUDA anyway.",
            reason,
        )

    ctranslate2_status = check_ctranslate2_cuda()
    if settings.device == "auto" and ctranslate2_status.cuda_device_count == 0:
        reason = "CTranslate2 reports zero CUDA devices."
        TRANSCRIPTION_LOGGER.warning("WHISPER_DEVICE=auto selected CPU fallback: %s", reason)
        return TranscriptionLoadContext(
            gpu=gpu,
            ctranslate2=ctranslate2_status,
            initial_cpu_fallback_reason=reason,
        )

    if ctranslate2_status.error:
        TRANSCRIPTION_LOGGER.info(
            "CTranslate2 CUDA precheck was inconclusive: %s. CUDA will be verified by Whisper initialization.",
            ctranslate2_status.error,
        )
    elif ctranslate2_status.cuda_device_count is not None:
        TRANSCRIPTION_LOGGER.info("CTranslate2 CUDA device count: %s.", ctranslate2_status.cuda_device_count)

    return TranscriptionLoadContext(gpu=gpu, ctranslate2=ctranslate2_status)


def iter_model_load_attempts(
    settings: TranscriptionModelSettings,
    load_context: TranscriptionLoadContext | None = None,
) -> typing.Iterable[tuple[str, TranscriptionModelSettings]]:
    load_context = load_context or build_transcription_load_context(settings)
    model_names = (settings.model_name, *WHISPER_MODEL_FALLBACKS)
    seen_model_names: set[str] = set()

    for model_name in model_names:
        if model_name in seen_model_names:
            continue
        seen_model_names.add(model_name)

        yielded_settings: set[tuple[str, str]] = set()

        if settings.device == "cpu":
            device_compute_candidates = [
                ("cpu", settings.compute_type),
            ]
        elif settings.device == "auto" and load_context.initial_cpu_fallback_reason:
            device_compute_candidates = [
                ("cpu", "int8"),
            ]
        else:
            device_compute_candidates = [
                ("cuda", settings.compute_type),
            ]
            device_compute_candidates.extend(("cuda", compute_type) for compute_type in CUDA_COMPUTE_TYPE_FALLBACKS)
            device_compute_candidates.extend(("cpu", compute_type) for compute_type in CPU_COMPUTE_TYPE_FALLBACKS)

        for device, compute_type in device_compute_candidates:
            attempt_settings = make_attempt_settings(
                settings,
                device=device,
                compute_type=compute_type,
                gpu=load_context.gpu,
            )
            settings_key = (attempt_settings.device, attempt_settings.compute_type)
            if settings_key in yielded_settings:
                continue
            yielded_settings.add(settings_key)
            yield model_name, attempt_settings


def get_transcription_model() -> WhisperModel:
    global _TRANSCRIPTION_BATCHED_MODEL
    global _TRANSCRIPTION_BATCHED_PIPELINE
    global _TRANSCRIPTION_MODEL
    global _TRANSCRIPTION_MODEL_NAME
    global _TRANSCRIPTION_MODEL_SETTINGS
    global _TRANSCRIPTION_RUNTIME_INFO

    settings = get_transcription_model_settings()

    if _TRANSCRIPTION_MODEL is not None and _TRANSCRIPTION_MODEL_SETTINGS == settings:
        return _TRANSCRIPTION_MODEL

    with _TRANSCRIPTION_MODEL_LOCK:
        if _TRANSCRIPTION_MODEL is not None and _TRANSCRIPTION_MODEL_SETTINGS == settings:
            return _TRANSCRIPTION_MODEL

        last_error: Exception | None = None
        last_cuda_error: Exception | None = None
        load_context = build_transcription_load_context(settings)
        for model_name, attempt_settings in iter_model_load_attempts(settings, load_context):
            try:
                _TRANSCRIPTION_MODEL = create_transcription_model(model_name, attempt_settings)
                _TRANSCRIPTION_MODEL_NAME = model_name
                _TRANSCRIPTION_MODEL_SETTINGS = settings
                fallback_reason = None
                if attempt_settings.device == "cpu" and settings.device in {"auto", "cuda"}:
                    if load_context.initial_cpu_fallback_reason:
                        fallback_reason = load_context.initial_cpu_fallback_reason
                    elif last_cuda_error is not None:
                        fallback_reason = f"CUDA model initialization failed: {last_cuda_error}"
                _TRANSCRIPTION_RUNTIME_INFO = TranscriptionRuntimeInfo(
                    requested_settings=settings,
                    active_settings=attempt_settings,
                    model_name=model_name,
                    gpu_name=load_context.gpu.name if load_context.gpu else None,
                    gpu_vram_mb=load_context.gpu.memory_total_mb if load_context.gpu else None,
                    fallback_reason=fallback_reason,
                )
                _TRANSCRIPTION_BATCHED_PIPELINE = None
                _TRANSCRIPTION_BATCHED_MODEL = None
                TRANSCRIPTION_LOGGER.info(
                    "Whisper model loaded: model=%s device=%s compute_type=%s.",
                    model_name,
                    attempt_settings.device,
                    attempt_settings.compute_type,
                )
                if fallback_reason:
                    TRANSCRIPTION_LOGGER.warning("Whisper is using CPU fallback: %s", fallback_reason)
                break
            except Exception as exc:
                last_error = exc
                if attempt_settings.device == "cuda":
                    last_cuda_error = exc
                    TRANSCRIPTION_LOGGER.warning(
                        "Whisper CUDA initialization failed for model=%s compute_type=%s: %s",
                        model_name,
                        attempt_settings.compute_type,
                        exc,
                    )
        else:
            assert last_error is not None
            raise last_error

    return _TRANSCRIPTION_MODEL


def get_active_transcription_runtime_info() -> TranscriptionRuntimeInfo | None:
    return _TRANSCRIPTION_RUNTIME_INFO


def get_effective_transcription_model_settings() -> TranscriptionModelSettings:
    runtime_info = get_active_transcription_runtime_info()
    if runtime_info is not None:
        return runtime_info.active_settings
    settings = get_transcription_model_settings()
    if settings.device == "cpu":
        return make_attempt_settings(settings, device="cpu", compute_type=settings.compute_type)
    return settings


def normalize_transcription_language(language: str | None) -> str | None:
    if language in TRANSCRIPTION_LANGUAGE_PROMPTS:
        return language
    return None


def get_default_transcription_language() -> str | None:
    configured = os.environ.get("WHISPER_DEFAULT_LANGUAGE", "").strip().lower()
    return normalize_transcription_language(configured)


def get_detected_transcription_language(info: typing.Any) -> str | None:
    detected_language = getattr(info, "language", None)
    if not detected_language:
        return None
    return str(detected_language).strip().lower() or None


def map_transcription_language_to_output_language(language: str | None) -> str | None:
    if language is None:
        return None
    return TRANSCRIPTION_TO_OUTPUT_LANGUAGE.get(language.strip().lower())


def should_retry_transcription_as_russian(language_hint: str | None, info: typing.Any) -> bool:
    if language_hint is not None:
        return False
    detected_language = get_detected_transcription_language(info)
    return detected_language in RUSSIAN_RETRY_SOURCE_LANGUAGES


def is_cuda_library_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in CUDA_LIBRARY_ERROR_MARKERS)


def is_unsupported_compute_type_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return all(marker in message for marker in ("compute type", "support"))


def get_cpu_transcription_model_for_retry() -> WhisperModel:
    global _TRANSCRIPTION_BATCHED_MODEL
    global _TRANSCRIPTION_BATCHED_PIPELINE
    global _TRANSCRIPTION_MODEL
    global _TRANSCRIPTION_MODEL_NAME
    global _TRANSCRIPTION_MODEL_SETTINGS
    global _TRANSCRIPTION_RUNTIME_INFO

    requested_settings = get_transcription_model_settings()
    cpu_settings = dataclasses.replace(requested_settings, device="cpu", compute_type="int8")
    model_name = _TRANSCRIPTION_MODEL_NAME or requested_settings.model_name

    with _TRANSCRIPTION_MODEL_LOCK:
        _TRANSCRIPTION_MODEL = create_transcription_model(model_name, cpu_settings)
        _TRANSCRIPTION_MODEL_NAME = model_name
        _TRANSCRIPTION_MODEL_SETTINGS = requested_settings
        _TRANSCRIPTION_RUNTIME_INFO = TranscriptionRuntimeInfo(
            requested_settings=requested_settings,
            active_settings=cpu_settings,
            model_name=model_name,
            gpu_name=(_TRANSCRIPTION_RUNTIME_INFO.gpu_name if _TRANSCRIPTION_RUNTIME_INFO is not None else None),
            gpu_vram_mb=(_TRANSCRIPTION_RUNTIME_INFO.gpu_vram_mb if _TRANSCRIPTION_RUNTIME_INFO is not None else None),
            fallback_reason="CUDA/cuBLAS libraries were not available during transcription.",
        )
        _TRANSCRIPTION_BATCHED_PIPELINE = None
        _TRANSCRIPTION_BATCHED_MODEL = None

    return _TRANSCRIPTION_MODEL


def get_transcription_model_for_compute_retry() -> WhisperModel:
    global _TRANSCRIPTION_BATCHED_MODEL
    global _TRANSCRIPTION_BATCHED_PIPELINE
    global _TRANSCRIPTION_MODEL
    global _TRANSCRIPTION_MODEL_NAME
    global _TRANSCRIPTION_MODEL_SETTINGS
    global _TRANSCRIPTION_RUNTIME_INFO

    requested_settings = get_transcription_model_settings()
    model_name = _TRANSCRIPTION_MODEL_NAME or requested_settings.model_name
    candidates: list[TranscriptionModelSettings] = []

    if requested_settings.device != "cpu":
        active_gpu = None
        if _TRANSCRIPTION_RUNTIME_INFO is not None and _TRANSCRIPTION_RUNTIME_INFO.gpu_name is not None:
            active_gpu = NvidiaGpuInfo(
                name=_TRANSCRIPTION_RUNTIME_INFO.gpu_name,
                memory_total_mb=_TRANSCRIPTION_RUNTIME_INFO.gpu_vram_mb,
            )
        for compute_type in CUDA_COMPUTE_TYPE_FALLBACKS:
            candidates.append(
                make_attempt_settings(
                    requested_settings,
                    device="cuda",
                    compute_type=compute_type,
                    gpu=active_gpu,
                )
            )

    candidates.append(dataclasses.replace(requested_settings, device="cpu", compute_type="int8"))

    last_error: Exception | None = None
    with _TRANSCRIPTION_MODEL_LOCK:
        for candidate_settings in candidates:
            if (
                _TRANSCRIPTION_MODEL is not None
                and _TRANSCRIPTION_RUNTIME_INFO is not None
                and _TRANSCRIPTION_RUNTIME_INFO.active_settings == candidate_settings
            ):
                continue
            try:
                _TRANSCRIPTION_MODEL = create_transcription_model(model_name, candidate_settings)
                _TRANSCRIPTION_MODEL_NAME = model_name
                _TRANSCRIPTION_MODEL_SETTINGS = requested_settings
                fallback_reason = None
                if candidate_settings.device == "cpu":
                    fallback_reason = "No supported CUDA compute type was available."
                _TRANSCRIPTION_RUNTIME_INFO = TranscriptionRuntimeInfo(
                    requested_settings=requested_settings,
                    active_settings=candidate_settings,
                    model_name=model_name,
                    gpu_name=active_gpu.name if active_gpu else None,
                    gpu_vram_mb=active_gpu.memory_total_mb if active_gpu else None,
                    fallback_reason=fallback_reason,
                )
                _TRANSCRIPTION_BATCHED_PIPELINE = None
                _TRANSCRIPTION_BATCHED_MODEL = None
                return _TRANSCRIPTION_MODEL
            except Exception as exc:
                last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("No transcription compute fallback is available.")


def whisper_batched_transcription_enabled() -> bool:
    if BatchedInferencePipeline is None:
        return False

    settings = get_effective_transcription_model_settings()
    if settings.device == "cpu":
        return False

    return env_flag_enabled("WHISPER_BATCHED", False)


def get_whisper_batch_size() -> int:
    configured = os.environ.get("WHISPER_BATCH_SIZE", "").strip().lower()
    if configured == "auto":
        runtime_info = get_active_transcription_runtime_info()
        if runtime_info is not None and runtime_info.gpu_name:
            return recommend_whisper_batch_size(
                NvidiaGpuInfo(
                    name=runtime_info.gpu_name,
                    memory_total_mb=runtime_info.gpu_vram_mb,
                )
            )
    return read_int_env("WHISPER_BATCH_SIZE", DEFAULT_WHISPER_BATCH_SIZE, min_value=1, max_value=64)


def get_whisper_beam_size() -> int:
    return read_int_env("WHISPER_BEAM_SIZE", DEFAULT_WHISPER_BEAM_SIZE, min_value=1, max_value=10)


def get_whisper_best_of() -> int:
    return read_int_env("WHISPER_BEST_OF", DEFAULT_WHISPER_BEST_OF, min_value=1, max_value=10)


def get_whisper_vad_filter() -> bool:
    return env_flag_enabled("WHISPER_VAD_FILTER", False)


def get_transcription_runner(model: WhisperModel) -> typing.Any:
    global _TRANSCRIPTION_BATCHED_MODEL
    global _TRANSCRIPTION_BATCHED_PIPELINE

    if not whisper_batched_transcription_enabled():
        return model

    if not isinstance(model, WhisperModel):
        return model

    if _TRANSCRIPTION_BATCHED_PIPELINE is None or _TRANSCRIPTION_BATCHED_MODEL is not model:
        try:
            _TRANSCRIPTION_BATCHED_PIPELINE = BatchedInferencePipeline(model)
            _TRANSCRIPTION_BATCHED_MODEL = model
        except Exception:
            return model

    return _TRANSCRIPTION_BATCHED_PIPELINE


def use_whisper_initial_prompt() -> bool:
    return env_flag_explicitly_enabled("WHISPER_INITIAL_PROMPT", False)


def condition_on_previous_text_enabled() -> bool:
    return env_flag_enabled("WHISPER_CONDITION_ON_PREVIOUS_TEXT", False)


def build_transcribe_kwargs(
    language_hint: str | None = None,
    *,
    use_initial_prompt: bool = True,
    condition_on_previous_text: bool | None = None,
) -> dict[str, typing.Any]:
    transcribe_kwargs: dict[str, typing.Any] = {
        "beam_size": get_whisper_beam_size(),
        "best_of": get_whisper_best_of(),
        "temperature": 0.0,
        "condition_on_previous_text": (
            condition_on_previous_text_enabled() if condition_on_previous_text is None else condition_on_previous_text
        ),
        "vad_filter": get_whisper_vad_filter(),
    }
    if language_hint is not None:
        transcribe_kwargs["language"] = language_hint
        if use_initial_prompt and use_whisper_initial_prompt():
            transcribe_kwargs["initial_prompt"] = TRANSCRIPTION_LANGUAGE_PROMPTS[language_hint]
    return transcribe_kwargs


def audio_preprocessing_enabled() -> bool:
    return env_flag_enabled("WHISPER_PREPROCESS_AUDIO", True)


def get_audio_preprocess_filter() -> str:
    return os.environ.get("WHISPER_PREPROCESS_FILTER", DEFAULT_AUDIO_PREPROCESS_FILTER).strip()


def preprocess_audio_for_transcription(audio_path: str) -> str | None:
    if not audio_preprocessing_enabled():
        return None

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as prepared_file:
        prepared_path = prepared_file.name

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        audio_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    preprocess_filter = get_audio_preprocess_filter()
    if preprocess_filter:
        command.extend(["-af", preprocess_filter])
    command.append(prepared_path)

    success = False
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        if result.returncode != 0 or not os.path.getsize(prepared_path):
            return None
        success = True
        return prepared_path
    except Exception:
        return None
    finally:
        if not success and os.path.exists(prepared_path):
            try:
                os.unlink(prepared_path)
            except OSError:
                pass


def transcribe_audio_once(
    model: WhisperModel,
    audio_path: str,
    language_hint: str | None = None,
    *,
    use_initial_prompt: bool = True,
    condition_on_previous_text: bool | None = None,
) -> tuple[str, typing.Any]:
    runner = get_transcription_runner(model)
    transcribe_kwargs = build_transcribe_kwargs(
        language_hint,
        use_initial_prompt=use_initial_prompt,
        condition_on_previous_text=condition_on_previous_text,
    )
    if runner is not model:
        transcribe_kwargs["batch_size"] = get_whisper_batch_size()
        transcribe_kwargs["vad_filter"] = True

    segments, info = runner.transcribe(audio_path, **transcribe_kwargs)

    transcript_parts: list[str] = []

    for segment in segments:
        segment_text = str(getattr(segment, "text", "") or "").strip()
        if segment_text:
            transcript_parts.append(segment_text)

    return " ".join(transcript_parts).strip(), info


async def transcribe_audio_once_with_cuda_fallback(
    model: WhisperModel,
    audio_path: str,
    language_hint: str | None,
    logger: Logger,
    *,
    use_initial_prompt: bool = True,
    condition_on_previous_text: bool | None = None,
) -> tuple[str, typing.Any]:
    try:
        return await asyncio.to_thread(
            transcribe_audio_once,
            model,
            audio_path,
            language_hint,
            use_initial_prompt=use_initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
        )
    except Exception as exc:
        if is_unsupported_compute_type_error(exc):
            await logger.partial_result(
                "The selected Whisper compute type is not supported by this backend. "
                "Retrying with a safer compute mode..."
            )
            fallback_model = await asyncio.to_thread(get_transcription_model_for_compute_retry)
            return await asyncio.to_thread(
                transcribe_audio_once,
                fallback_model,
                audio_path,
                language_hint,
                use_initial_prompt=use_initial_prompt,
                condition_on_previous_text=condition_on_previous_text,
            )

        if not is_cuda_library_error(exc):
            raise

        await logger.partial_result(
            "CUDA/cuBLAS libraries are not available. Retrying local Whisper transcription on CPU."
        )
        cpu_model = await asyncio.to_thread(get_cpu_transcription_model_for_retry)
        return await asyncio.to_thread(
            transcribe_audio_once,
            cpu_model,
            audio_path,
            language_hint,
            use_initial_prompt=use_initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
        )


def language_predetection_enabled() -> bool:
    return env_flag_enabled("WHISPER_PREDETECT_LANGUAGE", False)


def get_language_detection_seconds() -> int:
    default_seconds = max(1, LANGUAGE_DETECTION_MS // 1000)
    return read_int_env("WHISPER_LANGUAGE_DETECTION_SECONDS", default_seconds, min_value=1, max_value=120)


def detect_transcription_language_once(model: WhisperModel, audio_path: str) -> str | None:
    detection_seconds = get_language_detection_seconds()
    segments, info = model.transcribe(
        audio_path,
        beam_size=1,
        vad_filter=True,
        clip_timestamps=f"0,{detection_seconds}",
        language_detection_segments=1,
    )

    for _segment in segments:
        break

    return get_detected_transcription_language(info)


async def choose_transcription_language_hint(
    model: WhisperModel,
    audio_path: str,
    logger: Logger,
) -> tuple[str | None, str | None]:
    if not language_predetection_enabled():
        return None, None

    await logger.partial_result("Detecting audio language before full transcription...")
    try:
        detected_language = await asyncio.to_thread(detect_transcription_language_once, model, audio_path)
    except Exception:
        await logger.partial_result("Fast audio language detection failed; continuing with auto transcription.")
        return None, None

    language_hint = map_transcription_language_to_output_language(detected_language)
    if language_hint is not None:
        await logger.partial_result(
            f"Using {language_hint} transcription hint from detected audio language: {detected_language}."
        )
    return language_hint, detected_language


def strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def normalize_latex_text(text: str) -> str:
    normalized = strip_markdown_fences(text)

    substitutions = [
        (r"\*\*(.+?)\*\*", r"\\textbf{\1}"),
        (r"__(.+?)__", r"\\textbf{\1}"),
    ]

    for pattern, replacement in substitutions:
        normalized = re.sub(pattern, replacement, normalized, flags=re.DOTALL)

    normalized = normalized.replace("–", "--")
    normalized = normalized.replace("—", "---")
    normalized = normalized.replace("\u00a0", " ")
    normalized = normalized.replace("\u202f", " ")
    normalized = normalized.replace("\u2011", "-")
    normalized = normalized.replace("\u2013", "--")
    normalized = normalized.replace("\u2014", "---")
    normalized = normalized.replace("\u2212", "-")
    normalized = normalized.replace("вЂ‘", "-")
    return normalized.strip()


def normalize_repetition_text(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"[^\w\sа-яё'-]+", " ", normalized, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", normalized).strip()


def text_contains_refusal(text: str) -> bool:
    normalized = normalize_repetition_text(text)
    return any(marker in normalized for marker in REFUSAL_MARKERS)


def text_repeats_prompt_marker(text: str) -> bool:
    normalized = normalize_repetition_text(text)
    return any(normalized.count(marker) >= 2 for marker in TRANSCRIPTION_PROMPT_ECHO_MARKERS)


def text_has_repeated_sentence_loop(text: str) -> bool:
    sentences = [normalize_repetition_text(sentence) for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip())]
    sentences = [sentence for sentence in sentences if len(sentence) >= 12]
    if not sentences:
        return False

    sentence, repeats = Counter(sentences).most_common(1)[0]
    if repeats < 5:
        return False

    repeated_chars = repeats * len(sentence)
    total_chars = max(1, sum(len(item) for item in sentences))
    return repeated_chars / total_chars >= 0.6


def transcript_looks_degenerate(transcript: str) -> bool:
    text = transcript.strip()
    if not text:
        return True
    return text_contains_refusal(text) or text_repeats_prompt_marker(text) or text_has_repeated_sentence_loop(text)


def validate_transcript_text(transcript: str) -> None:
    if transcript_looks_degenerate(transcript):
        raise RuntimeError(
            "Transcription failed: Whisper returned repeated prompt/refusal text instead of speech. "
            "Check that the file contains audible speech and retry with the current transcription settings."
        )


def parse_summary_response(response_text: str) -> Summary:
    cleaned = normalize_latex_text(response_text)
    if text_contains_refusal(cleaned):
        raise RuntimeError("The AI provider refused to summarize the transcript.")

    parts = [part.strip() for part in cleaned.split("\n\n", maxsplit=1)]

    title = parts[0] if parts else "Lecture Summary"
    abstract = parts[1] if len(parts) > 1 else ""

    title = re.sub(r"^(title|заголовок)\s*:\s*", "", title, flags=re.IGNORECASE)
    abstract = re.sub(r"^(abstract|summary|аннотация)\s*:\s*", "", abstract, flags=re.IGNORECASE)

    if not abstract:
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if len(lines) >= 2:
            title = re.sub(r"^(title|заголовок)\s*:\s*", "", lines[0], flags=re.IGNORECASE)
            abstract = " ".join(lines[1:])
        else:
            abstract = cleaned

    return Summary(title=title.strip(), abstract=abstract.strip())


async def transcribe_audio_with_metadata(
    audio_file: bytes,
    logger: Logger = Logger(),
    filename: str | None = None,
    mime_type: str | None = None,
    language: str | None = None,
) -> TranscriptionResult:
    """Transcribe audio to text using local Whisper model."""
    await logger.stage("transcribing", 0)
    await logger.partial_result("Loading transcription model...")
    model = await asyncio.to_thread(get_transcription_model)
    runtime_info = get_active_transcription_runtime_info()
    if runtime_info is not None:
        if runtime_info.gpu_name:
            gpu_info = NvidiaGpuInfo(runtime_info.gpu_name, runtime_info.gpu_vram_mb)
            await logger.partial_result(f"NVIDIA GPU detected: {format_gpu_info(gpu_info)}.")
        await logger.partial_result(
            "Whisper transcription device: "
            f"{runtime_info.active_settings.device}, compute_type={runtime_info.active_settings.compute_type}."
        )
        if runtime_info.fallback_reason:
            await logger.partial_result(
                f"Whisper GPU fallback: {runtime_info.fallback_reason} Running transcription on CPU; this can be slow."
            )
    await logger.partial_result(
        "Stage around 19% is local Whisper transcription. "
        "If CPU mode is selected, large-v3 audio recognition can take a long time."
    )
    language_hint = normalize_transcription_language(language)
    if language_hint is None:
        language_hint = get_default_transcription_language()
        if language_hint is not None:
            await logger.partial_result(f"Using default transcription language hint: {language_hint}.")

    suffix = guess_audio_suffix(filename=filename, mime_type=mime_type)

    # Save audio to a temp file with its original suffix so ffmpeg/Whisper can decode it correctly.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(audio_file)
        temp_path = temp_file.name

    temp_paths = [temp_path]
    transcription_path = temp_path
    try:
        prepared_audio_path = await asyncio.to_thread(preprocess_audio_for_transcription, temp_path)
        if prepared_audio_path is not None:
            temp_paths.append(prepared_audio_path)
            transcription_path = prepared_audio_path
            await logger.partial_result("Prepared normalized mono audio for transcription.")

        pre_detected_language: str | None = None
        if language_hint is None:
            language_hint, pre_detected_language = await choose_transcription_language_hint(
                model,
                transcription_path,
                logger,
            )

        await logger.partial_result("Running local Whisper transcription...")
        transcript, info = await transcribe_audio_once_with_cuda_fallback(
            model,
            transcription_path,
            language_hint,
            logger,
        )
        detected_language = get_detected_transcription_language(info) or pre_detected_language or language_hint

        if transcript_looks_degenerate(transcript):
            await logger.partial_result(
                "Whisper returned repeated prompt/refusal text. Retrying transcription without prompt context..."
            )
            transcript, info = await transcribe_audio_once_with_cuda_fallback(
                model,
                transcription_path,
                language_hint,
                logger,
                use_initial_prompt=False,
                condition_on_previous_text=False,
            )
            detected_language = get_detected_transcription_language(info) or pre_detected_language or language_hint

        if should_retry_transcription_as_russian(language_hint, info):
            await logger.partial_result(
                "Whisper detected a Ukrainian/Belarusian language variant; retrying transcription as Russian."
            )
            await logger.stage("transcribing", 0)
            transcript, info = await transcribe_audio_once_with_cuda_fallback(model, transcription_path, "ru", logger)
            detected_language = get_detected_transcription_language(info)

        validate_transcript_text(transcript)

        await logger.stage("transcript_ready", 100)
        if detected_language:
            await logger.partial_result(f"Whisper audio language: {detected_language}")
        await logger.partial_result(f"<b>Transcription complete:</b> {len(transcript)} characters")
        await logger.file("transcript", transcript, Logger.FileType.TEXT)

        return TranscriptionResult(text=transcript.strip(), language=detected_language)
    finally:
        # Clean up temp file
        for path in temp_paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


async def transcribe_audio(
    audio_file: bytes,
    logger: Logger = Logger(),
    filename: str | None = None,
    mime_type: str | None = None,
    language: str | None = None,
) -> str:
    result = await transcribe_audio_with_metadata(
        audio_file,
        logger,
        filename=filename,
        mime_type=mime_type,
        language=language,
    )
    return result.text


async def detect_language_from_text(text: str, ai: openai.AsyncOpenAI) -> str:
    """Detect language from text using AI."""
    prompt = f"Detect the language of this text. Return only 'ru' for Russian or 'en' for English:\n\n{text[:1000]}"

    response = await create_chat_completion(
        ai, messages=[{"role": "user", "content": prompt}], max_tokens=10, temperature=0.1
    )

    detected = (response.choices[0].message.content or "").strip().lower()
    if detected not in ("ru", "en"):
        return "en"  # Default to English
    return detected


async def detect_language(
    audio_file: bytes,
    ai: openai.AsyncOpenAI,
    filename: str | None = None,
    mime_type: str | None = None,
) -> str:
    result = await transcribe_audio_with_metadata(audio_file, filename=filename, mime_type=mime_type)
    detected_language = map_transcription_language_to_output_language(result.language)
    if detected_language is not None:
        return detected_language
    return await detect_language_from_text(result.text, ai)


async def make_summary_from_transcript(
    transcript: str,
    ai: openai.AsyncOpenAI,
    language: str,
    logger: Logger = Logger(),
    detail_level: str = "standard",
) -> Summary:
    with open(
        pathlib.Path(__file__).parent / "prompts/summary_prompt.txt", encoding="utf-8", errors="replace"
    ) as prompt_file:
        prompt = prompt_file.read()
    prompt = prompt.replace("<OUTPUT_LANGUAGE>", LANGUAGE_NAMES[language])

    await logger.stage("summary", 10)
    summary_transcript, transcript_was_trimmed = build_summary_transcript_context(transcript)
    if transcript_was_trimmed:
        await logger.partial_result(
            "Shortened transcript context for the title/abstract request to fit provider prompt limits."
        )

    response = await create_chat_completion(
        ai,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "system",
                "content": (
                    f"Summary detail level: {detail_level}. "
                    f"{DETAIL_LEVEL_PROMPTS.get(detail_level, DETAIL_LEVEL_PROMPTS['standard'])}"
                ),
            },
            {
                "role": "user",
                "content": f"Create a summary from this lecture transcript:\n\n{summary_transcript}",
            },
        ],
        temperature=0.3,
        max_tokens=450,
    )
    response_text = response.choices[0].message.content or ""
    await logger.file("summary", response_text, Logger.FileType.TEXT)

    summary = parse_summary_response(response_text)

    await logger.stage("summary", 70)
    await logger.partial_result(f"<b>The topic of the lecture:</b>\n{summary.title}")
    await logger.stage("summary", 100)
    await logger.partial_result(f"<b>The abstract of the lecture:</b>\n{summary.abstract}")
    return summary


async def make_summary(
    audio_file: bytes,
    ai: openai.AsyncOpenAI,
    language: str,
    logger: Logger = Logger(),
    detail_level: str = "standard",
    filename: str | None = None,
    mime_type: str | None = None,
) -> Summary:
    transcript = await transcribe_audio(
        audio_file,
        logger,
        filename=filename,
        mime_type=mime_type,
        language=language,
    )
    return await make_summary_from_transcript(transcript, ai, language, logger, detail_level=detail_level)


def trim_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    marker = "\n\n[... transcript middle omitted for length ...]\n\n"
    if max_chars <= len(marker) + 2:
        return text[:max_chars].rstrip()

    available_chars = max_chars - len(marker)
    head_chars = available_chars // 2
    tail_chars = available_chars - head_chars
    return text[:head_chars].rstrip() + marker + text[-tail_chars:].lstrip()


def get_summary_transcript_context_chars() -> int:
    return read_int_env(
        "SUMMARY_TRANSCRIPT_CONTEXT_CHARS",
        DEFAULT_SUMMARY_TRANSCRIPT_CONTEXT_CHARS,
        min_value=0,
        max_value=120000,
    )


def build_summary_transcript_context(transcript: str) -> tuple[str, bool]:
    transcript_context_chars = get_summary_transcript_context_chars()
    if transcript_context_chars <= 0 or len(transcript) <= transcript_context_chars:
        return transcript, False
    return trim_middle(transcript, transcript_context_chars), True


async def postprocess_summary(
    tex_content: str,
    ai: openai.AsyncOpenAI,
    language: str,
    logger: Logger = Logger(),
    source_transcript: str | None = None,
) -> str:
    """
    Postprocess the LaTeX summary with a final academic editing pass.

    Args:
        tex_content: The complete LaTeX document content
        ai: chat-completion client
        language: Language of the summary ('ru' or 'en')
        logger: Logger for tracking progress
        source_transcript: Cleaned transcript for checking omissions and ASR artifacts

    Returns:
        Enhanced LaTeX content with improved structure and wording
    """
    with open(
        pathlib.Path(__file__).parent / "prompts/postprocess_prompt.txt", encoding="utf-8", errors="replace"
    ) as prompt_file:
        prompt = prompt_file.read()

    prompt = prompt.replace("<OUTPUT_LANGUAGE>", LANGUAGE_NAMES[language])

    await logger.stage("postprocess", 10)
    await logger.partial_result("<b>Starting final academic editing:</b> improving structure and consistency...")

    user_content_parts = []
    if source_transcript:
        transcript_context_chars = read_int_env(
            "POSTPROCESS_TRANSCRIPT_CONTEXT_CHARS",
            DEFAULT_POSTPROCESS_TRANSCRIPT_CONTEXT_CHARS,
            min_value=0,
            max_value=120000,
        )
        if transcript_context_chars > 0:
            user_content_parts.append(
                "Cleaned source transcript for fact checking and recovering missed lecture details:\n\n"
                f"{trim_middle(source_transcript, transcript_context_chars)}"
            )

    user_content_parts.append(f"Current LaTeX draft to improve:\n\n{tex_content}")

    response = await create_chat_completion(
        ai,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "\n\n---\n\n".join(user_content_parts)},
        ],
        temperature=0.3,
    )

    enhanced_content = normalize_latex_text(response.choices[0].message.content or "")

    # Basic validation: check that essential LaTeX structure is preserved
    required_elements = [r"\documentclass", r"\begin{document}", r"\end{document}"]

    missing_elements = [elem for elem in required_elements if elem not in enhanced_content]

    if missing_elements:
        await logger.stage("postprocess", 100)
        await logger.partial_result(
            f"<b>Postprocessing validation failed:</b> missing {', '.join(missing_elements)}. Using original version."
        )
        return tex_content

    # Check that document is not truncated
    if not enhanced_content.strip().endswith(r"\end{document}"):
        await logger.stage("postprocess", 100)
        await logger.partial_result(
            "<b>Postprocessing warning:</b> document appears truncated. Using original version."
        )
        return tex_content

    await logger.file("lecture_postprocessed", enhanced_content, Logger.FileType.TEX)
    await logger.stage("postprocess", 100)
    await logger.partial_result("<b>Postprocessing complete:</b> LaTeX and consistency checks finished")

    return enhanced_content
