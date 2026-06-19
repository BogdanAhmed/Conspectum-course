import dataclasses
import os
from pathlib import Path
import platform
import shutil
import site
import subprocess
import sys
import tempfile
import typing
import wave

NVIDIA_SMI_TIMEOUT_SECONDS = 8
_DLL_DIRECTORY_HANDLES: list[typing.Any] = []


@dataclasses.dataclass(frozen=True)
class NvidiaGpuInfo:
    name: str
    memory_total_mb: int | None = None

    @property
    def memory_total_gb(self) -> float | None:
        if self.memory_total_mb is None:
            return None
        return round(self.memory_total_mb / 1024, 1)


@dataclasses.dataclass(frozen=True)
class NvidiaSmiResult:
    command_available: bool
    returncode: int | None = None
    gpus: tuple[NvidiaGpuInfo, ...] = ()
    stdout: str = ""
    stderr: str = ""
    error: str | None = None

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpus)


@dataclasses.dataclass(frozen=True)
class CTranslate2CudaStatus:
    installed: bool
    cuda_device_count: int | None = None
    error: str | None = None

    @property
    def cuda_available(self) -> bool:
        return self.cuda_device_count is not None and self.cuda_device_count > 0


@dataclasses.dataclass(frozen=True)
class FasterWhisperCudaProbe:
    attempted: bool
    ok: bool
    model_name: str
    compute_type: str
    transcribe_checked: bool = False
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class GpuDiagnostics:
    python_version: str
    os: str
    nvidia_smi: NvidiaSmiResult
    primary_gpu: NvidiaGpuInfo | None
    ctranslate2: CTranslate2CudaStatus
    recommended_compute_type: str
    recommended_batch_size: int


def parse_nvidia_smi_query_output(output: str) -> tuple[NvidiaGpuInfo, ...]:
    gpus: list[NvidiaGpuInfo] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = [part.strip() for part in line.rsplit(",", maxsplit=1)]
        name = parts[0]
        memory_total_mb: int | None = None
        if len(parts) == 2:
            try:
                memory_total_mb = int(parts[1])
            except ValueError:
                memory_total_mb = None

        if name:
            gpus.append(NvidiaGpuInfo(name=name, memory_total_mb=memory_total_mb))
    return tuple(gpus)


def run_nvidia_smi(timeout: int = NVIDIA_SMI_TIMEOUT_SECONDS) -> NvidiaSmiResult:
    if shutil.which("nvidia-smi") is None:
        return NvidiaSmiResult(
            command_available=False,
            error="nvidia-smi was not found on PATH.",
        )

    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except Exception as exc:
        return NvidiaSmiResult(
            command_available=True,
            error=f"nvidia-smi failed to run: {exc}",
        )

    gpus = parse_nvidia_smi_query_output(result.stdout) if result.returncode == 0 else ()
    return NvidiaSmiResult(
        command_available=True,
        returncode=result.returncode,
        gpus=gpus,
        stdout=result.stdout.strip(),
        stderr=result.stderr.strip(),
        error=None if result.returncode == 0 else result.stderr.strip() or "nvidia-smi returned a non-zero exit code.",
    )


def choose_primary_gpu(result: NvidiaSmiResult) -> NvidiaGpuInfo | None:
    if not result.gpus:
        return None
    return max(result.gpus, key=lambda gpu: gpu.memory_total_mb or 0)


def recommend_whisper_compute_type(device: str, gpu: NvidiaGpuInfo | None = None) -> str:
    del gpu
    if device == "cuda":
        return "float16"
    return "int8"


def recommend_whisper_batch_size(gpu: NvidiaGpuInfo | None = None) -> int:
    if gpu is None or gpu.memory_total_mb is None:
        return 8
    if gpu.memory_total_mb < 8 * 1024:
        return 4
    if gpu.memory_total_mb < 16 * 1024:
        return 8
    return 16


def iter_site_package_paths() -> typing.Iterable[Path]:
    seen: set[Path] = set()
    candidates = [*site.getsitepackages(), site.getusersitepackages()]
    for candidate in candidates:
        path = Path(candidate)
        if path in seen:
            continue
        seen.add(path)
        yield path


def find_nvidia_runtime_dirs() -> tuple[Path, ...]:
    runtime_dirs: list[Path] = []
    for site_packages in iter_site_package_paths():
        nvidia_root = site_packages / "nvidia"
        for relative_dir in (
            "cublas/bin",
            "cudnn/bin",
            "cuda_nvrtc/bin",
            "cuda_runtime/bin",
        ):
            candidate = nvidia_root / relative_dir
            if candidate.exists():
                runtime_dirs.append(candidate)
    return tuple(runtime_dirs)


def configure_nvidia_runtime_paths() -> tuple[str, ...]:
    runtime_dirs = find_nvidia_runtime_dirs()
    if not runtime_dirs:
        return ()

    current_path_parts = os.environ.get("PATH", "").split(os.pathsep)
    current_path = {part.lower() for part in current_path_parts if part}
    added: list[str] = []

    for runtime_dir in runtime_dirs:
        runtime_dir_text = str(runtime_dir)
        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(runtime_dir_text))
            except OSError:
                pass

        if runtime_dir_text.lower() not in current_path:
            added.append(runtime_dir_text)
            current_path.add(runtime_dir_text.lower())

    if added:
        os.environ["PATH"] = os.pathsep.join([*added, *current_path_parts])

    return tuple(str(path) for path in runtime_dirs)


def check_ctranslate2_cuda() -> CTranslate2CudaStatus:
    configure_nvidia_runtime_paths()
    try:
        import ctranslate2
    except Exception as exc:
        return CTranslate2CudaStatus(installed=False, error=f"ctranslate2 import failed: {exc}")

    get_cuda_device_count = getattr(ctranslate2, "get_cuda_device_count", None)
    if not callable(get_cuda_device_count):
        return CTranslate2CudaStatus(
            installed=True,
            error="ctranslate2 does not expose get_cuda_device_count; CUDA will be verified by model initialization.",
        )

    try:
        return CTranslate2CudaStatus(
            installed=True,
            cuda_device_count=int(get_cuda_device_count()),
        )
    except Exception as exc:
        return CTranslate2CudaStatus(installed=True, error=f"ctranslate2 CUDA check failed: {exc}")


def create_probe_audio_file() -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name

    with wave.open(temp_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16000)

    return temp_path


def probe_faster_whisper_cuda(
    whisper_model_class: type,
    *,
    model_name: str = "tiny",
    compute_type: str = "float16",
) -> FasterWhisperCudaProbe:
    configure_nvidia_runtime_paths()
    audio_path: str | None = None
    try:
        model = whisper_model_class(model_name, device="cuda", compute_type=compute_type)
        audio_path = create_probe_audio_file()
        segments, _info = model.transcribe(
            audio_path,
            beam_size=1,
            best_of=1,
            vad_filter=False,
            language="en",
        )
        for _segment in segments:
            break
    except Exception as exc:
        return FasterWhisperCudaProbe(
            attempted=True,
            ok=False,
            model_name=model_name,
            compute_type=compute_type,
            transcribe_checked=audio_path is not None,
            error=str(exc),
        )
    finally:
        if audio_path is not None:
            try:
                os.unlink(audio_path)
            except OSError:
                pass

    return FasterWhisperCudaProbe(
        attempted=True,
        ok=True,
        model_name=model_name,
        compute_type=compute_type,
        transcribe_checked=True,
    )


def collect_gpu_diagnostics(
    *,
    nvidia_smi_timeout: int = NVIDIA_SMI_TIMEOUT_SECONDS,
) -> GpuDiagnostics:
    nvidia_smi = run_nvidia_smi(timeout=nvidia_smi_timeout)
    primary_gpu = choose_primary_gpu(nvidia_smi)
    device = "cuda" if primary_gpu is not None else "cpu"
    return GpuDiagnostics(
        python_version=sys.version.replace("\n", " "),
        os=f"{platform.system()} {platform.release()} ({platform.platform()})",
        nvidia_smi=nvidia_smi,
        primary_gpu=primary_gpu,
        ctranslate2=check_ctranslate2_cuda(),
        recommended_compute_type=recommend_whisper_compute_type(device, primary_gpu),
        recommended_batch_size=recommend_whisper_batch_size(primary_gpu),
    )


def format_gpu_info(gpu: NvidiaGpuInfo | None) -> str:
    if gpu is None:
        return "none"
    if gpu.memory_total_mb is None:
        return gpu.name
    return f"{gpu.name} ({gpu.memory_total_gb} GB VRAM)"


def summarize_probe(probe: FasterWhisperCudaProbe | None) -> str:
    if probe is None or not probe.attempted:
        return "not attempted"
    if probe.ok:
        suffix = ", transcribe ok" if probe.transcribe_checked else ""
        return f"ok ({probe.model_name}, compute_type={probe.compute_type}{suffix})"
    suffix = ", transcribe attempted" if probe.transcribe_checked else ""
    return f"failed ({probe.model_name}, compute_type={probe.compute_type}{suffix}): {probe.error}"


def describe_nvidia_smi(result: NvidiaSmiResult) -> str:
    if not result.command_available:
        return result.error or "nvidia-smi is not available."
    if result.returncode != 0:
        return result.error or "nvidia-smi returned a non-zero exit code."
    if not result.gpus:
        return "nvidia-smi ran, but no NVIDIA GPU was reported."
    return "; ".join(format_gpu_info(gpu) for gpu in result.gpus)


def ctranslate2_cuda_ready(status: CTranslate2CudaStatus) -> bool | None:
    if status.cuda_device_count is None:
        return None
    return status.cuda_device_count > 0


def get_probe_model_class(module_name: str = "faster_whisper") -> type | None:
    try:
        module = __import__(module_name, fromlist=["WhisperModel"])
    except Exception:
        return None
    model_class = getattr(module, "WhisperModel", None)
    if isinstance(model_class, type):
        return typing.cast(type, model_class)
    return None
