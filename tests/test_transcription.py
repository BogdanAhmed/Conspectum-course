import asyncio
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

if "faster_whisper" not in sys.modules:
    fake_faster_whisper = types.ModuleType("faster_whisper")

    class DummyWhisperModel:
        def __init__(self, *args, **kwargs):
            return

    fake_faster_whisper.WhisperModel = DummyWhisperModel
    sys.modules["faster_whisper"] = fake_faster_whisper

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from conspectum.summary import build_transcribe_kwargs  # noqa: E402
from conspectum.summary import get_default_transcription_language  # noqa: E402
from conspectum.summary import get_transcription_model_settings  # noqa: E402
from conspectum.summary import get_whisper_beam_size  # noqa: E402
from conspectum.summary import get_whisper_best_of  # noqa: E402
from conspectum.summary import language_predetection_enabled  # noqa: E402
from conspectum.summary import preprocess_audio_for_transcription  # noqa: E402
from conspectum.summary import transcribe_audio  # noqa: E402
from conspectum.summary import whisper_batched_transcription_enabled  # noqa: E402


class FakeSegment:
    def __init__(self, text: str, end: float):
        self.text = text
        self.end = end


class FakeInfo:
    def __init__(self, language: str, duration: float = 10.0):
        self.language = language
        self.duration = duration


class FakeWhisperModel:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def transcribe(self, _audio_path: str, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("language") == "ru":
            return [FakeSegment("Почти все продукты", 5.0), FakeSegment("заявки на кредит", 10.0)], FakeInfo("ru")
        return [FakeSegment("Почті все продукта", 10.0)], FakeInfo("uk")


class CudaLibraryFailureModel:
    def transcribe(self, _audio_path: str, **_kwargs):
        raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")


class UnsupportedComputeTypeFailureModel:
    def transcribe(self, _audio_path: str, **_kwargs):
        raise RuntimeError(
            "Requested float16 compute type, but the target device or backend do not support efficient float16 computation."
        )


class CpuRetryModel:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def transcribe(self, _audio_path: str, **kwargs):
        self.calls.append(kwargs)
        return [FakeSegment("CPU fallback transcript", 5.0)], FakeInfo("en")


class PromptEchoThenGoodModel:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def transcribe(self, _audio_path: str, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("initial_prompt"):
            repeated = " ".join(["Transcribe it in English without translating it."] * 8)
            return [FakeSegment(repeated, 10.0)], FakeInfo("en")
        return [FakeSegment("Real lecture transcript about scoring models.", 10.0)], FakeInfo("en")


class PromptEchoOnlyModel:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def transcribe(self, _audio_path: str, **kwargs):
        self.calls.append(kwargs)
        repeated = " ".join(["Transcribe it in English without translating it."] * 8)
        return [FakeSegment(repeated, 10.0)], FakeInfo("en")


class SilentLogger:
    async def stage(self, *_args, **_kwargs):
        return None

    async def progress(self, *_args, **_kwargs):
        return None

    async def partial_result(self, *_args, **_kwargs):
        return None

    async def file(self, *_args, **_kwargs):
        return None


class TranscriptionTests(unittest.TestCase):
    def test_default_transcription_settings_prioritize_quality_without_cuda(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = get_transcription_model_settings()
            predetection_enabled = language_predetection_enabled()
            batched_enabled = whisper_batched_transcription_enabled()

        self.assertEqual(settings.model_name, "large-v3")
        self.assertEqual(settings.device, "auto")
        self.assertEqual(settings.compute_type, "auto")
        self.assertFalse(predetection_enabled)
        self.assertFalse(batched_enabled)

    def test_default_language_and_beam_size_can_be_configured(self):
        with patch.dict(
            os.environ, {"WHISPER_DEFAULT_LANGUAGE": "ru", "WHISPER_BEAM_SIZE": "7", "WHISPER_BEST_OF": "6"}
        ):
            self.assertEqual(get_default_transcription_language(), "ru")
            self.assertEqual(get_whisper_beam_size(), 7)
            self.assertEqual(get_whisper_best_of(), 6)

    def test_russian_transcribe_kwargs_use_quality_settings_without_hotwords(self):
        with patch.dict(os.environ, {"WHISPER_BEAM_SIZE": "10", "WHISPER_BEST_OF": "10", "WHISPER_VAD_FILTER": "0"}):
            kwargs = build_transcribe_kwargs("ru")

        self.assertEqual(kwargs["beam_size"], 10)
        self.assertEqual(kwargs["best_of"], 10)
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertFalse(kwargs["vad_filter"])
        self.assertFalse(kwargs["condition_on_previous_text"])
        self.assertEqual(kwargs["language"], "ru")
        self.assertNotIn("initial_prompt", kwargs)
        self.assertNotIn("hotwords", kwargs)

    def test_initial_prompt_can_be_enabled_explicitly(self):
        with patch.dict(os.environ, {"WHISPER_INITIAL_PROMPT": "1"}):
            kwargs = build_transcribe_kwargs("ru")

        self.assertEqual(kwargs["language"], "ru")
        self.assertIn("initial_prompt", kwargs)

    def test_audio_preprocessing_can_be_disabled(self):
        with patch.dict(os.environ, {"WHISPER_PREPROCESS_AUDIO": "0"}):
            self.assertIsNone(preprocess_audio_for_transcription("missing.wav"))

    def test_invalid_default_language_is_ignored(self):
        with patch.dict(os.environ, {"WHISPER_DEFAULT_LANGUAGE": "de"}):
            self.assertIsNone(get_default_transcription_language())

    def test_default_russian_language_hint_is_passed_to_whisper(self):
        model = FakeWhisperModel()

        with patch.dict(
            os.environ,
            {"WHISPER_DEFAULT_LANGUAGE": "ru", "WHISPER_PREDETECT_LANGUAGE": "0", "WHISPER_PREPROCESS_AUDIO": "0"},
        ):
            with patch("conspectum.summary.get_transcription_model", return_value=model):
                transcript = asyncio.run(transcribe_audio(b"RIFFfake", SilentLogger(), filename="lecture.wav"))

        self.assertTrue(transcript)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["language"], "ru")

    def test_cuda_library_error_retries_transcription_on_cpu(self):
        cpu_model = CpuRetryModel()

        with patch.dict(
            os.environ,
            {"WHISPER_DEFAULT_LANGUAGE": "", "WHISPER_PREDETECT_LANGUAGE": "0", "WHISPER_PREPROCESS_AUDIO": "0"},
        ):
            with patch("conspectum.summary.get_transcription_model", return_value=CudaLibraryFailureModel()):
                with patch("conspectum.summary.get_cpu_transcription_model_for_retry", return_value=cpu_model):
                    transcript = asyncio.run(transcribe_audio(b"RIFFfake", SilentLogger(), filename="lecture.wav"))

        self.assertEqual(transcript, "CPU fallback transcript")
        self.assertEqual(len(cpu_model.calls), 1)

    def test_unsupported_compute_type_retries_with_safer_model(self):
        fallback_model = CpuRetryModel()

        with patch.dict(
            os.environ,
            {"WHISPER_DEFAULT_LANGUAGE": "", "WHISPER_PREDETECT_LANGUAGE": "0", "WHISPER_PREPROCESS_AUDIO": "0"},
        ):
            with patch("conspectum.summary.get_transcription_model", return_value=UnsupportedComputeTypeFailureModel()):
                with patch("conspectum.summary.get_transcription_model_for_compute_retry", return_value=fallback_model):
                    transcript = asyncio.run(transcribe_audio(b"RIFFfake", SilentLogger(), filename="lecture.wav"))

        self.assertEqual(transcript, "CPU fallback transcript")
        self.assertEqual(len(fallback_model.calls), 1)

    def test_auto_transcription_retries_russian_when_whisper_detects_ukrainian(self):
        model = FakeWhisperModel()

        with patch.dict(
            os.environ,
            {"WHISPER_DEFAULT_LANGUAGE": "", "WHISPER_PREDETECT_LANGUAGE": "0", "WHISPER_PREPROCESS_AUDIO": "0"},
        ):
            with patch("conspectum.summary.get_transcription_model", return_value=model):
                transcript = asyncio.run(transcribe_audio(b"RIFFfake", SilentLogger(), filename="lecture.wav"))

        self.assertEqual(transcript, "Почти все продукты заявки на кредит")
        self.assertEqual(len(model.calls), 2)
        self.assertNotIn("language", model.calls[0])
        self.assertEqual(model.calls[1]["language"], "ru")
        self.assertNotIn("initial_prompt", model.calls[1])

    def test_selected_russian_language_is_passed_to_whisper_without_auto_retry(self):
        model = FakeWhisperModel()

        with patch.dict(
            os.environ,
            {"WHISPER_DEFAULT_LANGUAGE": "", "WHISPER_PREDETECT_LANGUAGE": "0", "WHISPER_PREPROCESS_AUDIO": "0"},
        ):
            with patch("conspectum.summary.get_transcription_model", return_value=model):
                transcript = asyncio.run(
                    transcribe_audio(b"RIFFfake", SilentLogger(), filename="lecture.wav", language="ru")
                )

        self.assertEqual(transcript, "Почти все продукты заявки на кредит")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["language"], "ru")

    def test_prompt_echo_transcription_retries_without_initial_prompt(self):
        model = PromptEchoThenGoodModel()

        with patch.dict(
            os.environ,
            {
                "WHISPER_DEFAULT_LANGUAGE": "",
                "WHISPER_INITIAL_PROMPT": "1",
                "WHISPER_PREDETECT_LANGUAGE": "0",
                "WHISPER_PREPROCESS_AUDIO": "0",
            },
        ):
            with patch("conspectum.summary.get_transcription_model", return_value=model):
                transcript = asyncio.run(
                    transcribe_audio(b"RIFFfake", SilentLogger(), filename="lecture.wav", language="en")
                )

        self.assertEqual(transcript, "Real lecture transcript about scoring models.")
        self.assertEqual(len(model.calls), 2)
        self.assertIn("initial_prompt", model.calls[0])
        self.assertNotIn("initial_prompt", model.calls[1])
        self.assertFalse(model.calls[1]["condition_on_previous_text"])

    def test_repeated_prompt_transcription_fails_instead_of_generating_fake_output(self):
        model = PromptEchoOnlyModel()

        with patch.dict(
            os.environ,
            {
                "WHISPER_DEFAULT_LANGUAGE": "",
                "WHISPER_PREDETECT_LANGUAGE": "0",
                "WHISPER_PREPROCESS_AUDIO": "0",
            },
        ):
            with patch("conspectum.summary.get_transcription_model", return_value=model):
                with self.assertRaisesRegex(RuntimeError, "Transcription failed"):
                    asyncio.run(transcribe_audio(b"RIFFfake", SilentLogger(), filename="lecture.wav", language="en"))

        self.assertEqual(len(model.calls), 2)


if __name__ == "__main__":
    unittest.main()
