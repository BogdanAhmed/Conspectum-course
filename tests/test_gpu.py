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

from conspectum.gpu import CTranslate2CudaStatus  # noqa: E402
from conspectum.gpu import NvidiaGpuInfo  # noqa: E402
from conspectum.gpu import NvidiaSmiResult  # noqa: E402
from conspectum.gpu import parse_nvidia_smi_query_output  # noqa: E402
import conspectum.summary as summary  # noqa: E402


def reset_transcription_cache():
    summary._TRANSCRIPTION_MODEL = None
    summary._TRANSCRIPTION_MODEL_NAME = None
    summary._TRANSCRIPTION_MODEL_SETTINGS = None
    summary._TRANSCRIPTION_RUNTIME_INFO = None
    summary._TRANSCRIPTION_BATCHED_PIPELINE = None
    summary._TRANSCRIPTION_BATCHED_MODEL = None


class WhisperGpuSelectionTests(unittest.TestCase):
    def setUp(self):
        reset_transcription_cache()

    def tearDown(self):
        reset_transcription_cache()

    def test_parse_nvidia_smi_query_output_reads_gpu_and_vram(self):
        gpus = parse_nvidia_smi_query_output("NVIDIA RTX 4060, 8192\nNVIDIA RTX 4090, 24564\n")

        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0].name, "NVIDIA RTX 4060")
        self.assertEqual(gpus[0].memory_total_mb, 8192)
        self.assertEqual(gpus[1].name, "NVIDIA RTX 4090")
        self.assertEqual(gpus[1].memory_total_mb, 24564)

    def test_whisper_device_cpu_forces_cpu_without_gpu_probe(self):
        created_settings = []

        def create_model(_model_name, settings):
            created_settings.append(settings)
            return object()

        with patch.dict(os.environ, {"WHISPER_DEVICE": "cpu", "WHISPER_COMPUTE_TYPE": "auto"}, clear=True):
            with patch("conspectum.summary.run_nvidia_smi", side_effect=AssertionError("GPU probe should not run")):
                with patch("conspectum.summary.create_transcription_model", side_effect=create_model):
                    summary.get_transcription_model()

        self.assertEqual(len(created_settings), 1)
        self.assertEqual(created_settings[0].device, "cpu")
        self.assertEqual(created_settings[0].compute_type, "int8")

    def test_whisper_device_auto_without_gpu_falls_back_to_cpu(self):
        created_settings = []

        def create_model(_model_name, settings):
            created_settings.append(settings)
            return object()

        nvidia_smi = NvidiaSmiResult(command_available=False, error="nvidia-smi was not found on PATH.")
        with patch.dict(os.environ, {"WHISPER_DEVICE": "auto", "WHISPER_COMPUTE_TYPE": "auto"}, clear=True):
            with patch("conspectum.summary.run_nvidia_smi", return_value=nvidia_smi):
                with patch("conspectum.summary.create_transcription_model", side_effect=create_model):
                    summary.get_transcription_model()

        runtime_info = summary.get_active_transcription_runtime_info()
        self.assertEqual(len(created_settings), 1)
        self.assertEqual(created_settings[0].device, "cpu")
        self.assertEqual(created_settings[0].compute_type, "int8")
        self.assertIsNotNone(runtime_info)
        self.assertIn("nvidia-smi", runtime_info.fallback_reason or "")

    def test_whisper_device_auto_uses_cuda_when_probe_and_init_succeed(self):
        created_settings = []
        gpu = NvidiaGpuInfo(name="NVIDIA RTX Test", memory_total_mb=12288)
        nvidia_smi = NvidiaSmiResult(command_available=True, returncode=0, gpus=(gpu,))
        ctranslate2 = CTranslate2CudaStatus(installed=True, cuda_device_count=1)

        def create_model(_model_name, settings):
            created_settings.append(settings)
            return object()

        with patch.dict(os.environ, {"WHISPER_DEVICE": "auto", "WHISPER_COMPUTE_TYPE": "auto"}, clear=True):
            with patch("conspectum.summary.run_nvidia_smi", return_value=nvidia_smi):
                with patch("conspectum.summary.check_ctranslate2_cuda", return_value=ctranslate2):
                    with patch("conspectum.summary.create_transcription_model", side_effect=create_model):
                        summary.get_transcription_model()

        runtime_info = summary.get_active_transcription_runtime_info()
        self.assertEqual(len(created_settings), 1)
        self.assertEqual(created_settings[0].device, "cuda")
        self.assertEqual(created_settings[0].compute_type, "float16")
        self.assertIsNotNone(runtime_info)
        self.assertEqual(runtime_info.gpu_name, "NVIDIA RTX Test")
        self.assertIsNone(runtime_info.fallback_reason)

    def test_whisper_device_auto_cuda_initialization_failure_falls_back_to_cpu(self):
        created_settings = []
        gpu = NvidiaGpuInfo(name="NVIDIA RTX Test", memory_total_mb=12288)
        nvidia_smi = NvidiaSmiResult(command_available=True, returncode=0, gpus=(gpu,))
        ctranslate2 = CTranslate2CudaStatus(installed=True, cuda_device_count=1)

        def create_model(_model_name, settings):
            created_settings.append(settings)
            if settings.device == "cuda":
                raise RuntimeError("CUDA initialization failed")
            return object()

        with patch.dict(os.environ, {"WHISPER_DEVICE": "auto", "WHISPER_COMPUTE_TYPE": "auto"}, clear=True):
            with patch("conspectum.summary.run_nvidia_smi", return_value=nvidia_smi):
                with patch("conspectum.summary.check_ctranslate2_cuda", return_value=ctranslate2):
                    with patch("conspectum.summary.create_transcription_model", side_effect=create_model):
                        summary.get_transcription_model()

        runtime_info = summary.get_active_transcription_runtime_info()
        self.assertEqual([settings.device for settings in created_settings], ["cuda", "cuda", "cpu"])
        self.assertEqual([settings.compute_type for settings in created_settings], ["float16", "int8", "int8"])
        self.assertIsNotNone(runtime_info)
        self.assertEqual(runtime_info.active_settings.device, "cpu")
        self.assertIn("CUDA model initialization failed", runtime_info.fallback_reason or "")

    def test_whisper_device_cuda_tries_cuda_then_falls_back_to_cpu(self):
        created_settings = []
        nvidia_smi = NvidiaSmiResult(command_available=False, error="nvidia-smi was not found on PATH.")
        ctranslate2 = CTranslate2CudaStatus(installed=True, cuda_device_count=0)

        def create_model(_model_name, settings):
            created_settings.append(settings)
            if settings.device == "cuda":
                raise RuntimeError("CUDA libraries are missing")
            return object()

        with patch.dict(os.environ, {"WHISPER_DEVICE": "cuda", "WHISPER_COMPUTE_TYPE": "float16"}, clear=True):
            with patch("conspectum.summary.run_nvidia_smi", return_value=nvidia_smi):
                with patch("conspectum.summary.check_ctranslate2_cuda", return_value=ctranslate2):
                    with patch("conspectum.summary.create_transcription_model", side_effect=create_model):
                        summary.get_transcription_model()

        runtime_info = summary.get_active_transcription_runtime_info()
        self.assertEqual(created_settings[0].device, "cuda")
        self.assertEqual(created_settings[-1].device, "cpu")
        self.assertIsNotNone(runtime_info)
        self.assertEqual(runtime_info.active_settings.compute_type, "int8")


if __name__ == "__main__":
    unittest.main()
