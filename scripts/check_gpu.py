from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from conspectum.gpu import collect_gpu_diagnostics
from conspectum.gpu import ctranslate2_cuda_ready
from conspectum.gpu import describe_nvidia_smi
from conspectum.gpu import format_gpu_info
from conspectum.gpu import get_probe_model_class
from conspectum.gpu import probe_faster_whisper_cuda
from conspectum.gpu import summarize_probe


def main() -> int:
    diagnostics = collect_gpu_diagnostics()

    print(f"Python version: {diagnostics.python_version}")
    print(f"OS: {diagnostics.os}")
    print(f"nvidia-smi: {describe_nvidia_smi(diagnostics.nvidia_smi)}")
    if diagnostics.nvidia_smi.stdout:
        print("nvidia-smi raw output:")
        print(diagnostics.nvidia_smi.stdout)
    if diagnostics.nvidia_smi.stderr:
        print("nvidia-smi stderr:")
        print(diagnostics.nvidia_smi.stderr)

    print(f"NVIDIA GPU: {format_gpu_info(diagnostics.primary_gpu)}")
    print(
        "CTranslate2 CUDA: "
        f"installed={diagnostics.ctranslate2.installed}, "
        f"device_count={diagnostics.ctranslate2.cuda_device_count}, "
        f"error={diagnostics.ctranslate2.error or 'none'}"
    )
    print(f"Recommended compute_type: {diagnostics.recommended_compute_type}")
    print(f"Recommended WHISPER_BATCH_SIZE: {diagnostics.recommended_batch_size}")

    probe = None
    model_class = get_probe_model_class()
    if model_class is None:
        print("faster-whisper import: failed or WhisperModel was not available")
    elif diagnostics.primary_gpu is None:
        print("faster-whisper CUDA probe: skipped because no NVIDIA GPU was found")
    else:
        print("faster-whisper CUDA probe: initializing tiny model and running a short transcribe on device=cuda...")
        probe = probe_faster_whisper_cuda(
            model_class,
            model_name="tiny",
            compute_type=diagnostics.recommended_compute_type,
        )
        print(f"faster-whisper CUDA probe: {summarize_probe(probe)}")

    ct2_ready = ctranslate2_cuda_ready(diagnostics.ctranslate2)
    gpu_ready = diagnostics.primary_gpu is not None and probe is not None and probe.ok
    if ct2_ready is False:
        gpu_ready = False

    if gpu_ready:
        print("Final verdict: GPU mode ready")
        return 0

    print("Final verdict: fallback to CPU required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
