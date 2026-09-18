"""Central PyTorch device selection for local macOS and Colab runs.

The search mathematics is device-independent.  ``auto`` selects CUDA first,
then Apple MPS, and finally CPU.  Explicit ``cuda`` and ``mps`` requests fail
instead of silently falling back, so the displayed device is trustworthy.
"""

import os

import torch


TORCH_DEVICE_NAMES = ("auto", "mps", "cuda", "cpu")


def validate_torch_device_name(name: str) -> None:
    """Validate a requested logical device without probing the hardware."""
    if name not in TORCH_DEVICE_NAMES:
        raise ValueError(
            "torch device must be one of: {}".format(", ".join(TORCH_DEVICE_NAMES))
        )


def _mps_device() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS 無法使用：請在支援 Metal 的 Mac 與正確的 Python 環境執行。"
        )
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError(
            "請移除 PYTORCH_ENABLE_MPS_FALLBACK=1，確保搜尋計算實際在 MPS 上執行。"
        )
    return torch.device("mps")


def resolve_torch_device(name: str) -> torch.device:
    """Resolve ``auto|mps|cuda|cpu`` to an available ``torch.device``.

    ``auto`` prefers CUDA because it is the accelerator exposed by Google
    Colab, then MPS for Apple Silicon, and uses CPU only when neither
    accelerator exists.  Explicit accelerator requests never fall back.
    """
    validate_torch_device_name(name)
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return _mps_device()
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA 無法使用：請確認 Colab 已選擇 GPU runtime，且 PyTorch 可偵測 CUDA。"
            )
        return torch.device("cuda")
    if name == "mps":
        return _mps_device()
    return torch.device("cpu")


def validate_observation_backend(
    backend: str, requested_device: str, length: int
) -> None:
    """Validate the exact-observation implementation independently of loss.

    The custom Metal implementation is MPS-only.  Generic ``torch``
    observation works on CUDA, MPS and CPU.  With ``auto`` the definitive
    Metal check is performed after hardware resolution by the search object.
    """
    if backend not in ("torch", "metal"):
        raise ValueError("observation_backend must be torch or metal")
    validate_torch_device_name(requested_device)
    if backend == "metal" and requested_device in ("cuda", "cpu"):
        raise ValueError("metal observation_backend requires MPS")
    if backend == "metal" and length > 94:
        raise ValueError("metal observation_backend requires L<=94")
