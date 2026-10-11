"""Conservative hardware resource sizing; no search or mathematical policy."""

from dataclasses import dataclass
from functools import lru_cache
import os
import platform
import subprocess


@dataclass(frozen=True)
class HostResources:
    cpu_count: int
    memory_bytes: int
    apple_silicon: bool


@lru_cache(maxsize=1)
def host_resources() -> HostResources:
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    memory = 0
    if apple:
        try:
            memory = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], stderr=subprocess.DEVNULL,
                text=True, timeout=2,
            ).strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            pass  # Restricted environments retain the original resource limits.
    return HostResources(os.cpu_count() or 1, memory, apple)


def expanded_mps_resources(device: str, host: HostResources) -> bool:
    return (device == "mps" and host.apple_silicon
            and host.cpu_count >= 12 and host.memory_bytes >= 32 * 1024**3)


def default_batch_size(length: int, device: str, host: HostResources) -> int:
    base = 2048 if length <= 58 else 1024
    if not expanded_mps_resources(device, host):
        return base
    # Measured with exact observation and the production hybrid pipeline.
    # Avoid extrapolating the large-batch setting beyond the project lengths.
    if length <= 58:
        return 8192
    if length <= 94:
        return 16384
    return base * 2


def default_c_threads(device: str, host: HostResources) -> int:
    # Keep capacity for GPU submission, Python verification and completion.
    return 6 if expanded_mps_resources(device, host) else 4
