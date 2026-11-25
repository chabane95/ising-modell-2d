"""Build and call the C99 single-cluster sampler.  Run python native_wolff.py to compile."""

import ctypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import operator
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "native" / "wolff.c"
BUILD = ROOT / "build" / "native"
BUILD_LOCK = threading.Lock()
INT_MAX = 2**31 - 1


def compiler():
    """Return a C99 compiler driver supporting GCC/Clang command-line flags."""
    choices = [os.environ["CC"]] if os.environ.get("CC") else ["cc", "gcc", "clang"]
    for name in choices:
        executable = shutil.which(name)
        if executable:
            return executable
    raise FileNotFoundError("Building the Wolff kernel requires GCC or Clang on PATH (or set CC).")


def build_library():
    """Build once per source revision. Completed libraries can be used by multiple workers."""
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]
    suffix = ".dll" if os.name == "nt" else ".dylib" if sys.platform == "darwin" else ".so"
    target = BUILD / f"wolff-{platform.machine()}-{source_hash}{suffix}"
    with BUILD_LOCK:
        if target.is_file():
            return target
        driver = compiler()
        BUILD.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=BUILD) as temporary:
            output = Path(temporary) / ("wolff" + suffix)
            flags = ["-O3", "-std=c99", "-Wall", "-Wextra"]
            if os.name != "nt":
                flags.append("-fPIC")
            flags.append("-dynamiclib" if sys.platform == "darwin" else "-shared")
            subprocess.run([driver, *flags, str(SOURCE), "-o", str(output), "-lm"], check=True)
            try:
                os.replace(output, target)
            except PermissionError:
                # A parallel Windows process may already have this revision loaded.
                if not target.is_file():
                    raise
    return target


@lru_cache(maxsize=1)
def _kernel():
    library = ctypes.CDLL(str(build_library()))
    function = library.wolff_sample
    vec = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    function.argtypes = [
        ctypes.c_int32, ctypes.c_double, ctypes.c_uint64, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, vec, vec, ctypes.POINTER(ctypes.c_double),
    ]
    function.restype = ctypes.c_int
    return function


@dataclass
class Chain:
    energy: np.ndarray
    magnetization: np.ndarray
    mean_cluster_size: float


def sample_chain(L, temperature, *, warmup, samples, seed, initial="random", stride=1):
    """Return total E, signed M and mean seed-cluster size for the zero-field square torus.

    J = k_B = 1. Discard warmup cluster flips, then record after each stride flips.
    The mean cluster size uses every post-warmup update.  Each call owns its RNG
    and releases the Python GIL while simulating.
    """
    L, warmup, samples, stride, seed = map(operator.index, (L, warmup, samples, stride, seed))
    if not 2 <= L or 4 * L * L > INT_MAX:
        raise ValueError("L must be at least 2 and its lattice must fit the 32-bit kernel")
    temperature = float(temperature)
    if not (math.isfinite(temperature) and temperature > 0):
        raise ValueError("Temperature must be positive and finite")
    if not (0 <= warmup <= INT_MAX and 1 <= samples <= INT_MAX and 1 <= stride <= INT_MAX):
        raise ValueError("Invalid warmup, samples or stride")
    if not 0 <= seed < 2**64:
        raise ValueError("A 64-bit unsigned seed is required")
    starts = {"random": 0, "up": 1, "down": -1}
    if initial not in starts:
        raise ValueError("Choose initial='random', 'up' or 'down'")
    energy = np.empty(samples, dtype=np.int32)
    magnetization = np.empty(samples, dtype=np.int32)
    mean_size = ctypes.c_double()
    status = _kernel()(L, temperature, seed, starts[initial], warmup, samples, stride,
                       energy, magnetization, ctypes.byref(mean_size))
    if status == 2:
        raise MemoryError("Cannot allocate the spin lattice and its neighbor table")
    if status:
        raise ValueError(f"Wolff kernel rejected arguments (status {status})")
    return Chain(energy, magnetization, mean_size.value)


if __name__ == "__main__":
    print(build_library())
