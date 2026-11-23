"""Compiled C99 DOP853 fast path for Lorenz-63 integration and cycle Fourier projection.

When standard DOP853 solver options are used and a C99 compiler is available
(or a pre-built shared library sits alongside ``_dop853.c``), integration runs
in native C via ``ctypes``. Non-DOP853 solver configurations, invalid arguments,
or environments without a compiler cleanly fall back to SciPy ``solve_ivp``.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid

import numpy as np

_EPS = float(np.finfo(float).eps)
_SUPPORTED_SOLVER_KEYS = frozenset({"method", "rtol", "atol", "max_step", "first_step"})
_LIB = None
_LIB_FAILED = False


def _shared_library_name(source_path: Path) -> str:
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()[:16]
    machine = platform.machine() or "unknown"
    return f"_dop853_{digest}_{machine}.so"


def _configure_library(lib: ctypes.CDLL) -> ctypes.CDLL:
    c_double_p = ctypes.POINTER(ctypes.c_double)
    c_int64_p = ctypes.POINTER(ctypes.c_int64)

    lib.lorenz_dop853_spinup_batch.argtypes = [
        c_double_p,          # y0_batch
        ctypes.c_size_t,     # n_states
        ctypes.c_double,     # spinup_time
        ctypes.c_double,     # sigma
        ctypes.c_double,     # rho
        ctypes.c_double,     # beta
        ctypes.c_double,     # rtol
        c_double_p,          # atol[3]
        ctypes.c_double,     # max_step
        ctypes.c_double,     # first_step
        c_double_p,          # y_out_batch
    ]
    lib.lorenz_dop853_spinup_batch.restype = ctypes.c_int

    lib.lorenz_dop853_phase_samples.argtypes = [
        c_double_p,          # y0[3]
        c_double_p,          # forcing[3]
        c_double_p,          # rhs_phase[3]
        ctypes.c_double,     # omega
        c_double_p,          # sample_times
        ctypes.c_size_t,     # n_samples
        ctypes.c_double,     # sigma
        ctypes.c_double,     # rho
        ctypes.c_double,     # beta
        ctypes.c_double,     # rtol
        c_double_p,          # atol[3]
        ctypes.c_double,     # max_step
        ctypes.c_double,     # first_step
        c_double_p,          # values_out
    ]
    lib.lorenz_dop853_phase_samples.restype = ctypes.c_int

    lib.lorenz_dop853_fourier_block.argtypes = [
        c_double_p,          # y0[3]
        c_double_p,          # forcing_vectors
        c_double_p,          # forcing_phases
        ctypes.c_int,        # has_forcing_phases
        ctypes.c_size_t,     # n_cond
        ctypes.c_double,     # omega
        ctypes.c_double,     # phase
        ctypes.c_double,     # discard_time
        ctypes.c_size_t,     # n_cycle
        ctypes.c_size_t,     # n_phase
        c_int64_p,           # harmonics
        ctypes.c_size_t,     # n_harm
        ctypes.c_double,     # sigma
        ctypes.c_double,     # rho
        ctypes.c_double,     # beta
        ctypes.c_double,     # rtol
        c_double_p,          # atol[3]
        ctypes.c_double,     # max_step
        ctypes.c_double,     # first_step
        ctypes.c_int,        # average_cycles
        c_double_p,          # out_real
        c_double_p,          # out_imag
    ]
    lib.lorenz_dop853_fourier_block.restype = ctypes.c_int

    lib.lorenz_dop853_dc_moments_block.argtypes = [
        c_double_p,          # y0[3]
        c_double_p,          # forcing_vectors
        c_double_p,          # forcing_phases
        ctypes.c_int,        # has_forcing_phases
        ctypes.c_size_t,     # n_cond
        ctypes.c_double,     # omega
        ctypes.c_double,     # phase
        ctypes.c_double,     # discard_time
        ctypes.c_size_t,     # n_cycle
        ctypes.c_size_t,     # n_phase
        c_int64_p,           # harmonics
        ctypes.c_size_t,     # n_harm
        ctypes.c_double,     # sigma
        ctypes.c_double,     # rho
        ctypes.c_double,     # beta
        ctypes.c_double,     # rtol
        c_double_p,          # atol[3]
        ctypes.c_double,     # max_step
        ctypes.c_double,     # first_step
        c_double_p,          # out_real
        c_double_p,          # out_imag
        c_double_p,          # out_moments (n_cond, 10)
    ]
    lib.lorenz_dop853_dc_moments_block.restype = ctypes.c_int

    lib.lorenz_dop853_dual_freq_block.argtypes = [
        c_double_p,          # y0[3]
        c_double_p,          # forcing1_vectors
        c_double_p,          # forcing1_phases
        ctypes.c_int,        # has_forcing1_phases
        ctypes.c_double,     # omega1
        c_double_p,          # forcing2_vectors
        c_double_p,          # forcing2_phases
        ctypes.c_int,        # has_forcing2_phases
        ctypes.c_double,     # omega2
        ctypes.c_size_t,     # n_cond
        ctypes.c_double,     # discard_time
        ctypes.c_double,     # dt
        ctypes.c_size_t,     # n_samples
        c_double_p,          # target_omegas
        ctypes.c_size_t,     # n_freq
        ctypes.c_double,     # sigma
        ctypes.c_double,     # rho
        ctypes.c_double,     # beta
        ctypes.c_double,     # rtol
        c_double_p,          # atol[3]
        ctypes.c_double,     # max_step
        ctypes.c_double,     # first_step
        c_double_p,          # out_real
        c_double_p,          # out_imag
        c_double_p,          # out_moments (n_cond, 10) or NULL
    ]
    lib.lorenz_dop853_dual_freq_block.restype = ctypes.c_int
    return lib


def _compiler_commands() -> list[list[str]]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(raw: str | Path | None):
        if not raw:
            return
        resolved = shutil.which(str(raw)) if os.sep not in str(raw) else str(raw)
        if resolved and Path(resolved).is_file() and resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)

    add_candidate(os.environ.get("CC"))
    py_bin = Path(sys.executable).parent
    arch = platform.machine() or "x86_64"
    for name in (f"{arch}-conda-linux-gnu-gcc", "gcc", "cc", "clang"):
        add_candidate(py_bin / name)
    for name in ("gcc", "cc", "clang"):
        add_candidate(name)

    commands: list[list[str]] = []
    for cc in candidates:
        commands.append([cc])
        if Path(cc).name in {"gcc", "cc"}:
            commands.append([cc, "-fuse-ld=gold"])
    return commands


def compile_native_extension(target_dir: Path | None = None) -> Path | None:
    """Compile ``_dop853.c`` atomically into ``target_dir`` or the cache directory."""
    source_path = Path(__file__).with_name("_dop853.c")
    if not source_path.is_file():
        return None
    compiler_commands = _compiler_commands()
    if not compiler_commands:
        return None

    if target_dir is None:
        env_cache = os.environ.get("LORENZ_C_CACHE_DIR")
        if env_cache:
            out_dir = Path(env_cache)
        else:
            uid = os.getuid() if hasattr(os, "getuid") else "user"
            out_dir = Path(tempfile.gettempdir()) / f"lorenz-c-cache-{uid}"
    else:
        out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / _shared_library_name(source_path)
    if target.is_file():
        return target

    for prefix in compiler_commands:
        tmp = out_dir / f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}.so"
        cmd = [
            *prefix,
            "-O3",
            "-ffp-contract=off",
            "-fPIC",
            "-shared",
            "-std=c99",
            str(source_path),
            "-lm",
            "-o",
            str(tmp),
        ]
        try:
            completed = subprocess.run(
                cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if completed.returncode == 0 and tmp.is_file():
                os.replace(tmp, target)
                return target
        except OSError:
            pass
        tmp.unlink(missing_ok=True)
    return None


def load_c_solver() -> ctypes.CDLL | None:
    """Return the compiled C DOP853 CDLL, or ``None`` if disabled/unavailable."""
    global _LIB, _LIB_FAILED
    if os.environ.get("LORENZ_DISABLE_C_SOLVER") == "1":
        return None
    if _LIB is not None:
        return _LIB
    if _LIB_FAILED:
        return None

    source_path = Path(__file__).with_name("_dop853.c")
    if not source_path.is_file():
        _LIB_FAILED = True
        return None

    so_name = _shared_library_name(source_path)
    bundled = source_path.with_name(so_name)
    candidates = [bundled] if bundled.is_file() else []
    if not candidates:
        compiled = compile_native_extension()
        if compiled is not None:
            candidates.append(compiled)

    for path in candidates:
        try:
            _LIB = _configure_library(ctypes.CDLL(str(path)))
            return _LIB
        except OSError:
            continue

    _LIB_FAILED = True
    return None


def _extract_params(cfg: dict, t_bound: float):
    """Validate and extract C kernel parameters, or return ``None`` to fall back."""
    solver = cfg.get("solver", {})
    if solver.get("method") != "DOP853":
        return None
    if not set(solver).issubset(_SUPPORTED_SOLVER_KEYS):
        return None

    try:
        rtol = float(solver.get("rtol", 1e-3))
        atol_raw = np.asarray(solver.get("atol", 1e-6), dtype=np.float64)
        max_step = float(solver.get("max_step", np.inf))
        first_step_raw = solver.get("first_step")
    except (TypeError, ValueError):
        return None

    if not np.isfinite(rtol) or rtol < 100.0 * _EPS:
        return None
    if atol_raw.ndim == 0:
        atol_vec = np.array([float(atol_raw)] * 3, dtype=np.float64)
    elif atol_raw.shape == (3,):
        atol_vec = np.ascontiguousarray(atol_raw, dtype=np.float64)
    else:
        return None
    if not np.isfinite(atol_vec).all() or np.any(atol_vec <= 0.0):
        return None
    if np.isnan(max_step) or max_step <= 0.0:
        return None

    if first_step_raw is None:
        first_step = -1.0
    else:
        try:
            first_step = float(first_step_raw)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(first_step) or first_step <= 0.0:
            return None
        if t_bound > 0.0 and first_step > t_bound:
            return None

    lorenz = cfg["lorenz"]
    sigma = float(lorenz["sigma"])
    rho = float(lorenz["rho"])
    beta = float(lorenz["beta"])
    if not (np.isfinite(sigma) and np.isfinite(rho) and np.isfinite(beta)):
        return None

    return sigma, rho, beta, rtol, atol_vec, max_step, first_step


def _check_rc(rc: int) -> bool:
    if rc == 0:
        return True
    if rc == -1:
        raise RuntimeError("solve_ivp failed: Required step size is less than spacing between numbers.")
    if rc == -2:
        raise RuntimeError("non-finite values in solve_ivp output")
    return False


def try_spinup_batch(raw_proposals: np.ndarray, spinup_time: float, cfg: dict) -> np.ndarray | None:
    """Integrate unforced Lorenz spinups in C for a batch of shape ``(n, 3)``."""
    lib = load_c_solver()
    if lib is None:
        return None
    params = _extract_params(cfg, float(spinup_time))
    if params is None:
        return None
    sigma, rho, beta, rtol, atol_vec, max_step, first_step = params

    y0_arr = np.ascontiguousarray(raw_proposals, dtype=np.float64)
    if y0_arr.ndim != 2 or y0_arr.shape[1] != 3:
        return None
    n_states = y0_arr.shape[0]
    y_out = np.empty_like(y0_arr)
    if n_states == 0:
        return y_out

    rc = lib.lorenz_dop853_spinup_batch(
        y0_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(n_states),
        ctypes.c_double(float(spinup_time)),
        ctypes.c_double(sigma),
        ctypes.c_double(rho),
        ctypes.c_double(beta),
        ctypes.c_double(rtol),
        atol_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(max_step),
        ctypes.c_double(first_step),
        y_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if not _check_rc(rc):
        return None
    return y_out


def try_phase_samples(
    initial_state: np.ndarray,
    forcing_vector: np.ndarray,
    omega: float,
    rhs_phase,
    flat_times: np.ndarray,
    cfg: dict,
) -> np.ndarray | None:
    """Integrate one forced trajectory and evaluate at ``flat_times`` in C."""
    lib = load_c_solver()
    if lib is None:
        return None
    t_bound = float(flat_times[-1])
    params = _extract_params(cfg, t_bound)
    if params is None:
        return None
    sigma, rho, beta, rtol, atol_vec, max_step, first_step = params

    y0_arr = np.ascontiguousarray(initial_state, dtype=np.float64)
    fv_arr = np.ascontiguousarray(forcing_vector, dtype=np.float64)
    if not np.isfinite(fv_arr).all():
        return None
    phase_arr = np.asarray(rhs_phase, dtype=np.float64)
    if phase_arr.ndim == 0:
        phase_vec = np.array([float(phase_arr)] * 3, dtype=np.float64)
    elif phase_arr.shape == (3,):
        phase_vec = np.ascontiguousarray(phase_arr, dtype=np.float64)
    else:
        return None
    if not np.isfinite(phase_vec).all():
        return None

    times_arr = np.ascontiguousarray(flat_times, dtype=np.float64)
    n_samples = times_arr.size
    values_out = np.empty((3, n_samples), dtype=np.float64)

    rc = lib.lorenz_dop853_phase_samples(
        y0_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        fv_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        phase_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(float(omega)),
        times_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(n_samples),
        ctypes.c_double(sigma),
        ctypes.c_double(rho),
        ctypes.c_double(beta),
        ctypes.c_double(rtol),
        atol_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(max_step),
        ctypes.c_double(first_step),
        values_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if not _check_rc(rc):
        return None
    return values_out


def try_fourier_block(
    initial_state,
    forcing_vectors,
    forcing_phases,
    omega: float,
    phase: float,
    discard_time: float,
    n_cycle: int,
    n_phase: int,
    harmonics,
    cfg: dict,
    *,
    average_cycles: bool,
) -> np.ndarray | None:
    """Integrate every condition for one block and return Fourier coefficients."""
    lib = load_c_solver()
    if lib is None:
        return None

    y0_arr = np.ascontiguousarray(initial_state, dtype=np.float64)
    fv_arr = np.ascontiguousarray(forcing_vectors, dtype=np.float64)
    if y0_arr.shape != (3,) or fv_arr.ndim != 2 or fv_arr.shape[1] != 3 or not np.isfinite(fv_arr).all():
        return None
    n_cond = fv_arr.shape[0]
    if n_cond < 1:
        return None

    has_forcing_phases = int(forcing_phases is not None)
    if has_forcing_phases:
        fp_arr = np.ascontiguousarray(forcing_phases, dtype=np.float64)
        if fp_arr.shape != (n_cond, 3) or not np.isfinite(fp_arr).all():
            return None
        fp_ptr = fp_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        fp_arr = None
        fp_ptr = ctypes.POINTER(ctypes.c_double)()

    harm_arr = np.ascontiguousarray(harmonics, dtype=np.int64)
    if harm_arr.ndim != 1 or harm_arr.size < 1:
        return None
    n_harm = harm_arr.size

    period = (2.0 * np.pi) / float(omega)
    last_time = float(discard_time) + (float(n_cycle - 1) + float(n_phase - 1) / float(n_phase)) * period
    if not np.isfinite(last_time):
        return None

    params = _extract_params(cfg, last_time)
    if params is None:
        return None
    sigma, rho, beta, rtol, atol_vec, max_step, first_step = params

    out_shape = (n_cond, 3, n_harm) if average_cycles else (n_cond, 3, n_cycle, n_harm)
    out_real = np.empty(out_shape, dtype=np.float64)
    out_imag = np.empty(out_shape, dtype=np.float64)

    rc = lib.lorenz_dop853_fourier_block(
        y0_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        fv_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        fp_ptr,
        ctypes.c_int(has_forcing_phases),
        ctypes.c_size_t(n_cond),
        ctypes.c_double(float(omega)),
        ctypes.c_double(float(phase)),
        ctypes.c_double(float(discard_time)),
        ctypes.c_size_t(n_cycle),
        ctypes.c_size_t(n_phase),
        harm_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_size_t(n_harm),
        ctypes.c_double(sigma),
        ctypes.c_double(rho),
        ctypes.c_double(beta),
        ctypes.c_double(rtol),
        atol_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(max_step),
        ctypes.c_double(first_step),
        ctypes.c_int(1 if average_cycles else 0),
        out_real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if not _check_rc(rc):
        return None
    return out_real + 1j * out_imag


def try_dc_moments_block(
    initial_state,
    forcing_vectors,
    forcing_phases,
    omega: float,
    phase: float,
    discard_time: float,
    n_cycle: int,
    n_phase: int,
    harmonics,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Integrate one block, returning (Fourier means, 10 DC phase-space moments)."""
    lib = load_c_solver()
    if lib is None:
        return None

    y0_arr = np.ascontiguousarray(initial_state, dtype=np.float64)
    fv_arr = np.ascontiguousarray(forcing_vectors, dtype=np.float64)
    if y0_arr.shape != (3,) or fv_arr.ndim != 2 or fv_arr.shape[1] != 3 or not np.isfinite(fv_arr).all():
        return None
    n_cond = fv_arr.shape[0]
    if n_cond < 1:
        return None

    has_forcing_phases = int(forcing_phases is not None)
    if has_forcing_phases:
        fp_arr = np.ascontiguousarray(forcing_phases, dtype=np.float64)
        if fp_arr.shape != (n_cond, 3) or not np.isfinite(fp_arr).all():
            return None
        fp_ptr = fp_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        fp_arr = None
        fp_ptr = ctypes.POINTER(ctypes.c_double)()

    harm_arr = np.ascontiguousarray(harmonics, dtype=np.int64)
    if harm_arr.ndim != 1 or harm_arr.size < 1:
        return None
    n_harm = harm_arr.size

    period = (2.0 * np.pi) / float(omega)
    last_time = float(discard_time) + (float(n_cycle - 1) + float(n_phase - 1) / float(n_phase)) * period
    if not np.isfinite(last_time):
        return None

    params = _extract_params(cfg, last_time)
    if params is None:
        return None
    sigma, rho, beta, rtol, atol_vec, max_step, first_step = params

    out_real = np.empty((n_cond, 3, n_harm), dtype=np.float64)
    out_imag = np.empty((n_cond, 3, n_harm), dtype=np.float64)
    out_moments = np.empty((n_cond, 10), dtype=np.float64)

    rc = lib.lorenz_dop853_dc_moments_block(
        y0_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        fv_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        fp_ptr,
        ctypes.c_int(has_forcing_phases),
        ctypes.c_size_t(n_cond),
        ctypes.c_double(float(omega)),
        ctypes.c_double(float(phase)),
        ctypes.c_double(float(discard_time)),
        ctypes.c_size_t(n_cycle),
        ctypes.c_size_t(n_phase),
        harm_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_size_t(n_harm),
        ctypes.c_double(sigma),
        ctypes.c_double(rho),
        ctypes.c_double(beta),
        ctypes.c_double(rtol),
        atol_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(max_step),
        ctypes.c_double(first_step),
        out_real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_moments.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if not _check_rc(rc):
        return None
    return out_real + 1j * out_imag, out_moments


def try_dual_freq_block(
    initial_state,
    forcing1_vectors,
    forcing1_phases,
    omega1: float,
    forcing2_vectors,
    forcing2_phases,
    omega2: float,
    discard_time: float,
    dt: float,
    n_samples: int,
    target_omegas,
    cfg: dict,
    *,
    return_moments: bool = False,
):
    """Integrate dual-frequency forced conditions for one block and project onto ``target_omegas``.

    Computes the uniform-grid DFT projection
        Z[m] = (1 / n_samples) * sum_{s=0}^{n_samples-1} X(discard_time + s*dt) * exp(-1j * target_omegas[m] * t_s)
    using fused C99 DOP853 dense output and periodic re-anchored phase-rotation recurrence.
    """
    lib = load_c_solver()
    if lib is None:
        return None

    y0_arr = np.ascontiguousarray(initial_state, dtype=np.float64)
    fv1_arr = np.ascontiguousarray(forcing1_vectors, dtype=np.float64)
    fv2_arr = np.ascontiguousarray(forcing2_vectors, dtype=np.float64)
    if (
        y0_arr.shape != (3,)
        or fv1_arr.ndim != 2
        or fv1_arr.shape[1] != 3
        or fv2_arr.shape != fv1_arr.shape
        or not np.isfinite(fv1_arr).all()
        or not np.isfinite(fv2_arr).all()
    ):
        return None
    n_cond = fv1_arr.shape[0]
    if n_cond < 1 or n_samples < 1 or dt <= 0.0:
        return None

    has_fp1 = int(forcing1_phases is not None)
    if has_fp1:
        fp1_arr = np.ascontiguousarray(forcing1_phases, dtype=np.float64)
        if fp1_arr.shape != (n_cond, 3) or not np.isfinite(fp1_arr).all():
            return None
        fp1_ptr = fp1_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        fp1_arr = None
        fp1_ptr = ctypes.POINTER(ctypes.c_double)()

    has_fp2 = int(forcing2_phases is not None)
    if has_fp2:
        fp2_arr = np.ascontiguousarray(forcing2_phases, dtype=np.float64)
        if fp2_arr.shape != (n_cond, 3) or not np.isfinite(fp2_arr).all():
            return None
        fp2_ptr = fp2_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        fp2_arr = None
        fp2_ptr = ctypes.POINTER(ctypes.c_double)()

    om_arr = np.ascontiguousarray(target_omegas, dtype=np.float64)
    if om_arr.ndim != 1 or om_arr.size < 1 or not np.isfinite(om_arr).all():
        return None
    n_freq = om_arr.size

    last_time = float(discard_time) + float(n_samples - 1) * float(dt)
    if not np.isfinite(last_time):
        return None

    params = _extract_params(cfg, last_time)
    if params is None:
        return None
    sigma, rho, beta, rtol, atol_vec, max_step, first_step = params

    out_real = np.empty((n_cond, 3, n_freq), dtype=np.float64)
    out_imag = np.empty((n_cond, 3, n_freq), dtype=np.float64)
    if return_moments:
        out_moments = np.empty((n_cond, 10), dtype=np.float64)
        mom_ptr = out_moments.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        out_moments = None
        mom_ptr = ctypes.POINTER(ctypes.c_double)()

    rc = lib.lorenz_dop853_dual_freq_block(
        y0_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        fv1_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        fp1_ptr,
        ctypes.c_int(has_fp1),
        ctypes.c_double(float(omega1)),
        fv2_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        fp2_ptr,
        ctypes.c_int(has_fp2),
        ctypes.c_double(float(omega2)),
        ctypes.c_size_t(n_cond),
        ctypes.c_double(float(discard_time)),
        ctypes.c_double(float(dt)),
        ctypes.c_size_t(n_samples),
        om_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(n_freq),
        ctypes.c_double(sigma),
        ctypes.c_double(rho),
        ctypes.c_double(beta),
        ctypes.c_double(rtol),
        atol_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(max_step),
        ctypes.c_double(first_step),
        out_real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        mom_ptr,
    )
    if not _check_rc(rc):
        return None
    z_out = out_real + 1j * out_imag
    if return_moments:
        return z_out, out_moments
    return z_out
