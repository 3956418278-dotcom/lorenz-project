#!/usr/bin/env python3
"""Compute and plot unforced Lorenz-63 natural spectra for x, y, and z.

The script integrates multiple independent unforced Lorenz-63 trajectories,
discards a spin-up interval, computes one-sided complex FFT coefficients for
each coordinate, and plots the sample standard deviation of those complex
coefficients across trajectories.
"""

from __future__ import annotations
from datetime import datetime
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "lorenz_unforced_xyz_spectrum_mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


FS = 16.0
SPINUP_TIME = 60.0
EFFECTIVE_TIME = 4096.0
N_TRAJ = 4096
SEED = 123456789      
##############################################################################
RK4_SUBSTEPS = 8
OUTPUT_DIR = Path("outputs/unforced_xyz_spectrum")

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0


def lorenz_rhs(state: np.ndarray) -> np.ndarray:
    """Return the unforced Lorenz-63 vector field for state.shape == (n, 3)."""
    x = state[:, 0]
    y = state[:, 1]
    z = state[:, 2]

    rhs = np.empty_like(state)
    rhs[:, 0] = SIGMA * (y - x)
    rhs[:, 1] = RHO * x - y - x * z
    rhs[:, 2] = x * y - BETA * z
    return rhs


def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    """Advance all trajectories by one fixed RK4 step."""
    k1 = lorenz_rhs(state)
    k2 = lorenz_rhs(state + 0.5 * dt * k1)
    k3 = lorenz_rhs(state + 0.5 * dt * k2)
    k4 = lorenz_rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def advance_one_sample(state: np.ndarray, internal_dt: float, rk4_substeps: int) -> np.ndarray:
    """Advance by one output sampling interval using RK4 substeps."""
    for _ in range(rk4_substeps):
        state = rk4_step(state, internal_dt)
    return state


def sample_count(duration: float, fs: float, name: str) -> int:
    """Convert a duration to an integer number of samples."""
    value = duration * fs
    count = int(round(value))
    if count <= 0:
        raise ValueError(f"{name} must produce at least one sample; got {value!r}")
    if not np.isclose(value, count, rtol=0.0, atol=1e-9):
        raise ValueError(f"{name} * FS must be an integer number of samples; got {value!r}")
    return count


def integrate_samples(
    *,
    n_traj: int,
    fs: float,
    spinup_time: float,
    effective_time: float,
    rk4_substeps: int,
    seed: int,
) -> np.ndarray:
    """Integrate trajectories and return samples with shape (n_traj, 3, n_samples)."""
    if n_traj < 2:
        raise ValueError("n_traj must be at least 2 for a sample standard deviation")
    if fs <= 0.0:
        raise ValueError("fs must be positive")
    if spinup_time < 0.0:
        raise ValueError("spinup_time must be non-negative")
    if effective_time <= 0.0:
        raise ValueError("effective_time must be positive")
    if rk4_substeps <= 0:
        raise ValueError("rk4_substeps must be positive")

    sample_dt = 1.0 / fs
    internal_dt = sample_dt / rk4_substeps
    n_spinup = int(round(spinup_time * fs))
    if not np.isclose(spinup_time * fs, n_spinup, rtol=0.0, atol=1e-9):
        raise ValueError(f"spinup_time * FS must be an integer number of samples; got {spinup_time * fs!r}")
    n_samples = sample_count(effective_time, fs, "effective_time")

    rng = np.random.default_rng(seed)
    state = rng.normal(0.0, 1.0, size=(n_traj, 3)).astype(np.float64)

    for _ in range(n_spinup):
        state = advance_one_sample(state, internal_dt, rk4_substeps)

    samples = np.empty((n_traj, 3, n_samples), dtype=np.float64)
    for i in range(n_samples):
        samples[:, :, i] = state
        state = advance_one_sample(state, internal_dt, rk4_substeps)

    return samples


def compute_complex_fft_statistics(samples: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return freq, mean complex FFT, and std of complex FFT coefficients."""
    n_traj = samples.shape[0]
    n_samples = samples.shape[-1]

    q = np.fft.rfft(samples, axis=-1) / n_samples
    if n_samples % 2 == 0:
        q[..., 1:-1] *= 2.0
    else:
        q[..., 1:] *= 2.0

    freq = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    mean_q = q.mean(axis=0)
    variance = np.sum(np.abs(q - mean_q[None, ...]) ** 2, axis=0) / (n_traj - 1)
    std_q = np.sqrt(np.maximum(variance, 0.0))

    return freq, mean_q, std_q


def add_run_caption(fig: plt.Figure, run_info: dict[str, float | int]) -> None:
    caption = (
        f"SEED={run_info['SEED']}, N_TRAJ={run_info['N_TRAJ']}, FS={run_info['FS']}, "
        f"SPINUP_TIME={run_info['SPINUP_TIME']}, EFFECTIVE_TIME={run_info['EFFECTIVE_TIME']}, "
        f"RK4_SUBSTEPS={run_info['RK4_SUBSTEPS']}"
    )
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9)


def plot_spectrum(
    *,
    freq: np.ndarray,
    mean_q: np.ndarray,
    std_q: np.ndarray,
    run_info: dict[str, float | int],
    output_path: Path,
    zoom: tuple[float, float] | None = None,
) -> None:
    labels = ("x", "y", "z")
    mask = freq > 0.0
    if zoom is not None:
        lo, hi = zoom
        mask &= (freq >= lo) & (freq <= hi)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis_index, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(
            freq[mask],
            std_q[axis_index, mask] / np.sqrt(run_info["N_TRAJ"]),
            linewidth=0.7,
            color="C0",
            label="std_Q / sqrt(N_TRAJ)",
            zorder=2,
        )
        ax.plot(
            freq[mask],
            np.abs(mean_q[axis_index, mask]),
            linewidth=0.8,
            color="C1",
            alpha=0.75,
            label="abs(mean_Q)",
            zorder=1,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{label} direction")
        ax.set_ylabel("FFT coefficient amplitude")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()

    if zoom is not None:
        axes[-1].set_xlim(*zoom)
    axes[-1].set_xlabel("Frequency")

    add_run_caption(fig, run_info)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_outputs(
    *,
    output_dir: Path,
    freq: np.ndarray,
    mean_q: np.ndarray,
    std_q: np.ndarray,
    run_info: dict[str, float | int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = make_run_tag(run_info)
    np.savez(
        output_dir / f"spectrum_data_{run_tag}.npz",
        freq=freq,
        std_x=std_q[0],
        std_y=std_q[1],
        std_z=std_q[2],
        mean_fft_x=mean_q[0],
        mean_fft_y=mean_q[1],
        mean_fft_z=mean_q[2],
    )

    plot_spectrum(
        freq=freq,
        mean_q=mean_q,
        std_q=std_q,
        run_info=run_info,
        output_path=output_dir / f"spectrum_full_{run_tag}.png"
    )
    plot_spectrum(
        freq=freq,
        mean_q=mean_q,
        std_q=std_q,
        run_info=run_info,
        output_path=output_dir / f"spectrum_zoom_0p5_4Hz_{run_tag}.png",
        zoom=(0.5, 4.0),
    )

    with (output_dir / f"run_info_{run_tag}.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, sort_keys=True)
        f.write("\n")

def filename_number(value: float | int) -> str:
    """Convert a numeric parameter to a compact filename-safe string."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def make_run_tag(run_info: dict[str, float | int]) -> str:

    return (
        f"seed{run_info['SEED']}"
        f"_ntraj{run_info['N_TRAJ']}"
        f"_fs{filename_number(run_info['FS'])}"
        f"_spin{filename_number(run_info['SPINUP_TIME'])}"
        f"_time{filename_number(run_info['EFFECTIVE_TIME'])}"
        f"_rk4sub{run_info['RK4_SUBSTEPS']}"
    )
def parse_seed_list(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer seed")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute unforced Lorenz-63 x/y/z natural spectra from complex FFT coefficients."
    )
    parser.add_argument("--n-traj", type=int, default=N_TRAJ)
    parser.add_argument("--fs", type=float, default=FS)
    parser.add_argument("--spinup-time", type=float, default=SPINUP_TIME)
    parser.add_argument("--effective-time", type=float, default=EFFECTIVE_TIME)
    parser.add_argument("--rk4-substeps", type=int, default=RK4_SUBSTEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--seeds",
        type=parse_seed_list,
        default=None,
        help="Optional comma-separated seeds, e.g. --seeds 0,1,2. Each seed writes to output-dir/seed_<seed>/.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def run_seed(args: argparse.Namespace, seed: int, output_dir: Path) -> None:
    start = time.perf_counter()

    samples = integrate_samples(
        n_traj=args.n_traj,
        fs=args.fs,
        spinup_time=args.spinup_time,
        effective_time=args.effective_time,
        rk4_substeps=args.rk4_substeps,
        seed=seed,
    )
    freq, mean_q, std_q = compute_complex_fft_statistics(samples, args.fs)

    elapsed_seconds = time.perf_counter() - start
    run_info: dict[str, float | int] = {
        "N_TRAJ": args.n_traj,
        "FS": args.fs,
        "SPINUP_TIME": args.spinup_time,
        "EFFECTIVE_TIME": args.effective_time,
        "N_SAMPLES": samples.shape[-1],
        "RK4_SUBSTEPS": args.rk4_substeps,
        "SEED": seed,
        "elapsed_seconds": elapsed_seconds,
    }
    write_outputs(
        output_dir=output_dir,
        freq=freq,
        mean_q=mean_q,
        std_q=std_q,
        run_info=run_info,
    )

    print(f"Wrote outputs to {output_dir}")
    print(f"std_Q shape: {std_q.shape}")


def main() -> None:
    args = parse_args()
    if args.seeds is None:
        run_seed(args, args.seed, args.output_dir)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        run_seed(args, seed, args.output_dir / f"seed_{seed}")
    seed_tag = "-".join(str(seed) for seed in args.seeds)
    with (
        args.output_dir / f"multi_seed_run_info_seeds{seed_tag}.json"
            
    ).open("w", encoding="utf-8") as f:
        json.dump({"SEEDS": args.seeds}, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()