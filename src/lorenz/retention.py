"""Current production retains condition/block Fourier means and restart chunks."""
from __future__ import annotations
from dataclasses import dataclass
import os
import numpy as np
from .artifacts import write_npz_atomic

@dataclass(frozen=True)
class RetentionPolicy:
    chunk_blocks: int

def parse_retention_policy(config: dict, block_count: int) -> RetentionPolicy:
    retention = config.get("retention") or {}
    if not isinstance(retention, dict):
        raise ValueError("retention must be a mapping")
    for name in ("per_cycle_fourier_blocks", "phase_samples_blocks", "dense_trajectory_blocks", "spectrum_blocks"):
        if retention.get(name) != 0 or isinstance(retention.get(name), bool):
            raise ValueError(f"production requires retention.{name}=0; archival retention modes are not supported")
    if retention.get("block_spectra", False):
        raise ValueError("retention.block_spectra is not supported in production runs")
    if config.get("generate_figures") is not False:
        raise ValueError("production requires generate_figures=false")
    if config.get("extension") is not None:
        raise ValueError("extension artifacts are not supported")
    count = retention.get("chunk_blocks", 8)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("retention.chunk_blocks must be a positive integer")
    return RetentionPolicy(chunk_blocks=count)

def chunk_ranges(block_count: int, chunk_blocks: int):
    """Yield ``(start, end)`` block index ranges covering the block axis."""
    block_count = int(block_count)
    chunk_blocks = int(chunk_blocks)
    if block_count < 1 or chunk_blocks < 1:
        raise ValueError("block counts must be positive")
    start = 0
    while start < block_count:
        end = min(start + chunk_blocks, block_count)
        yield start, end
        start = end


def chunk_file_name(prefix: str, start: int, end: int) -> str:
    """Stable chunk file name for one block range of one frequency cell."""
    return f"{prefix}_blocks_{start:06d}_{end:06d}.npz"


def write_block_level_chunk(path: str | os.PathLike[str], arrays) -> None:
    """Write one block-chunk archive of retained block-level objects."""
    write_npz_atomic(path, arrays)
