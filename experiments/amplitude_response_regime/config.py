"""Configuration for the independent amplitude-response-regime experiment."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_FREQUENCIES = (
    2.265625,
    1.0546875,
    0.765625,
    3.1484375,
)

DEFAULT_A_LIST = (
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.08,
    0.10,
    0.12,
)


@dataclass(frozen=True)
class RegimeConfig:
    """Top-level parameters specific to this experiment."""

    frequencies: tuple[float, ...] = DEFAULT_FREQUENCIES
    a_list: tuple[float, ...] = DEFAULT_A_LIST
    sample_rate: float = 32.0
    n_record_samples: int = 16384
    seed_offset: int = 2_000_000
    alpha: float = 0.05
