"""Forcing-direction designs and Lorenz-equivariant tensor sectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .response import monochromatic_quadratic_design


INPUT_PAIR_ORDER = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
LORENZ_PARITY = np.asarray((-1, -1, 1), dtype=int)
STATE_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class DirectionDesign:
    """A fixed unit-vector design with reconstruction diagnostics."""

    name: str
    direction_names: tuple[str, ...]
    directions: np.ndarray
    linear_rank: int
    quadratic_rank: int
    linear_condition_number: float
    quadratic_condition_number: float
    linear_residual_degrees_of_freedom: int
    quadratic_residual_degrees_of_freedom: int


def build_direction_design(name: str, direction_names, directions) -> DirectionDesign:
    """Build and diagnose a named unit-vector direction design."""
    directions = np.asarray(directions, dtype=float)
    direction_names = tuple(direction_names)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape (n_direction, 3)")
    if len(direction_names) != len(directions) or len(set(direction_names)) != len(
        direction_names
    ):
        raise ValueError("direction names must be unique and match the direction rows")
    if not np.isfinite(directions).all():
        raise ValueError("directions must be finite")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1.0):
        raise ValueError("directions must be unit vectors")
    normalized_lines = []
    for direction in directions:
        first = int(np.flatnonzero(np.abs(direction) > 1e-14)[0])
        oriented = direction if direction[first] > 0 else -direction
        normalized_lines.append(tuple(np.round(oriented, 14)))
    if len(set(normalized_lines)) != len(normalized_lines):
        raise ValueError("directions must be unique up to global sign")

    quadratic = monochromatic_quadratic_design(directions)
    linear_rank = int(np.linalg.matrix_rank(directions))
    quadratic_rank = int(np.linalg.matrix_rank(quadratic))
    return DirectionDesign(
        name=name,
        direction_names=direction_names,
        directions=directions,
        linear_rank=linear_rank,
        quadratic_rank=quadratic_rank,
        linear_condition_number=float(np.linalg.cond(directions)),
        quadratic_condition_number=float(np.linalg.cond(quadratic)),
        linear_residual_degrees_of_freedom=int(len(directions) - linear_rank),
        quadratic_residual_degrees_of_freedom=int(len(directions) - quadratic_rank),
    )


def minimal_quadratic_direction_design() -> DirectionDesign:
    """Return the six-direction full-rank exploratory reconstruction design."""
    inverse_sqrt_two = 1 / np.sqrt(2)
    return build_direction_design(
        "D6",
        ("x", "y", "z", "xy_plus", "xz_plus", "yz_plus"),
        (
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (inverse_sqrt_two, inverse_sqrt_two, 0),
            (inverse_sqrt_two, 0, inverse_sqrt_two),
            (0, inverse_sqrt_two, inverse_sqrt_two),
        ),
    )


def balanced_quadratic_direction_design() -> DirectionDesign:
    """Return the nine-direction sum/difference design with residual checks."""
    inverse_sqrt_two = 1 / np.sqrt(2)
    return build_direction_design(
        "D9",
        (
            "x",
            "y",
            "z",
            "xy_plus",
            "xy_minus",
            "xz_plus",
            "xz_minus",
            "yz_plus",
            "yz_minus",
        ),
        (
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (inverse_sqrt_two, inverse_sqrt_two, 0),
            (inverse_sqrt_two, -inverse_sqrt_two, 0),
            (inverse_sqrt_two, 0, inverse_sqrt_two),
            (inverse_sqrt_two, 0, -inverse_sqrt_two),
            (0, inverse_sqrt_two, inverse_sqrt_two),
            (0, inverse_sqrt_two, -inverse_sqrt_two),
        ),
    )


def direction_identity(direction) -> tuple[str, str]:
    """Return a readable label and filesystem-safe slug for one direction."""
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("direction must be a finite three-vector")
    nonzero = np.flatnonzero(np.abs(direction) > 1e-12)
    if len(nonzero) == 1 and np.isclose(direction[nonzero[0]], 1.0):
        name = STATE_NAMES[nonzero[0]]
        return name, name
    if (
        len(nonzero) == 2
        and np.allclose(direction[nonzero], 2 ** -0.5, atol=1e-12)
    ):
        left, right = (STATE_NAMES[index] for index in nonzero)
        return f"({left}+{right})/sqrt(2)", f"{left}_plus_{right}"
    label = "".join(
        f"{direction[index]:+g}{STATE_NAMES[index]}" for index in nonzero
    )
    slug = label.replace("+", "plus_").replace("-", "minus_").replace(".", "p")
    return label, slug


def lorenz_linear_sector_mask() -> np.ndarray:
    """Return allowed ``(observable, input)`` entries under Lorenz parity."""
    return LORENZ_PARITY[:, None] * LORENZ_PARITY[None, :] == 1


def lorenz_effective_quadratic_sector_mask() -> np.ndarray:
    """Return allowed ``(observable, symmetric-input-pair)`` entries."""
    pair_parity = np.asarray(
        [LORENZ_PARITY[first] * LORENZ_PARITY[second] for first, second in INPUT_PAIR_ORDER]
    )
    return LORENZ_PARITY[:, None] * pair_parity[None, :] == 1
