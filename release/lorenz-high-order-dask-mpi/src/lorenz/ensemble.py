"""Reproducible proposal and spinup of Lorenz initial-state blocks."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.integrate import solve_ivp

from .core import check_solution, lorenz_rhs
from .parallel import map_tasks


@dataclass(frozen=True)
class SymmetricXYUniformProposal:
    """Independent uniform box with x/y intervals symmetric about zero."""

    x_half_width: float
    y_half_width: float
    z_bounds: tuple[float, float]

    def __post_init__(self):
        try:
            z_bounds = tuple(float(value) for value in self.z_bounds)
        except (TypeError, ValueError):
            raise ValueError("z_bounds must contain two finite values") from None
        object.__setattr__(self, "x_half_width", float(self.x_half_width))
        object.__setattr__(self, "y_half_width", float(self.y_half_width))
        object.__setattr__(self, "z_bounds", z_bounds)
        values = np.asarray(
            [self.x_half_width, self.y_half_width, *self.z_bounds], dtype=float
        )
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("proposal bounds must be finite")
        if self.x_half_width <= 0 or self.y_half_width <= 0:
            raise ValueError("x and y half-widths must be positive")
        if self.z_bounds[0] >= self.z_bounds[1]:
            raise ValueError("z_bounds must be strictly increasing")

    @property
    def lower(self) -> np.ndarray:
        return np.array(
            [-self.x_half_width, -self.y_half_width, self.z_bounds[0]],
            dtype=float,
        )

    @property
    def upper(self) -> np.ndarray:
        return np.array(
            [self.x_half_width, self.y_half_width, self.z_bounds[1]], dtype=float
        )


@dataclass(frozen=True)
class InitialStateBlocks:
    """Generated blocks and sufficient in-memory generation provenance."""

    block_ids: tuple[int, ...]
    raw_proposals: np.ndarray
    final_states: np.ndarray
    child_spawn_keys: tuple[tuple[int, ...], ...]
    root_entropy: int | tuple[int, ...]
    root_spawn_key: tuple[int, ...]
    bit_generator: str
    proposal: SymmetricXYUniformProposal
    spinup_time: float
    lorenz_parameters: Mapping[str, float]
    solver_options: Mapping[str, object]


def _checked_block_ids(block_ids) -> tuple[int, ...]:
    result = tuple(block_ids)
    if not result:
        raise ValueError("at least one block_id is required")
    if any(
        isinstance(block_id, (bool, np.bool_))
        or not isinstance(block_id, (int, np.integer))
        or block_id < 0
        or block_id > np.iinfo(np.uint32).max
        for block_id in result
    ):
        raise ValueError("block_ids must be unsigned 32-bit integers")
    result = tuple(int(block_id) for block_id in result)
    if len(set(result)) != len(result):
        raise ValueError("block_ids must be unique")
    return result


def _frozen_copy(array) -> np.ndarray:
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _spinup_initial_state(arguments) -> np.ndarray:
    initial_state, spinup_time, cfg = arguments
    sol = solve_ivp(
        lambda t, state: lorenz_rhs(t, state, cfg),
        [0.0, spinup_time],
        initial_state,
        t_eval=[spinup_time],
        **cfg["solver"],
    )
    check_solution(sol, (3, 1))
    return sol.y[:, -1]


def generate_initial_state_blocks(
    block_ids,
    root_seed,
    proposal: SymmetricXYUniformProposal,
    spinup_time: float,
    cfg: dict,
) -> InitialStateBlocks:
    """Generate order-independent blocks from named SeedSequence children.

    A nonnegative integer block ID is appended to the root ``spawn_key``. This
    gives each block a stable PCG64DXSM stream independent of request order.
    Finite spinup only approximates a draw from the physical measure; its bias
    remains a separate convergence question.
    """
    block_ids = _checked_block_ids(block_ids)
    if not isinstance(proposal, SymmetricXYUniformProposal):
        raise TypeError("proposal must be a SymmetricXYUniformProposal")
    if not np.isfinite(spinup_time) or spinup_time < 0:
        raise ValueError("spinup_time must be finite and nonnegative")

    root = np.random.SeedSequence(root_seed)
    raw_proposals = np.empty((len(block_ids), 3), dtype=float)
    final_states = np.empty_like(raw_proposals)
    child_spawn_keys = []
    for index, block_id in enumerate(block_ids):
        child = np.random.SeedSequence(
            entropy=root.entropy,
            spawn_key=tuple(root.spawn_key) + (block_id,),
            pool_size=root.pool_size,
        )
        child_spawn_keys.append(tuple(child.spawn_key))
        rng = np.random.Generator(np.random.PCG64DXSM(child))
        raw = rng.uniform(proposal.lower, proposal.upper)
        raw_proposals[index] = raw

    if spinup_time == 0:
        final_states[:] = raw_proposals
    else:
        spinup_time = float(spinup_time)
        final_states[:] = np.asarray(
            map_tasks(
                _spinup_initial_state,
                (
                    (raw, spinup_time, cfg)
                    for raw in raw_proposals
                ),
                # An active Dask runtime owns distribution; keep the existing
                # direct-call fallback serial when no runtime was initialized.
                workers=1,
            ),
            dtype=float,
        )

    entropy = root.entropy
    if isinstance(entropy, (list, tuple, np.ndarray)):
        entropy = tuple(int(value) for value in entropy)
    else:
        entropy = int(entropy)
    lorenz_parameters = MappingProxyType(
        {name: float(value) for name, value in cfg["lorenz"].items()}
    )
    solver_options = MappingProxyType(dict(cfg["solver"]))
    return InitialStateBlocks(
        block_ids=block_ids,
        raw_proposals=_frozen_copy(raw_proposals),
        final_states=_frozen_copy(final_states),
        child_spawn_keys=tuple(child_spawn_keys),
        root_entropy=entropy,
        root_spawn_key=tuple(root.spawn_key),
        bit_generator="PCG64DXSM",
        proposal=proposal,
        spinup_time=float(spinup_time),
        lorenz_parameters=lorenz_parameters,
        solver_options=solver_options,
    )


def generate_configured_initial_state_blocks(config: dict) -> InitialStateBlocks:
    """Generate the shared initial-state ensemble described by a run config.

    This adapter translates the repository's common ``initial_ensemble``
    contract into the lower-level block generator. The blocks can be shared by
    any experiment composition; they do not belong to a frequency or forcing
    direction pilot.
    """
    initial = config["initial_ensemble"]
    proposal_config = initial["proposal"]
    proposal = SymmetricXYUniformProposal(
        x_half_width=proposal_config["x_half_width"],
        y_half_width=proposal_config["y_half_width"],
        z_bounds=tuple(proposal_config["z_bounds"]),
    )
    block_id_start = int(initial["block_id_start"])
    block_count = int(config["block_count"])
    block_ids = tuple(range(block_id_start, block_id_start + block_count))
    numerical_config = {
        "lorenz": dict(config["lorenz"]),
        "solver": dict(config["solver"]),
    }
    return generate_initial_state_blocks(
        block_ids,
        initial["root_entropy"],
        proposal,
        float(initial["spinup_time"]),
        numerical_config,
    )
