import numpy as np
import pytest

from lorenz.ensemble import (
    SymmetricXYUniformProposal,
    generate_initial_state_blocks,
)


@pytest.fixture
def cfg():
    return {
        "lorenz": {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        "solver": {"rtol": 1e-8, "atol": 1e-10},
    }


@pytest.fixture
def proposal():
    return SymmetricXYUniformProposal(
        x_half_width=2.0, y_half_width=4.0, z_bounds=(1.0, 7.0)
    )


def _by_id(blocks, values):
    return {block_id: values[index] for index, block_id in enumerate(blocks.block_ids)}


def test_proposal_encodes_symmetric_xy_and_explicit_z_bounds(proposal):
    np.testing.assert_array_equal(proposal.lower, [-2.0, -4.0, 1.0])
    np.testing.assert_array_equal(proposal.upper, [2.0, 4.0, 7.0])
    np.testing.assert_array_equal(proposal.lower[:2], -proposal.upper[:2])


def test_proposal_canonicalizes_mutable_bounds_input():
    z_bounds = [1, 3]
    proposal = SymmetricXYUniformProposal(2, 4, z_bounds)
    z_bounds[0] = -100

    assert proposal.z_bounds == (1.0, 3.0)
    assert isinstance(proposal.z_bounds, tuple)


def test_blocks_reproduce_by_immutable_id_independent_of_request_order(
    cfg, proposal
):
    first = generate_initial_state_blocks(
        [7, 2, 19], 12345, proposal, spinup_time=0.02, cfg=cfg
    )
    reordered = generate_initial_state_blocks(
        [19, 7, 2], 12345, proposal, spinup_time=0.02, cfg=cfg
    )

    first_raw = _by_id(first, first.raw_proposals)
    reordered_raw = _by_id(reordered, reordered.raw_proposals)
    first_final = _by_id(first, first.final_states)
    reordered_final = _by_id(reordered, reordered.final_states)
    for block_id in first.block_ids:
        np.testing.assert_array_equal(first_raw[block_id], reordered_raw[block_id])
        np.testing.assert_array_equal(
            first_final[block_id], reordered_final[block_id]
        )

    assert first.bit_generator == "PCG64DXSM"
    assert first.root_entropy == 12345
    assert first.lorenz_parameters == cfg["lorenz"]
    assert first.solver_options == cfg["solver"]


def test_each_block_has_unique_spawn_lineage_and_bounded_raw_proposal(
    cfg, proposal
):
    blocks = generate_initial_state_blocks(
        [0, 1, 8], [10, 20], proposal, spinup_time=0.01, cfg=cfg
    )

    assert blocks.child_spawn_keys == ((0,), (1,), (8,))
    assert len(set(blocks.child_spawn_keys)) == len(blocks.block_ids)
    assert np.all(blocks.raw_proposals >= proposal.lower)
    assert np.all(blocks.raw_proposals < proposal.upper)
    assert len(np.unique(blocks.raw_proposals, axis=0)) == len(blocks.block_ids)


def test_raw_proposals_and_post_spinup_states_are_separate(cfg, proposal):
    blocks = generate_initial_state_blocks(
        [3, 4], 789, proposal, spinup_time=0.05, cfg=cfg
    )

    assert not np.shares_memory(blocks.raw_proposals, blocks.final_states)
    assert not blocks.raw_proposals.flags.writeable
    assert not blocks.final_states.flags.writeable
    assert np.any(blocks.raw_proposals != blocks.final_states)


@pytest.mark.parametrize("block_ids", [[], [1, 1], [-1], [True]])
def test_invalid_block_ids_are_rejected(cfg, proposal, block_ids):
    with pytest.raises(ValueError, match="block_id"):
        generate_initial_state_blocks(
            block_ids, 1, proposal, spinup_time=0.0, cfg=cfg
        )


@pytest.mark.parametrize("spinup_time", [-1.0, np.nan, np.inf])
def test_invalid_spinup_time_is_rejected(cfg, proposal, spinup_time):
    with pytest.raises(ValueError, match="spinup_time"):
        generate_initial_state_blocks([0], 1, proposal, spinup_time, cfg)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x_half_width": 0.0, "y_half_width": 1.0, "z_bounds": (0.0, 1.0)},
        {"x_half_width": 1.0, "y_half_width": -1.0, "z_bounds": (0.0, 1.0)},
        {"x_half_width": 1.0, "y_half_width": 1.0, "z_bounds": (1.0, 1.0)},
        {"x_half_width": 1.0, "y_half_width": 1.0, "z_bounds": (2.0, 1.0)},
    ],
)
def test_invalid_proposal_bounds_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SymmetricXYUniformProposal(**kwargs)
