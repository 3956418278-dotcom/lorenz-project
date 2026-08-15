import numpy as np

from lorenz.storage_scale import (
    StoragePlan,
    condition_count,
    cycles_for,
    dense_sample_count,
    plan_from_config,
    storage_report,
)


def _plan():
    return StoragePlan(
        block_count=2,
        direction_count=1,
        frequencies=(1.0,),
        strengths=(0.5,),
        minimum_cycles=2,
        minimum_physical_time=1.0,
        discard_time=10.0,
        dt=0.5,
        n_phase=32,
        harmonic_count=6,
        spectrum_bins=256,
        n_theta_bins=256,
        welch_segment=512,
        dense_trajectory_blocks=2,
        per_cycle_fourier_blocks=2,
        phase_samples_blocks=2,
    )


def test_cycle_and_dense_counts_follow_the_observation_rule():
    # minimum_physical_time=1.0 at omega=1 -> ceil(1/(2*pi)) = 1 cycle
    assert cycles_for(1.0, 2, 1.0) == 2
    # at omega=8, 400 physical time dominates: ceil(400*8/(2*pi)) = 510
    assert cycles_for(8.0, 64, 400.0) == 510
    period = 2 * np.pi
    assert dense_sample_count(1.0, 2, 0.5) == int(np.floor(2 * period / 0.5)) + 1
    assert condition_count(4, 1) == 9


def test_storage_report_matches_the_documented_formulas():
    report = storage_report(_plan())
    row = report["frequencies"][0]
    blocks = 2
    n_cond = 3
    n_cycle = 2
    n_dense = dense_sample_count(1.0, n_cycle, 0.5)
    assert row["dense_trajectories"] == blocks * n_cond * 3 * n_dense * 8
    assert row["phase_samples"] == blocks * n_cond * n_cycle * 32 * 3 * 8
    assert row["per_cycle_fourier"] == blocks * n_cond * n_cycle * 3 * 6 * 16
    assert row["block_spectra"] == blocks * n_cond * 3 * 256 * 16
    derived = (
        blocks * n_cond * 3 * 6 * 16
        + blocks * n_cond * 3 * 6 * 2 * 8
        + blocks * n_cond * 3 * 256 * 8
        + blocks * n_cond * 3 * 256 * 8
        + blocks * n_cond * 8
    )
    assert row["derived_summaries"] == derived
    grids = (
        n_cycle * 32 * 8 + n_dense * 8 + 256 * 8 + n_cond * 5 * 8
    )
    assert row["identity_and_grids"] == grids
    totals = report["totals"]
    assert totals["dense_trajectories"] == row["dense_trajectories"]
    assert totals["total"] == sum(row[key] for key in (
        "dense_trajectories", "phase_samples", "per_cycle_fourier",
        "block_spectra", "derived_summaries", "identity_and_grids",
    ))


def test_storage_report_respects_retention_counts():
    from dataclasses import replace

    report = storage_report(
        replace(
            _plan(),
            dense_trajectory_blocks=None,
            per_cycle_fourier_blocks=0,
            phase_samples_blocks="all",
        )
    )
    row = report["frequencies"][0]
    assert row["dense_trajectories"] == 0
    assert row["per_cycle_fourier"] == 0
    assert row["phase_samples"] > 0
    assert report["plan"]["retention_blocks"]["dense_trajectories"] == 0
    assert report["plan"]["retention_blocks"]["phase_samples"] == 2


def test_plan_from_config_reads_production_parameters():
    config = {
        "block_count": 64,
        "frequencies": [0.5, 8.0],
        "strengths": [0.25, 0.5, 1.0, 2.0],
        "observation_rule": {"minimum_cycles": 64, "minimum_physical_time": 400.0},
        "discard_time": 160.0,
        "n_phase": 32,
        "harmonics": [0, 1, 2, 3, 4, 5],
        "dense": {
            "dt": 0.05,
            "n_theta_bins": 256,
            "welch_segment": 512,
            "max_psd_bins": 256,
            "raw_segment_blocks": 2,
            "raw_segment_samples": 1200,
        },
        "retention": {
            "per_cycle_fourier_blocks": "all",
            "phase_samples_blocks": "all",
            "dense_trajectory_blocks": None,
            "block_spectra": True,
        },
    }
    plan = plan_from_config(config)
    assert plan.block_count == 64
    assert plan.frequencies == (0.5, 8.0)
    assert plan.harmonic_count == 6
    assert plan.dense_trajectory_blocks is None
    assert plan.per_cycle_fourier_blocks == 64
    report = storage_report(plan)
    assert len(report["frequencies"]) == 2
    assert report["frequencies"][0]["cycles"] == 64
    assert report["frequencies"][1]["cycles"] == 510
