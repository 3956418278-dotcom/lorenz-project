import numpy as np

from lorenz.direction_design import (
    INPUT_PAIR_ORDER,
    balanced_quadratic_direction_design,
    build_direction_design,
    direction_identity,
    lorenz_effective_quadratic_sector_mask,
    lorenz_linear_sector_mask,
    minimal_quadratic_direction_design,
)


def test_public_direction_design_builder_owns_rank_diagnostics():
    expected = minimal_quadratic_direction_design()
    built = build_direction_design(
        "configured", expected.direction_names, expected.directions
    )

    assert built.name == "configured"
    assert built.linear_rank == 3
    assert built.quadratic_rank == 6
    assert built.quadratic_residual_degrees_of_freedom == 0


def test_d6_is_minimal_full_rank_without_quadratic_residuals():
    design = minimal_quadratic_direction_design()

    assert design.name == "D6"
    assert design.directions.shape == (6, 3)
    assert design.linear_rank == 3
    assert design.quadratic_rank == 6
    assert design.linear_residual_degrees_of_freedom == 3
    assert design.quadratic_residual_degrees_of_freedom == 0
    np.testing.assert_allclose(design.linear_condition_number, np.sqrt(2))
    np.testing.assert_allclose(
        design.quadratic_condition_number, (3 + np.sqrt(5)) / 2
    )


def test_d9_is_balanced_and_has_reconstruction_residuals():
    design = balanced_quadratic_direction_design()

    assert design.directions.shape == (9, 3)
    assert design.linear_rank == 3
    assert design.quadratic_rank == 6
    assert design.linear_residual_degrees_of_freedom == 6
    assert design.quadratic_residual_degrees_of_freedom == 3
    np.testing.assert_allclose(design.linear_condition_number, 1.0)
    np.testing.assert_allclose(design.quadratic_condition_number, np.sqrt(2))


def test_direction_identity_uses_the_design_coordinate_convention():
    assert direction_identity([1, 0, 0]) == ("x", "x")
    assert direction_identity([2**-0.5, 0, 2**-0.5]) == (
        "(x+z)/sqrt(2)",
        "x_plus_z",
    )


def test_lorenz_sector_masks_recover_known_x_direction_null_controls():
    linear = lorenz_linear_sector_mask()
    quadratic = lorenz_effective_quadratic_sector_mask()

    assert linear.shape == (3, 3)
    assert quadratic.shape == (3, 6)
    assert int(linear.sum()) == 5
    assert int(quadratic.sum()) == 8

    x_input = 0
    xx_pair = INPUT_PAIR_ORDER.index((0, 0))
    np.testing.assert_array_equal(linear[:, x_input], [True, True, False])
    np.testing.assert_array_equal(quadratic[:, xx_pair], [False, False, True])
