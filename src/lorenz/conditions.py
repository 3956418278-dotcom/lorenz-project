"""Forcing designs and stable condition-axis identity, independent of retention."""
from __future__ import annotations
import numpy as np
STATE_NAMES = ("x", "y", "z")

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

def condition_table(forcing_vectors) -> dict[str, np.ndarray]:
    """Build identity arrays aligned with a condition axis.

    ``forcing_vectors`` has axes ``condition, state``.  The first condition
    is conventionally the unforced reference.  The returned arrays allow
    machine pairing of the same block across sign, strength, frequency,
    direction, and unforced reference without reconstructing identity from
    filenames.
    """
    forcing_vectors = np.asarray(forcing_vectors, dtype=float)
    if (
        forcing_vectors.ndim != 2
        or forcing_vectors.shape[0] < 1
        or forcing_vectors.shape[1] != 3
        or not np.isfinite(forcing_vectors).all()
    ):
        raise ValueError("forcing_vectors must have finite axes condition,state")
    norms = np.linalg.norm(forcing_vectors, axis=1)
    reference = None
    for forcing in forcing_vectors[1:]:
        if float(np.linalg.norm(forcing)) > 0:
            reference = forcing
            break
    if reference is None:
        sign = np.zeros(len(forcing_vectors), dtype=np.int8)
        signed_strength = np.zeros(len(forcing_vectors), dtype=float)
    else:
        projection = forcing_vectors @ reference / float(
            np.linalg.norm(reference)
        )
        collinear = np.isclose(np.abs(projection), norms)
        sign = np.where(
            collinear & (norms > 0), np.sign(projection), 0
        ).astype(np.int8)
        # Forced conditions not collinear with the reference direction carry
        # a canonical positive strength; the direction array identifies them.
        signed_strength = np.where(
            sign == 0, np.where(norms > 0, norms, 0.0), sign * norms
        )
    directions = np.zeros_like(forcing_vectors)
    nonzero = norms > 0
    directions[nonzero] = forcing_vectors[nonzero] / norms[nonzero, None]
    return {
        "sign": sign,
        "signed_strength": signed_strength,
        "direction": directions,
    }


def condition_labels(forcing_vectors) -> tuple[str, ...]:
    """Human-readable labels for one condition axis."""
    table = condition_table(forcing_vectors)
    labels = ["unforced"]
    for index in range(1, len(forcing_vectors)):
        signed = float(table["signed_strength"][index])
        labels.append(f"{'+' if signed > 0 else '-'}{abs(signed):g}")
    return tuple(labels)


def paired_condition_vectors(
    directions, strengths, *, include_unforced: bool = True
) -> np.ndarray:
    """Build the canonical unforced/paired-forcing condition axis.

    Forced conditions are ordered by direction, then increasing strength,
    then positive/negative sign. This is the shared condition-axis contract
    used by production generation, extension merging, and artifact views.
    """
    directions = np.asarray(directions, dtype=float)
    strengths = np.asarray(strengths, dtype=float)
    if (
        directions.ndim != 2
        or directions.shape[1] != 3
        or not np.isfinite(directions).all()
    ):
        raise ValueError("directions must have finite axes direction,state")
    if strengths.ndim != 1 or not np.isfinite(strengths).all():
        raise ValueError("strengths must be a finite one-dimensional array")
    return paired_direction_strength_vectors(
        directions,
        [strengths] * len(directions),
        include_unforced=include_unforced,
    )


def paired_direction_strength_vectors(
    directions, strengths_by_direction, *, include_unforced: bool = True
) -> np.ndarray:
    """Build a paired condition axis with a strength set per direction.

    The ordering remains direction, then increasing strength, then positive/
    negative sign, so the resulting array has the same condition-axis contract
    as :func:`paired_condition_vectors`.
    """
    directions = np.asarray(directions, dtype=float)
    if (
        directions.ndim != 2
        or directions.shape[1] != 3
        or not np.isfinite(directions).all()
    ):
        raise ValueError("directions must have finite axes direction,state")
    if len(strengths_by_direction) != len(directions):
        raise ValueError("strengths_by_direction must match the direction axis")
    resolved_strengths = []
    for strengths in strengths_by_direction:
        values = np.asarray(strengths, dtype=float)
        if (
            values.ndim != 1
            or not len(values)
            or not np.isfinite(values).all()
            or np.any(values <= 0)
            or np.any(np.diff(values) <= 0)
        ):
            raise ValueError(
                "each direction strength set must contain finite, positive, "
                "strictly increasing values"
            )
        resolved_strengths.append(values)
    vectors = [np.zeros(3)] if include_unforced else []
    vectors.extend(
        sign * strength * direction
        for direction, strengths in zip(directions, resolved_strengths)
        for strength in strengths
        for sign in (1.0, -1.0)
    )
    return np.asarray(vectors, dtype=float)


def mirrored_phase_pair_conditions(
    directions,
    direction_names,
    scales,
    base_amplitudes_by_direction,
    *,
    reference_phase: float = 0.0,
    phase_offset: float = np.pi / 4,
    axis_strengths=None,
) -> dict:
    """Build the mirrored two-axis forcing design used to isolate mixed chi2.

    For active axes ``j,k`` the two conditions use equal amplitudes but swapped
    component phases

        P+ : (theta + pi/4, theta - pi/4)
        P- : (theta - pi/4, theta + pi/4).

    At harmonic ``2*omega`` the diagonal quadratic terms from one component
    acquire opposite phase under this swap and cancel in ``(P+ + P-)/2``.
    The mixed product is unchanged: for ``phi=pi/4``,
    ``sin(theta+phi)*sin(theta-phi) = -cos(2*theta)/2``.  After subtracting the
    matched unforced coefficient, the surviving Fourier coefficient therefore
    satisfies, under the project's no-factorial Volterra convention,

        chi2_jk = -2 * Z_cross / (a_j * a_k),
        Z_cross = (Z_P+ + Z_P-)/2 - Z0.

    This design isolates the effective input-symmetric mixed coefficient visible
    to the same-frequency experiment; it does not establish a general symmetry
    of the ordered-frequency susceptibility.

    ``axis_strengths`` optionally maps a single-axis name (``x``, ``y`` or
    ``z``) to strictly increasing positive strengths.  Those diagonal
    single-axis conditions are appended after every mirrored pair, use the
    reference phase on all three components, and are described by records in
    the returned ``axis_records``.
    """
    directions = np.asarray(directions, dtype=float)
    names = tuple(str(value) for value in direction_names)
    if (
        directions.ndim != 2
        or directions.shape[1] != 3
        or not np.isfinite(directions).all()
        or len(names) != len(directions)
        or len(set(names)) != len(names)
    ):
        raise ValueError("mixed directions and unique names must share a finite direction axis")
    scales = np.asarray(scales, dtype=float)
    if (
        scales.ndim != 1
        or len(scales) < 3
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
        or np.any(np.diff(scales) <= 0)
    ):
        raise ValueError("mixed scales must be positive and strictly increasing")
    if len(base_amplitudes_by_direction) != len(directions):
        raise ValueError("base amplitudes must match the direction axis")
    if not np.isfinite(reference_phase) or not np.isfinite(phase_offset):
        raise ValueError("mixed forcing phases must be finite")
    if not np.isclose(phase_offset, np.pi / 4, rtol=0.0, atol=1e-15):
        raise ValueError("mirrored second-order probes require phase_offset=pi/4")

    vectors = [np.zeros(3, dtype=float)]
    phases = [np.full(3, reference_phase, dtype=float)]
    labels = ["unforced"]
    records = []
    axis_names = ("x", "y", "z")
    for direction, name, configured_base in zip(
        directions, names, base_amplitudes_by_direction
    ):
        active = np.flatnonzero(direction != 0)
        if len(active) != 2 or not np.all(direction[active] == 1.0):
            raise ValueError("each mixed direction must select exactly two axes with value 1")
        if name != "".join(axis_names[index] for index in active):
            raise ValueError("mixed direction names must identify their two active axes")
        base = np.asarray(configured_base, dtype=float)
        if (
            base.shape != (2,)
            or not np.isfinite(base).all()
            or np.any(base <= 0)
        ):
            raise ValueError("each mixed direction requires two positive base amplitudes")
        for scale in scales:
            pair = scale * base
            vector = np.zeros(3, dtype=float)
            vector[active] = pair
            plus_phase = np.full(3, reference_phase, dtype=float)
            minus_phase = np.full(3, reference_phase, dtype=float)
            plus_phase[active] = reference_phase + np.array((phase_offset, -phase_offset))
            minus_phase[active] = reference_phase + np.array((-phase_offset, phase_offset))
            plus_index = len(vectors)
            vectors.extend((vector.copy(), vector.copy()))
            phases.extend((plus_phase, minus_phase))
            pair_label = f"{name}:{pair[0]:g},{pair[1]:g}"
            labels.extend((f"{pair_label}:P+", f"{pair_label}:P-"))
            records.append(
                {
                    "name": name,
                    "axes": [axis_names[index] for index in active],
                    "axis_indices": active.tolist(),
                    "scale": float(scale),
                    "base_amplitudes": base.tolist(),
                    "amplitudes": pair.tolist(),
                    "mirror_plus_component_phases": plus_phase[active].tolist(),
                    "mirror_minus_component_phases": minus_phase[active].tolist(),
                    "mirror_plus_index": plus_index,
                    "mirror_minus_index": plus_index + 1,
                }
            )
    axis_records = []
    if axis_strengths is not None:
        if not isinstance(axis_strengths, dict):
            raise ValueError("axis_strengths must be a mapping from axis name to strengths")
        if not set(axis_strengths) <= set(axis_names):
            raise ValueError("axis_strengths keys must be x, y, or z")
        for axis_name in axis_names:
            if axis_name not in axis_strengths:
                continue
            values = np.asarray(axis_strengths[axis_name], dtype=float)
            if (
                values.ndim != 1
                or len(values) < 1
                or not np.isfinite(values).all()
                or np.any(values <= 0)
                or np.any(np.diff(values) <= 0)
            ):
                raise ValueError(
                    "each axis_strengths entry must be finite, positive, "
                    "and strictly increasing"
                )
            axis_index = axis_names.index(axis_name)
            indices = []
            for value in values:
                vector = np.zeros(3, dtype=float)
                vector[axis_index] = float(value)
                indices.append(len(vectors))
                vectors.append(vector)
                phases.append(np.full(3, reference_phase, dtype=float))
                labels.append(f"{axis_name}:{float(value):g}")
            axis_records.append(
                {
                    "name": axis_name,
                    "axis": axis_name,
                    "strengths": values.tolist(),
                    "condition_indices": indices,
                }
            )
    return {
        "forcing_vectors": np.asarray(vectors, dtype=float),
        "forcing_phases": np.asarray(phases, dtype=float),
        "condition_labels": tuple(labels),
        "scales": scales.copy(),
        "pairs": records,
        "axis_records": axis_records,
    }


def condition_index(forcing_vectors, target, *, atol: float = 1e-9) -> int:
    """Return the unique condition-axis index matching a forcing vector."""
    forcing_vectors = np.asarray(forcing_vectors, dtype=float)
    target = np.asarray(target, dtype=float)
    if forcing_vectors.ndim != 2 or target.shape != (forcing_vectors.shape[1],):
        raise ValueError("forcing_vectors and target have incompatible state axes")
    matches = np.flatnonzero(
        np.all(np.isclose(forcing_vectors, target, rtol=0.0, atol=atol), axis=1)
    )
    if len(matches) != 1:
        raise ValueError(f"expected one condition matching {target}, found {len(matches)}")
    return int(matches[0])


def paired_condition_indices(
    forcing_vectors, direction, strength: float, *, atol: float = 1e-9
) -> tuple[int, int]:
    """Return the positive/negative indices for one direction and strength."""
    direction = np.asarray(direction, dtype=float)
    strength = float(strength)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("direction must be a finite three-vector")
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("strength must be finite and positive")
    return (
        condition_index(forcing_vectors, strength * direction, atol=atol),
        condition_index(forcing_vectors, -strength * direction, atol=atol),
    )


def condition_axis_permutation(source_vectors, target_vectors) -> np.ndarray:
    """Map a source condition axis onto the same vectors in target order."""
    source_vectors = np.asarray(source_vectors, dtype=float)
    target_vectors = np.asarray(target_vectors, dtype=float)
    if source_vectors.shape != target_vectors.shape:
        raise ValueError("source and target condition axes must have equal shapes")
    permutation = np.asarray(
        [condition_index(source_vectors, target) for target in target_vectors],
        dtype=int,
    )
    if len(set(permutation.tolist())) != len(source_vectors):
        raise ValueError("source and target condition axes are not one-to-one")
    return permutation
