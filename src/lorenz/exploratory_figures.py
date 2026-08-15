"""Read-only scientific views of retained exploratory artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lorenz-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .artifacts import (
    active_source_identifiers,
    environment_provenance,
    file_sha256,
    git_provenance,
    verify_file_identifiers,
    write_json_atomic,
)
from .strength_series import fit_block_power_series


STATE_INDEX = {"x": 0, "y": 1, "z": 2}
COLORS = {"first": "#176B87", "second": "#B94A48", "higher": "#D28E2B"}


@dataclass(frozen=True)
class FigureArtifact:
    """One verified artifact and the JSON documents used for plotting."""

    name: str
    root: Path
    manifest_identifier: str
    manifest: dict[str, Any]
    config: dict[str, Any]
    derived: dict[str, Any]

    def raw_path(self) -> Path:
        matches = [name for name in self.manifest["files"] if name.endswith(".npz")]
        if len(matches) != 1:
            raise ValueError(f"{self.name} artifact must contain exactly one NPZ file")
        return self.root / matches[0]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_figure_artifact(
    repo_root: str | os.PathLike[str], name: str, specification: dict[str, Any]
) -> FigureArtifact:
    """Load an explicitly pinned artifact after schema and hash validation."""
    repo_root = Path(repo_root).resolve()
    root = (repo_root / specification["path"]).resolve()
    try:
        root.relative_to(repo_root)
    except ValueError as error:
        raise ValueError(f"artifact path escapes repository: {root}") from error
    manifest_path = root / "manifest.json"
    actual_manifest = file_sha256(manifest_path)
    if actual_manifest != specification["manifest_sha256"]:
        raise ValueError(f"{name} manifest hash mismatch")
    manifest = _json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{name} artifact schema is not supported")
    for field in ("classification", "study_id"):
        if manifest.get(field) != specification[field]:
            raise ValueError(f"{name} artifact {field} mismatch")
    verify_file_identifiers(root, manifest["files"])
    return FigureArtifact(
        name=name,
        root=root,
        manifest_identifier=f"sha256:{actual_manifest}",
        manifest=manifest,
        config=_json(root / "config_snapshot.json"),
        derived=_json(root / "derived_diagnostics.json"),
    )


def _artifact_source(artifact: FigureArtifact, repo_root: Path) -> dict[str, Any]:
    return {
        "name": artifact.name,
        "path": artifact.root.relative_to(repo_root).as_posix(),
        "manifest_identifier": artifact.manifest_identifier,
        "classification": artifact.manifest["classification"],
        "study_id": artifact.manifest["study_id"],
        "schema_version": artifact.manifest["schema_version"],
        "files": artifact.manifest["files"],
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.6,
        }
    )


def _strength_key(value: float) -> str:
    return str(float(value))


def _component(records: list[dict[str, Any]], observable: str) -> dict[str, Any]:
    index = STATE_INDEX[observable]
    for record in records:
        if record.get("observable") in (observable, index):
            return record
    raise KeyError(f"observable {observable} is absent")


def complex_block_summary(values: np.ndarray) -> dict[str, float]:
    """Return magnitude and radial block-mean SE for complex block values."""
    values = np.asarray(values)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("complex summary requires at least two finite block values")
    mean = values.mean()
    scale = np.sqrt(len(values))
    se_real = values.real.std(ddof=1) / scale
    se_imag = values.imag.std(ddof=1) / scale
    return {
        "mean_real": float(mean.real),
        "mean_imaginary": float(mean.imag),
        "mean_magnitude": float(abs(mean)),
        "radial_standard_error": float(np.hypot(se_real, se_imag)),
    }


def _errorbar_interval(
    axis, x, mean, lower, upper, *, color: str, label: str, marker: str = "o"
) -> None:
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if np.any(mean < lower - 1e-12) or np.any(mean > upper + 1e-12):
        raise ValueError("point estimate lies outside its stored interval")
    axis.errorbar(
        x,
        mean,
        yerr=np.vstack((mean - lower, upper - mean)),
        color=color,
        marker=marker,
        capsize=3,
        label=label,
    )


def _mark_identified(axis, x, y, identified, color: str) -> None:
    for x_value, y_value, flag in zip(x, y, identified):
        axis.scatter(
            [x_value],
            [y_value],
            s=34,
            marker="o",
            facecolor=color if flag else "white",
            edgecolor=color,
            linewidth=1.1,
            zorder=4,
        )


def _save_atomic(figure, path: Path, file_format: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, format=file_format, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_figure(
    figure,
    output_dir: Path,
    figure_id: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    png = output_dir / f"{figure_id}.png"
    pdf = output_dir / f"{figure_id}.pdf"
    _save_atomic(figure, png, "png")
    _save_atomic(figure, pdf, "pdf")
    plt.close(figure)
    source = {
        **source,
        "figure_id": figure_id,
        "outputs": {
            png.name: f"sha256:{file_sha256(png)}",
            pdf.name: f"sha256:{file_sha256(pdf)}",
        },
    }
    sidecar = output_dir / f"{figure_id}.source.json"
    write_json_atomic(sidecar, source)
    return {
        "figure_id": figure_id,
        "png": png.name,
        "pdf": pdf.name,
        "source": sidecar.name,
    }


def _heldout_a2(artifact: FigureArtifact, omega: float) -> tuple[np.ndarray, np.ndarray]:
    key = str(float(omega)).replace(".", "p")
    if key.endswith("p0"):
        key = key[:-2]
    with np.load(artifact.raw_path(), allow_pickle=False) as raw:
        strengths = np.asarray(raw["strengths"], dtype=float)
        positive = np.asarray(raw[f"omega_{key}_positive_cycle_fourier"]).mean(axis=-2)
        negative = np.asarray(raw[f"omega_{key}_negative_cycle_fourier"]).mean(axis=-2)
    return strengths, (positive + negative) / 2


def _strength_first_data(
    artifact: FigureArtifact, observable: str
) -> dict[str, Any]:
    index = STATE_INDEX[observable]
    strengths = np.asarray(artifact.derived["strengths"], dtype=float)
    rows = artifact.derived["target_series"]["odd_fundamental"]["raw_by_strength"]
    identification = artifact.derived["confirmed_block_bootstrap"]["identification"][
        "odd_fundamental"
    ]
    mean, se, lower, upper, identified = [], [], [], [], []
    for strength, row in zip(strengths, rows):
        summary = row["raw_contrast"]
        decision = _component(
            identification[_strength_key(strength)]["components"], observable
        )
        mean.append(summary["mean_magnitude"][index])
        se.append(summary["radial_standard_error"][index])
        lower.append(decision["magnitude_lower"])
        upper.append(decision["magnitude_upper"])
        identified.append(decision["simultaneous_interval_excludes_origin"])
    return {
        "strengths": strengths,
        "mean": np.asarray(mean),
        "radial_se": np.asarray(se),
        "lower": np.asarray(lower),
        "upper": np.asarray(upper),
        "identified": np.asarray(identified, dtype=bool),
    }


def _heldout_second_data(
    artifact: FigureArtifact, omega: float, observable: str
) -> dict[str, Any]:
    index = STATE_INDEX[observable]
    strengths, all_a2 = _heldout_a2(artifact, omega)
    a2 = all_a2[:, :, index, 2]
    identification = artifact.derived["second_harmonic_identification"][
        "by_frequency"
    ][_strength_key(omega)]["alternative_A2"]["identification"]
    summaries, lower, upper, identified = [], [], [], []
    for position, strength in enumerate(strengths):
        summaries.append(complex_block_summary(a2[:, position]))
        decision = _component(
            identification[_strength_key(strength)]["components"], observable
        )
        lower.append(decision["magnitude_lower"])
        upper.append(decision["magnitude_upper"])
        identified.append(decision["simultaneous_interval_excludes_origin"])
    return {
        "strengths": strengths,
        "block_values": a2,
        "mean": np.asarray([item["mean_magnitude"] for item in summaries]),
        "radial_se": np.asarray([item["radial_standard_error"] for item in summaries]),
        "lower": np.asarray(lower),
        "upper": np.asarray(upper),
        "identified": np.asarray(identified, dtype=bool),
    }


def plot_strength_scaling(
    output_dir: Path,
    strength_artifact: FigureArtifact,
    heldout_artifact: FigureArtifact,
    selection: dict[str, Any],
    common: dict[str, Any],
) -> dict[str, Any]:
    """Plot raw and strength-normalized first/second harmonic amplitudes."""
    first = _strength_first_data(
        strength_artifact, selection["fundamental_observable"]
    )
    second = _heldout_second_data(
        heldout_artifact, selection["omega"], selection["second_harmonic_observable"]
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.1), sharex="col")
    panels = (
        (axes[0, 0], first, 1, False, r"$|A_1|=|\widehat O_1|$", "state units", COLORS["first"]),
        (axes[0, 1], second, 1, False, r"$|A_2|=|(C_{+,2}+C_{-,2})/2|$", "state units", COLORS["second"]),
        (axes[1, 0], first, 1, True, r"$|A_1|/h$", "state / forcing unit", COLORS["first"]),
        (axes[1, 1], second, 2, True, r"$|A_2|/h^2$", r"state / forcing unit$^2$", COLORS["second"]),
    )
    for axis, data, power, normalized, ylabel, units, color in panels:
        divisor = data["strengths"] ** power if normalized else 1
        _errorbar_interval(
            axis,
            data["strengths"],
            data["mean"] / divisor,
            data["lower"] / divisor,
            data["upper"] / divisor,
            color=color,
            label="simultaneous 95% magnitude interval",
        )
        _mark_identified(
            axis,
            data["strengths"],
            data["mean"] / divisor,
            data["identified"],
            color,
        )
        axis.set_ylabel(ylabel + f" ({units})")
        axis.set_xlabel("forcing strength h (model units)")
    axes[0, 0].set_title("Fundamental, x output; B=256")
    axes[0, 1].set_title("Second harmonic, z output; held-out B=32")
    axes[0, 0].legend(loc="upper left")
    figure.suptitle(
        "Forcing-strength scaling at omega=2, x-direction forcing\n"
        "filled marker: stored full-family simultaneous CI excludes zero"
    )
    figure.tight_layout()
    plotted = {
        "strengths": first["strengths"],
        "fundamental": {key: first[key] for key in ("mean", "radial_se", "lower", "upper", "identified")},
        "second_harmonic_A2": {key: second[key] for key in ("mean", "radial_se", "lower", "upper", "identified")},
    }
    return _save_figure(
        figure,
        output_dir,
        "01_strength_scaling",
        {
            **common,
            "question": "Do the fundamental and second harmonic scale approximately as h and h^2?",
            "quantity": "A1=odd fundamental Fourier contrast; A2=forced-even second harmonic without U2 subtraction; Fourier exp(-i*n*theta)",
            "uncertainty": "stored whole-block simultaneous 95% complex-rectangle magnitude bounds; point values are block means",
            "sources": [common["artifact_sources"]["strength"], common["artifact_sources"]["heldout"]],
            "plotted_data": plotted,
        },
    )


def plot_identifiability_snr(
    output_dir: Path,
    strength_artifact: FigureArtifact,
    heldout_artifact: FigureArtifact,
    selection: dict[str, Any],
    common: dict[str, Any],
) -> dict[str, Any]:
    first = _strength_first_data(strength_artifact, selection["fundamental_observable"])
    second = _heldout_second_data(
        heldout_artifact, selection["omega"], selection["second_harmonic_observable"]
    )
    snr_first = first["mean"] / first["radial_se"]
    snr_second = second["mean"] / second["radial_se"]
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    axis.plot(first["strengths"], snr_first, color=COLORS["first"], marker="o", label="fundamental x (B=256)")
    axis.plot(second["strengths"], snr_second, color=COLORS["second"], marker="s", label="second harmonic z, A2 (held-out B=32)")
    _mark_identified(axis, first["strengths"], snr_first, first["identified"], COLORS["first"])
    _mark_identified(axis, second["strengths"], snr_second, second["identified"], COLORS["second"])
    axis.axhline(1.0, color="0.35", linestyle="--", label="|estimate| = radial SE")
    axis.set_yscale("log")
    axis.set_xlabel("forcing strength h (model units)")
    axis.set_ylabel(r"$|\overline{A}| / \sqrt{SE_{Re}^2+SE_{Im}^2}$")
    axis.set_title("Signal relative to across-block sampling noise at omega=2")
    axis.legend(ncol=2)
    figure.tight_layout()
    return _save_figure(
        figure,
        output_dir,
        "02_identifiability_snr",
        {
            **common,
            "question": "At which strengths does response exceed chaotic across-block variation?",
            "quantity": "complex mean magnitude divided by radial standard error; scaling by h or h^2 leaves this ratio unchanged",
            "uncertainty": "filled markers reproduce the stored full-family simultaneous 95% CI exclusion decision; SNR itself is descriptive",
            "sources": [common["artifact_sources"]["strength"], common["artifact_sources"]["heldout"]],
            "plotted_data": {
                "strengths": first["strengths"],
                "fundamental_snr": snr_first,
                "fundamental_simultaneously_identified": first["identified"],
                "second_harmonic_snr": snr_second,
                "second_harmonic_simultaneously_identified": second["identified"],
            },
        },
    )


def _strength_raw_odd(
    artifact: FigureArtifact, observable: str
) -> tuple[np.ndarray, np.ndarray]:
    index = STATE_INDEX[observable]
    with np.load(artifact.raw_path(), allow_pickle=False) as raw:
        strengths = np.asarray(raw["strengths"], dtype=float)
        harmonics = list(np.asarray(raw["harmonics"], dtype=int))
        harmonic = harmonics.index(1)
        positive = np.asarray(raw["positive_cycle_fourier"]).mean(axis=-2)
        negative = np.asarray(raw["negative_cycle_fourier"]).mean(axis=-2)
    return strengths, ((positive - negative) / 2)[:, :, index, harmonic]


def _nested_contributions(
    values: np.ndarray, strengths: np.ndarray, orders: tuple[int, int]
) -> dict[str, np.ndarray]:
    upper_strengths, leading, higher, ratio = [], [], [], []
    for stop in range(3, len(strengths) + 1):
        prefix = strengths[:stop]
        fit = fit_block_power_series(values[:, :stop, None], prefix, orders)
        upper = float(prefix[-1])
        low_values = fit.coefficients[:, 0, 0] * upper ** orders[0]
        high_values = fit.coefficients[:, 1, 0] * upper ** orders[1]
        low_mean = abs(low_values.mean())
        high_mean = abs(high_values.mean())
        upper_strengths.append(upper)
        leading.append(low_mean)
        higher.append(high_mean)
        ratio.append(high_mean / low_mean if low_mean > 0 else np.inf)
    return {
        "upper_strengths": np.asarray(upper_strengths),
        "leading": np.asarray(leading),
        "higher": np.asarray(higher),
        "point_ratio": np.asarray(ratio),
    }


def _adequacy_intervals(
    decisions: dict[str, Any], strengths: np.ndarray, observable: str, heldout: bool
) -> dict[str, np.ndarray]:
    low_lower, low_upper, high_lower, high_upper, ratio_upper, resolved = [], [], [], [], [], []
    for strength in strengths:
        decision = decisions[_strength_key(strength)]
        if heldout:
            low_records = decision["components"]["low"]
            high_records = decision["components"]["higher"]
        else:
            low_records = decision["low_order_components"]
            high_records = decision["higher_order_components"]
        low = _component(low_records, observable)
        high = _component(high_records, observable)
        low_lower.append(low["magnitude_lower"])
        low_upper.append(low["magnitude_upper"])
        high_lower.append(high["magnitude_lower"])
        high_upper.append(high["magnitude_upper"])
        value = decision.get("higher_to_dominant_low_upper")
        ratio_upper.append(np.nan if value is None else value)
        resolved.append(decision.get("denominator_status") == "simultaneously_resolved")
    return {
        "leading_lower": np.asarray(low_lower),
        "leading_upper": np.asarray(low_upper),
        "higher_lower": np.asarray(high_lower),
        "higher_upper": np.asarray(high_upper),
        "family_ratio_upper": np.asarray(ratio_upper),
        "denominator_resolved": np.asarray(resolved, dtype=bool),
    }


def _contribution_panel(axis, fits, intervals, title: str) -> None:
    _errorbar_interval(
        axis,
        fits["upper_strengths"],
        fits["leading"],
        intervals["leading_lower"],
        intervals["leading_upper"],
        color=COLORS["first"],
        label="leading contribution",
    )
    _errorbar_interval(
        axis,
        fits["upper_strengths"],
        fits["higher"],
        intervals["higher_lower"],
        intervals["higher_upper"],
        color=COLORS["higher"],
        label="first higher-order contribution",
        marker="s",
    )
    axis.set_title(title)
    axis.set_xlabel("nested fit upper strength h")
    axis.set_ylabel("contribution magnitude (state units)")
    axis.legend()


def _ratio_panel(axis, fits, intervals, title: str) -> None:
    x = fits["upper_strengths"]
    axis.plot(x, fits["point_ratio"], marker="o", color=COLORS["higher"], label="point ratio")
    finite = np.isfinite(intervals["family_ratio_upper"])
    if finite.any():
        axis.plot(
            x[finite],
            intervals["family_ratio_upper"][finite],
            marker="^",
            color="#7A3E9D",
            label="simultaneous family upper bound",
        )
    unresolved = ~intervals["denominator_resolved"]
    if unresolved.any():
        marker_height = max(2.0, float(np.nanmax(fits["point_ratio"])) * 1.8)
        axis.scatter(x[unresolved], np.full(unresolved.sum(), marker_height), marker="^", color="#7A3E9D")
        for value in x[unresolved]:
            axis.annotate("unbounded", (value, marker_height), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)
    axis.axhline(0.2, color="0.3", linestyle="--", label="20% provisional design criterion")
    axis.set_yscale("log")
    axis.set_title(title)
    axis.set_xlabel("nested fit upper strength h")
    axis.set_ylabel("higher / leading magnitude")
    axis.legend()


def plot_higher_order_pollution(
    output_dir: Path,
    strength_artifact: FigureArtifact,
    heldout_artifact: FigureArtifact,
    selection: dict[str, Any],
    common: dict[str, Any],
) -> dict[str, Any]:
    first_strengths, first_values = _strength_raw_odd(
        strength_artifact, selection["fundamental_observable"]
    )
    second_strengths, second_all = _heldout_a2(heldout_artifact, selection["omega"])
    second_values = second_all[:, :, STATE_INDEX[selection["second_harmonic_observable"]], 2]
    first_fit = _nested_contributions(first_values, first_strengths, (1, 3))
    second_fit = _nested_contributions(second_values, second_strengths, (2, 4))
    first_decisions = strength_artifact.derived["confirmed_block_bootstrap"]["adequacy"][
        "odd_fundamental"
    ]
    second_decisions = heldout_artifact.derived["second_harmonic_identification"][
        "by_frequency"
    ][_strength_key(selection["omega"])]["alternative_A2"]["adequacy"]
    first_intervals = _adequacy_intervals(
        first_decisions,
        first_fit["upper_strengths"],
        selection["fundamental_observable"],
        False,
    )
    second_intervals = _adequacy_intervals(
        second_decisions,
        second_fit["upper_strengths"],
        selection["second_harmonic_observable"],
        True,
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.3))
    _contribution_panel(axes[0, 0], first_fit, first_intervals, r"Fundamental: $c_1h$ versus $c_3h^3$")
    _contribution_panel(axes[0, 1], second_fit, second_intervals, r"Second harmonic A2: $c_2h^2$ versus $c_4h^4$")
    _ratio_panel(axes[1, 0], first_fit, first_intervals, "Fundamental higher/leading")
    _ratio_panel(axes[1, 1], second_fit, second_intervals, "Second-harmonic higher/leading")
    figure.suptitle(
        "Nested block-level strength fits at omega=2, x-direction forcing\n"
        "simultaneous bounds retain the original full decision families"
    )
    figure.tight_layout()
    return _save_figure(
        figure,
        output_dir,
        "03_higher_order_pollution",
        {
            **common,
            "question": "Are first omitted fitted powers small relative to leading response contributions?",
            "quantity": "nested c1*h+c3*h^3 fundamental fits and c2*h^2+c4*h^4 current-A2 fits",
            "uncertainty": "stored full-family simultaneous 95% magnitude bounds; unbounded means the leading denominator is not simultaneously resolved",
            "sources": [common["artifact_sources"]["strength"], common["artifact_sources"]["heldout"]],
            "plotted_data": {
                "fundamental": {**first_fit, **first_intervals},
                "second_harmonic_A2": {**second_fit, **second_intervals},
                "provisional_higher_order_fraction": 0.2,
            },
        },
    )


def _sampling_se_rms(drift: dict[str, Any]) -> float:
    values = np.asarray(drift["reference_standard_error_realified"], dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def _numerical_series(section: dict[str, Any], target: str, harmonic: str):
    keys = sorted(section, key=lambda value: float(value) if value.replace(".", "", 1).isdigit() else value)
    ratio, paired = [], []
    parity = "odd" if target == "first" else "even"
    for key in keys:
        drift = section[key][parity][harmonic]["drift"]
        reference = _sampling_se_rms(drift)
        ratio.append(drift["mean_delta_rms"] / reference)
        paired.append(drift["paired_standard_error_rms"] / reference)
    return keys, np.asarray(ratio), np.asarray(paired)


def plot_numerical_convergence(
    output_dir: Path,
    artifact: FigureArtifact,
    common: dict[str, Any],
) -> dict[str, Any]:
    derived = artifact.derived
    sections = (
        ("Observation length", derived["observation_window_comparisons_to_longest"], "cycles"),
        ("Phase resolution", derived["phase_refinement_comparisons_to_finest"], "phase points/cycle"),
        ("Solver profile", derived["solver_profile_comparisons_to_reference"], "profile"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 4.3))
    plotted = {}
    for axis, (title, section, xlabel) in zip(axes, sections):
        key_first, first, first_paired = _numerical_series(section, "first", "1")
        key_second, second, second_paired = _numerical_series(section, "second", "2")
        if key_first != key_second:
            raise ValueError("numerical target grids differ")
        positions = np.arange(len(key_first), dtype=float)
        axis.errorbar(positions - 0.08, first, yerr=first_paired, marker="o", capsize=3, color=COLORS["first"], label="odd n=1")
        axis.errorbar(positions + 0.08, second, yerr=second_paired, marker="s", capsize=3, color=COLORS["second"], label="even n=2")
        axis.axhline(1.0, color="0.35", linestyle="--", label="reference sampling SE")
        axis.set_xticks(positions, key_first)
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        plotted[title.lower().replace(" ", "_")] = {
            "settings": key_first,
            "fundamental_shift_over_sampling_se": first,
            "fundamental_paired_se_over_sampling_se": first_paired,
            "second_shift_over_sampling_se": second,
            "second_paired_se_over_sampling_se": second_paired,
        }
    axes[0].set_ylabel("mean numerical shift / reference sampling SE (RMS)")
    axes[0].legend(fontsize=7)
    figure.suptitle(
        "Numerical sensitivity relative to sampling uncertainty\n"
        "omega=2, h=0.5, x-direction forcing; bars show paired sensitivity-study SE"
    )
    figure.tight_layout()
    return _save_figure(
        figure,
        output_dir,
        "04_numerical_convergence",
        {
            **common,
            "question": "Are observation, phase-grid, and solver shifts small relative to current sampling uncertainty?",
            "quantity": "RMS block-mean contrast shift divided by RMS reference standard error",
            "uncertainty": "error bars are paired sensitivity-study SE on the same scale; they are not residual-bias confidence bounds",
            "sources": [common["artifact_sources"]["numerical"]],
            "plotted_data": plotted,
        },
    )


def plot_frequency_dependence(
    output_dir: Path,
    frequency_artifact: FigureArtifact,
    heldout_artifact: FigureArtifact,
    selection: dict[str, Any],
    common: dict[str, Any],
) -> dict[str, Any]:
    strength = float(selection["frequency_strength"])
    first_observable = selection["fundamental_observable"]
    second_observable = selection["second_harmonic_observable"]
    first_index = STATE_INDEX[first_observable]
    frequencies = np.asarray(frequency_artifact.derived["frequencies"], dtype=float)
    first_mean, first_se, first_identified = [], [], []
    for omega in frequencies:
        key = _strength_key(omega)
        report = frequency_artifact.derived["target_reports"][key][
            "odd_fundamental"
        ][_strength_key(strength)][first_index]["normalized_response"]
        first_mean.append(report["mean_magnitude"])
        first_se.append(report["radial_standard_error"])
        decision = _component(
            frequency_artifact.derived["joint_bootstrap"]["by_frequency"][key][
                "identification"
            ]["odd_fundamental"][_strength_key(strength)]["components"],
            first_observable,
        )
        first_identified.append(decision["simultaneous_interval_excludes_origin"])

    heldout_strengths, _ = _heldout_a2(heldout_artifact, frequencies[0])
    strength_position = int(np.flatnonzero(np.isclose(heldout_strengths, strength))[0])
    second_mean, second_se, second_identified = [], [], []
    for omega in frequencies:
        _, all_a2 = _heldout_a2(heldout_artifact, omega)
        values = -4 * all_a2[:, strength_position, STATE_INDEX[second_observable], 2] / strength**2
        summary = complex_block_summary(values)
        second_mean.append(summary["mean_magnitude"])
        second_se.append(summary["radial_standard_error"])
        decision = _component(
            heldout_artifact.derived["second_harmonic_identification"]["by_frequency"]
            [_strength_key(omega)]["alternative_A2"]["identification"]
            [_strength_key(strength)]["components"],
            second_observable,
        )
        second_identified.append(decision["simultaneous_interval_excludes_origin"])

    first_mean = np.asarray(first_mean)
    first_se = np.asarray(first_se)
    second_mean = np.asarray(second_mean)
    second_se = np.asarray(second_se)
    first_identified = np.asarray(first_identified, dtype=bool)
    second_identified = np.asarray(second_identified, dtype=bool)
    figure, axes = plt.subplots(1, 2, figsize=(10.1, 4.4), sharex=True)
    axes[0].errorbar(
        frequencies,
        first_mean,
        yerr=np.vstack((np.minimum(first_mean, first_se), first_se)),
        color=COLORS["first"],
        marker="o",
        capsize=3,
        label="radial SE (lower clipped at zero)",
    )
    axes[1].errorbar(
        frequencies,
        second_mean,
        yerr=np.vstack((np.minimum(second_mean, second_se), second_se)),
        color=COLORS["second"],
        marker="s",
        capsize=3,
        label="radial SE (lower clipped at zero)",
    )
    _mark_identified(axes[0], frequencies, first_mean, first_identified, COLORS["first"])
    _mark_identified(axes[1], frequencies, second_mean, second_identified, COLORS["second"])
    axes[0].set_ylabel(r"$|\chi^{(1)}_{xx}(\omega)|$ (state / forcing unit)")
    axes[1].set_ylabel(r"$|\chi^{(2)}_{zxx}(2\omega;\omega,\omega)|$ (state / forcing unit$^2$)")
    axes[0].set_title("Fundamental, reconnaissance B=32")
    axes[1].set_title("Second harmonic, held-out A2 B=32")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(frequencies, [str(value).rstrip("0").rstrip(".") for value in frequencies])
        axis.set_xlabel(r"forcing frequency $\omega$ (inverse Lorenz time)")
        axis.legend()
    figure.suptitle(
        f"Frequency dependence at h={strength:g}, x-direction forcing\n"
        "error bars: radial block SE; filled marker: stored simultaneous 95% CI excludes zero"
    )
    figure.tight_layout()
    return _save_figure(
        figure,
        output_dir,
        "05_frequency_dependence",
        {
            **common,
            "question": "Is omega=2 an isolated weak frequency for first- or second-order response?",
            "quantity": "chi1=2i*A1/h and current chi2=-4*A2/h^2, Fourier exp(-i*n*theta), no second-order factorial",
            "uncertainty": "error bars are radial block standard errors with the magnitude-axis lower arm clipped at zero; filled markers reproduce each artifact full-family simultaneous 95% exclusion decision",
            "sources": [common["artifact_sources"]["frequency"], common["artifact_sources"]["heldout"]],
            "plotted_data": {
                "frequencies": frequencies,
                "strength": strength,
                "chi1_mean_magnitude": first_mean,
                "chi1_radial_se": first_se,
                "chi1_simultaneously_identified": first_identified,
                "chi2_mean_magnitude": second_mean,
                "chi2_radial_se": second_se,
                "chi2_simultaneously_identified": second_identified,
            },
        },
    )


def generate_exploratory_figures(
    config_path: str | os.PathLike[str],
) -> tuple[Path, dict[str, Any]]:
    """Generate all supported views without running integrations or bootstraps."""
    _style()
    config_path = Path(config_path).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    config = _json(config_path)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported figure config schema")
    config_identifier = f"sha256:{file_sha256(config_path)}"
    output_dir = repo_root / config["output_root"] / config_identifier[7:19]
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        name: load_figure_artifact(repo_root, name, specification)
        for name, specification in config["artifacts"].items()
    }
    common = {
        "figure_set_id": config["figure_set_id"],
        "config_identifier": config_identifier,
        "generator": {
            "code_identifiers": active_source_identifiers(repo_root, config["runner_path"]),
            "git": git_provenance(repo_root),
            "environment": {
                **environment_provenance(),
                "matplotlib": matplotlib.__version__,
                "backend": matplotlib.get_backend(),
            },
        },
        "artifact_sources": {
            name: _artifact_source(artifact, repo_root)
            for name, artifact in artifacts.items()
        },
    }
    selection = config["selection"]
    figures = [
        plot_strength_scaling(output_dir, artifacts["strength"], artifacts["heldout"], selection, common),
        plot_identifiability_snr(output_dir, artifacts["strength"], artifacts["heldout"], selection, common),
        plot_higher_order_pollution(output_dir, artifacts["strength"], artifacts["heldout"], selection, common),
        plot_numerical_convergence(output_dir, artifacts["numerical"], common),
        plot_frequency_dependence(output_dir, artifacts["frequency"], artifacts["heldout"], selection, common),
    ]
    unsupported = {
        "figure_id": "raw_signal_and_noise",
        "status": "unsupported_by_retained_artifacts",
        "reason": (
            "Existing artifacts retain per-cycle Fourier coefficients for n=0..5, "
            "not raw time trajectories or complete phase-grid samples. A truncated "
            "inverse Fourier curve would understate unresolved chaotic energy."
        ),
        "required_data": [
            "sample_times with cycle and phase axes",
            "matched unforced and representative forced state samples",
            "block ID, protocol, discard, phase grid, and solver provenance",
        ],
        "sources_checked": list(common["artifact_sources"].values()),
    }
    write_json_atomic(output_dir / "unsupported_raw_signal.source.json", unsupported)
    manifest = {
        "schema_version": 1,
        "figure_set_id": config["figure_set_id"],
        "config_identifier": config_identifier,
        "figures": figures,
        "unsupported": [unsupported["figure_id"]],
        "unsupported_record": "unsupported_raw_signal.source.json",
        "artifact_sources": common["artifact_sources"],
        "generator": common["generator"],
    }
    write_json_atomic(output_dir / "figure_manifest.json", manifest)
    return output_dir, manifest
