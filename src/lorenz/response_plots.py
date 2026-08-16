"""Direct visualizations of the response objects themselves.

Each function consumes plain arrays with documented axes and draws exactly
one statistical object per visual role:

- a line or marker is a measured/estimated value,
- a band or error bar is uncertainty of that estimate,
- a background cloud/envelope is the empirical block-level distribution,
- a reference line is labeled as a reference, never as data.

The functions accept the tensor-shaped production data (block, condition,
state, ...) and leave the selection of which measured entries to render to
the caller, so the full first-order production view extends to the complete
output x forcing index grid without changing the plotting semantics.  No
function drops observations to fit a fixed panel count: layouts are computed
from the number of real series.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e7e6e1"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#4a3aa7", "#e34948"]
CLOUD = "#9a9a96"        # neutral gray: individual block populations
CLOUD_BAND = "#d6d4cf"   # light gray: block distribution band
COHERENT = "#0d366b"     # dark blue: coherent ensemble response (dominant)
COHERENT_BAND = "#86b6ef"  # light blue: uncertainty of the coherent mean
BACKGROUND = "#c47a52"   # muted warm: chaotic/background spectrum
REFERENCE = "#898781"    # neutral dotted references


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "axes.labelcolor": INK2,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.4,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "errorbar.capsize": 2,
        }
    )


def save_figure(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = directory / f"{name}.{extension}"
        try:
            fig.savefig(path, bbox_inches="tight", dpi=160)
        except PermissionError:
            # A viewer (common on Windows) can lock the previous PDF; the
            # PNG is still updated and the failure is reported, never hidden.
            print(
                f"warning: could not overwrite locked {path}; "
                f"{name}.{extension} was not updated"
            )
    plt.close(fig)
    print("wrote", directory / f"{name}.png")


def _footer(fig, text):
    fig.text(
        0.5, 0.005, text, ha="center", va="bottom", fontsize=7.2, color=MUTED,
        wrap=True,
    )


def mean_and_standard_error(values, axis: int = 0):
    """Block mean and per-part standard error of the mean.

    Returns ``(mean, standard_error_real, standard_error_imaginary)``.  For
    real-valued input the imaginary error is zero.
    """
    values = np.asarray(values)
    if values.shape[axis] < 2:
        raise ValueError("at least two blocks are required for an SE")
    mean = values.mean(axis=axis)
    divisor = np.sqrt(values.shape[axis])
    if np.iscomplexobj(values):
        se_real = values.real.std(axis=axis, ddof=1) / divisor
        se_imag = values.imag.std(axis=axis, ddof=1) / divisor
    else:
        se_real = values.std(axis=axis, ddof=1) / divisor
        se_imag = np.zeros_like(se_real)
    return mean, se_real, se_imag


def statistical_snr(values):
    """Statistical SNR of a complex block-level response coefficient.

    For block estimates ``r_b = [Re(R_b), Im(R_b)]`` with across-block
    covariance of the mean ``Cov(mean(r)) = sample_covariance / B``,

        SNR_stat = sqrt( mean(r)^T Cov(mean(r))^{-1} mean(r) ).

    Returns ``(snr, used_pseudoinverse)``; the pseudoinverse is used only
    when the covariance of the mean is not invertible, and that case is
    reported by the second return value.  This is a compact numerical
    annotation, not a hypothesis-testing system.
    """
    values = np.asarray(values)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("at least two blocks are required for an SNR")
    if not np.iscomplexobj(values):
        raise ValueError("values must be complex response coefficients")
    realified = np.column_stack((values.real, values.imag))
    mean = realified.mean(axis=0)
    covariance_of_mean = np.cov(realified, rowvar=False, ddof=1) / len(values)
    used_pseudoinverse = False
    try:
        score = float(mean @ np.linalg.solve(covariance_of_mean, mean))
    except np.linalg.LinAlgError:
        used_pseudoinverse = True
        score = float(mean @ np.linalg.pinv(covariance_of_mean) @ mean)
    return float(np.sqrt(max(score, 0.0))), used_pseudoinverse


# ---------------------------------------------------------------------------
# figure 1 - individual block spectra and the coherent ensemble mean
# ---------------------------------------------------------------------------

def figure_block_spectra(
    spectra,
    frequency_grid,
    forcing_frequency,
    *,
    harmonics=None,
    quantile=(0.05, 0.95),
    resamples=2000,
    seed=2026081601,
    coherent_mean=True,
    min_blocks_for_uncertainty=8,
    background=None,
    background_band=None,
    background_label=None,
    harmonic_annotations=None,
    title="Individual block spectra and coherent ensemble mean",
    caption=None,
    x_range=None,
    y_scale="log",
    amplitude_label="|S_b(Omega)| (state)",
):
    """Fig 1: per-block amplitude spectra, block distribution, coherent mean.

    ``spectra`` has axes ``(block, frequency_bin)`` and holds the complex
    block-level coefficients ``S_b(Omega)`` on the explicit
    ``frequency_grid``.  The visual hierarchy separates distinct statistical
    objects:

    - neutral gray, very thin, low-opacity curves: ``|S_b(Omega)|`` for
      every block,
    - a light gray band: the central-quantile band of the empirical
      block-to-block spectral distribution computed from all blocks,
    - the dominant dark blue curve: ``|mean_b S_b(Omega)|`` (complex
      ensemble averaging before taking the magnitude),
    - a light blue band: resampling uncertainty of that coherent mean
      (bootstrap over whole blocks; omitted below
      ``min_blocks_for_uncertainty`` blocks),
    - a muted warm curve (``background``, with optional ``background_band``):
      the chaotic/background spectrum, e.g. the unforced Welch RMS per bin,
      drawn thinner than the coherent response,
    - thin neutral dotted references at the forcing frequency and its
      harmonics, with compact ``harmonic_annotations`` (a mapping from
      harmonic to a short string, e.g. a response/background ratio) placed
      next to each reference line.

    Individual spectra are never smoothed.  With ``coherent_mean=False``
    only the per-block cloud and its quantile band are drawn; this mode
    exists to render a reconstruction-limited view and must be labeled as
    such by the caption.
    """
    spectra = np.asarray(spectra)
    frequency_grid = np.asarray(frequency_grid, dtype=float)
    if spectra.ndim != 2 or spectra.shape[1] != len(frequency_grid):
        raise ValueError("spectra must have axes (block, frequency_bin)")
    if coherent_mean and not np.iscomplexobj(spectra):
        raise ValueError("spectra must be complex S_b(Omega) coefficients")
    if harmonics is None:
        harmonics = (1, 2, 3)
    if background is not None:
        background_grid, background_amplitude = background
        background_grid = np.asarray(background_grid, dtype=float)
        background_amplitude = np.asarray(background_amplitude, dtype=float)
        if len(background_grid) != len(background_amplitude):
            raise ValueError("background grid and amplitude must match")
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    amplitude = np.abs(spectra)
    low, high = np.quantile(amplitude, quantile, axis=0)
    ax.fill_between(
        frequency_grid, low, high, color=CLOUD_BAND, alpha=0.5, linewidth=0,
        label=(
            f"central {100 * (quantile[1] - quantile[0]):.0f}% quantile band of "
            f"|S_b| across {len(spectra)} blocks"
        ),
    )
    for block_amplitude in amplitude:
        ax.plot(
            frequency_grid, block_amplitude, color=CLOUD, alpha=0.25,
            linewidth=0.4, rasterized=True,
        )
    ax.plot([], [], color=CLOUD, alpha=0.6, linewidth=0.4,
            label=f"individual blocks ({len(spectra)}), |S_b|")
    if background is not None:
        if background_band is not None:
            background_low, background_high = background_band
            ax.fill_between(
                background_grid, background_low, background_high,
                color=BACKGROUND, alpha=0.18, linewidth=0, zorder=1,
            )
        ax.plot(
            background_grid, background_amplitude, color=BACKGROUND,
            linewidth=1.1, zorder=2,
            label=background_label or "chaotic background spectrum",
        )
    if coherent_mean:
        coherent = spectra.mean(axis=0)
        ax.plot(
            frequency_grid, np.abs(coherent), color=COHERENT, linewidth=1.8,
            zorder=4, label="coherent ensemble mean |mean_b S_b(Omega)|",
        )
        if len(spectra) >= min_blocks_for_uncertainty:
            rng = np.random.default_rng(seed)
            resampled = np.empty((resamples, len(frequency_grid)))
            for index in range(resamples):
                draw = rng.integers(0, len(spectra), len(spectra))
                resampled[index] = np.abs(spectra[draw].mean(axis=0))
            lower, upper = np.quantile(resampled, (0.025, 0.975), axis=0)
            ax.fill_between(
                frequency_grid, lower, upper, color=COHERENT_BAND, alpha=0.4,
                linewidth=0, zorder=3,
                label="uncertainty of the coherent mean (block bootstrap, 95%)",
            )
        else:
            ax.plot(
                [], [], color=COHERENT_BAND, linewidth=2.5,
                label=(
                    "uncertainty of the coherent mean omitted "
                    f"(only {len(spectra)} blocks)"
                ),
            )
    if harmonic_annotations is None:
        harmonic_annotations = {}
    for harmonic in harmonics:
        reference = float(forcing_frequency) * harmonic
        if x_range is not None and not (x_range[0] <= reference <= x_range[1]):
            continue
        ax.axvline(
            reference, color=REFERENCE, linestyle=":", linewidth=0.9, zorder=1,
            label=(
                f"reference: {harmonic} x forcing frequency"
                if harmonic == harmonics[0]
                else None
            ),
        )
        annotation = harmonic_annotations.get(harmonic)
        if annotation:
            ax.text(
                reference, 0.90, f"  n={harmonic}: {annotation}",
                transform=ax.get_xaxis_transform(), fontsize=7, color=INK2,
                va="top", ha="left",
            )
    if x_range is not None:
        ax.set_xlim(x_range)
    if y_scale == "log":
        ax.set_yscale("log")
    ax.set_xlabel("physical angular frequency Omega (rad/time)")
    ax.set_ylabel(amplitude_label)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper right", fontsize=7.5)
    if caption:
        _footer(fig, caption)
    return fig


# ---------------------------------------------------------------------------
# figure 2 - raw time-domain trajectories
# ---------------------------------------------------------------------------

def figure_raw_trajectories(
    values,
    times,
    *,
    trajectory_labels=None,
    trajectory_colors=None,
    max_traces=None,
    title="Raw physical-time trajectories",
    caption=None,
    value_label="state",
    envelope_quantile=(0.05, 0.95),
    panel_labels=None,
):
    """Fig 2: raw chaotic trajectories before any ensemble averaging.

    ``values`` has axes ``(trajectory, time)`` or ``(panel, trajectory,
    time)``.  The central-quantile envelope is computed from ALL available
    trajectories per panel; when ``max_traces`` is set, only a deterministic
    subset of individual traces is drawn (selection: evenly spaced indices
    over the trajectory axis), which is reported in the legend.  Optional
    ``trajectory_colors`` mark condition identity per trace.  No phase or
    Fourier reconstruction is involved.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if values.ndim == 2:
        values = values[None, :, :]
    if values.shape[-1] != len(times):
        raise ValueError("values must have a trailing time axis matching times")
    n_panel, n_trajectory = values.shape[:2]
    if trajectory_labels is None:
        trajectory_labels = [
            f"trajectory {index}" for index in range(n_trajectory)
        ]
    if len(trajectory_labels) != n_trajectory:
        raise ValueError("trajectory_labels must match the trajectory axis")
    if trajectory_colors is not None and len(trajectory_colors) != n_trajectory:
        raise ValueError("trajectory_colors must match the trajectory axis")
    if panel_labels is None:
        panel_labels = [f"panel {index}" for index in range(n_panel)]
    if len(panel_labels) != n_panel:
        raise ValueError("panel_labels must match the panel axis")
    drawn = np.arange(n_trajectory)
    if max_traces is not None and n_trajectory > max_traces:
        drawn = np.linspace(0, n_trajectory - 1, int(max_traces)).astype(int)
    fig, axes = plt.subplots(
        n_panel, 1, figsize=(10.5, 3.0 * n_panel), squeeze=False,
    )
    for panel_index in range(n_panel):
        ax = axes[panel_index, 0]
        panel_values = values[panel_index]
        low, high = np.quantile(panel_values, envelope_quantile, axis=0)
        ax.fill_between(
            times, low, high, color=CLOUD, alpha=0.18, linewidth=0,
            label=(
                f"central {100 * (envelope_quantile[1] - envelope_quantile[0]):.0f}% "
                f"envelope across all {n_trajectory} trajectories"
            ),
        )
        for trace_index in drawn:
            color = (
                trajectory_colors[trace_index]
                if trajectory_colors is not None
                else CLOUD
            )
            ax.plot(
                times, panel_values[trace_index], color=color, alpha=0.55,
                linewidth=0.6, rasterized=True,
            )
        ax.plot(
            [], [], color=MUTED, alpha=0.8, linewidth=0.6,
            label=(
                f"{len(drawn)} deterministically selected traces "
                f"(indices {drawn[0]}..{drawn[-1]})"
                if len(drawn) < n_trajectory
                else f"all {n_trajectory} trajectories"
            ),
        )
        ax.set_xlabel("physical time (Lorenz time units)")
        ax.set_ylabel(value_label)
        ax.set_title(f"{panel_labels[panel_index]}", loc="left")
        if panel_index == 0:
            ax.legend(loc="upper right", fontsize=7.5)
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, color=INK)
    fig.subplots_adjust(top=0.9, hspace=0.45)
    if caption:
        _footer(fig, caption)
    return fig


# ---------------------------------------------------------------------------
# figure 3 - frequency response curves
# ---------------------------------------------------------------------------

def figure_frequency_response(
    frequencies,
    values,
    uncertainty,
    *,
    entry_labels=None,
    panel_labels=None,
    title="Frequency response",
    caption=None,
    log_x=True,
    value_label="response (state per forcing unit)",
):
    """Fig 3: point estimate and uncertainty versus ordered omega.

    ``values`` has axes ``(frequency, entry)`` and holds the complex measured
    response (e.g. first-order susceptibility entries).  ``uncertainty`` is
    a ``(real, imaginary)`` pair of arrays, each with the same shape as
    ``values``, holding the per-part standard errors; the real panel uses
    only the real error and the imaginary panel only the imaginary error
    (directional uncertainty is never collapsed).  One row per entry, one
    column per real/imaginary part; curves are connected because
    ``frequencies`` is an ordered scan of one physical quantity.  No
    detection/validity classes are encoded.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    values = np.asarray(values)
    uncertainty_real, uncertainty_imag = uncertainty
    uncertainty_real = np.asarray(uncertainty_real, dtype=float)
    uncertainty_imag = np.asarray(uncertainty_imag, dtype=float)
    if values.shape[0] != len(frequencies):
        raise ValueError("values must have axes (frequency, entry)")
    for part_error in (uncertainty_real, uncertainty_imag):
        if part_error.shape != values.shape:
            raise ValueError(
                "each uncertainty part must match the values shape"
            )
    n_entry = values.shape[1]
    if entry_labels is None:
        entry_labels = [f"entry {index}" for index in range(n_entry)]
    if len(entry_labels) != n_entry:
        raise ValueError("entry_labels must match the entry axis")
    fig, axes = plt.subplots(
        n_entry, 2, figsize=(10.5, 1.2 + 2.5 * n_entry), squeeze=False,
    )
    for entry_index in range(n_entry):
        for part_index, part in enumerate(("real", "imaginary")):
            ax = axes[entry_index, part_index]
            estimate = values[:, entry_index].real if part == "real" else values[:, entry_index].imag
            part_error = (
                uncertainty_real if part == "real" else uncertainty_imag
            )[:, entry_index]
            ax.plot(frequencies, estimate, color=SERIES[entry_index % len(SERIES)],
                    marker="o", markersize=3, linewidth=1.1, zorder=3)
            ax.fill_between(
                frequencies,
                estimate - part_error,
                estimate + part_error,
                color=SERIES[entry_index % len(SERIES)], alpha=0.13, linewidth=0,
                zorder=2, label=f"standard error of the block mean ({part} part)",
            )
            if log_x:
                ax.set_xscale("log")
            ax.set_xlabel("omega (log)" if log_x else "omega")
            ax.set_ylabel(f"{part} part {value_label}")
            label = entry_labels[entry_index]
            if part == "real":
                ax.set_title(f"{label} - real", loc="left")
            else:
                ax.set_title(f"{label} - imaginary", loc="left")
            if entry_index == 0 and part_index == 0:
                ax.legend(loc="upper right", fontsize=7.5)
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, color=INK)
    fig.subplots_adjust(top=0.92)
    if caption:
        _footer(fig, caption)
    return fig


# ---------------------------------------------------------------------------
# figure 4 - strength dependence
# ---------------------------------------------------------------------------

def figure_strength_dependence(
    frequency_strengths,
    first_order,
    first_order_uncertainty,
    second_order,
    second_order_uncertainty,
    *,
    frequency_labels=None,
    title="Strength dependence of normalized contrasts",
    caption=None,
    first_label="A1(h) / h",
    second_label="A2(h) / h^2",
):
    """Fig 4: normalized raw contrasts directly against forcing strength.

    ``frequency_strengths`` is a sequence of per-frequency strength arrays
    (the artifact's own strength grids; frequencies may differ).  For each
    frequency, ``first_order`` (``A1(h)/h``) and ``second_order``
    (``A2(h)/h^2``) are complex arrays matching that frequency's strength
    grid; the uncertainties are ``(real, imaginary)`` pairs of arrays with
    the same shapes, and the real/imaginary panels use only their own part
    (directional uncertainty is never collapsed).  One column per real
    frequency series (layout computed from the number of series), one row
    per quantity; every available strength is plotted.  No h/h^2 guide
    lines are drawn.
    """
    frequency_strengths = list(frequency_strengths)
    n_frequency = len(frequency_strengths)
    if frequency_labels is None:
        frequency_labels = [f"omega {index}" for index in range(n_frequency)]
    if len(frequency_labels) != n_frequency:
        raise ValueError("frequency_labels must match the frequency axis")
    checked = []
    for strengths in frequency_strengths:
        strengths = np.asarray(strengths, dtype=float)
        if strengths.ndim != 1:
            raise ValueError("each strength grid must be one-dimensional")
        checked.append(strengths)
    quantity_sets = []
    for values, uncertainty in (
        (first_order, first_order_uncertainty),
        (second_order, second_order_uncertainty),
    ):
        value_series = []
        error_series = []
        for column, strengths in enumerate(checked):
            entry = np.asarray(values[column])
            real, imag = uncertainty[column]
            real = np.asarray(real, dtype=float)
            imag = np.asarray(imag, dtype=float)
            if entry.shape != (len(strengths),):
                raise ValueError(
                    "each strength series must match its frequency's "
                    "strength grid"
                )
            if real.shape != entry.shape or imag.shape != entry.shape:
                raise ValueError(
                    "each uncertainty part must match its series shape"
                )
            value_series.append(entry)
            error_series.append((real, imag))
        quantity_sets.append((value_series, error_series))
    fig, axes = plt.subplots(
        2, n_frequency, figsize=(2.4 + 3.2 * n_frequency, 7.6), squeeze=False,
    )
    for quantity_row, (label, color, (value_series, error_series)) in enumerate(
        (
            (first_label, SERIES[0], quantity_sets[0]),
            (second_label, SERIES[1], quantity_sets[1]),
        )
    ):
        for column, strengths in enumerate(checked):
            ax = axes[quantity_row, column]
            for part_index, part in enumerate(("real", "imaginary")):
                estimate = (
                    value_series[column].real
                    if part == "real"
                    else value_series[column].imag
                )
                part_error = error_series[column][part_index]
                line_style = "-" if part == "real" else "--"
                ax.errorbar(
                    strengths, estimate,
                    yerr=part_error,
                    color=color,
                    linestyle=line_style,
                    marker="o",
                    markersize=3,
                    linewidth=1.2,
                    capsize=2,
                    label=part,
                )
            ax.set_xlabel("forcing strength h")
            ax.set_ylabel(f"{label} (state units)")
            ax.set_title(f"omega={frequency_labels[column]}", loc="left")
            if quantity_row == 0 and column == 0:
                ax.legend(loc="upper right", fontsize=7.5)
    # Comparable panels share one y scale: each row uses the union of the
    # per-panel data limits so independent autoscaling cannot hide magnitude
    # differences across frequencies.
    for quantity_row in range(2):
        limits = [
            axes[quantity_row, column].get_ylim() for column in range(n_frequency)
        ]
        low = min(limit[0] for limit in limits)
        high = max(limit[1] for limit in limits)
        for column in range(n_frequency):
            axes[quantity_row, column].set_ylim(low, high)
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, color=INK)
    fig.subplots_adjust(top=0.9, hspace=0.55)
    if caption:
        _footer(fig, caption)
    return fig


# ---------------------------------------------------------------------------
# figure 5 - harmonic content
# ---------------------------------------------------------------------------

def figure_harmonic_content(
    panels,
    *,
    block_values=None,
    block_values_panel=0,
    block_values_harmonic=None,
    title="Harmonic content",
    caption=None,
    value_label="harmonic coefficient (state units)",
):
    """Fig 5: complete retained inferential harmonic set.

    ``panels`` is a sequence of dictionaries, each holding one physically
    relevant response component::

        {"harmonics": array(n,), "point": complex array(n,),
         "uncertainty": (real array(n,), imag array(n,)), "label": str}

    Markers always sit at the estimated value of each part; the real and
    imaginary panels draw only their own per-part uncertainty as error bars
    (directional uncertainty is never collapsed).  When ``block_values`` is given (complex
    axes ``(block, harmonic)`` matching panel ``block_values_panel``), an
    accompanying panel shows the underlying block distribution of
    ``block_values_harmonic`` as Re(R_b) vs Im(R_b) for all blocks, together
    with the origin, the block mean, and the 95% covariance ellipse of the
    mean.  When no block-level values are available the panel is omitted
    rather than substituted.  The layout is computed from the number of
    panels.
    """
    panels = list(panels)
    if not panels:
        raise ValueError("at least one harmonic panel is required")
    checked = []
    for panel in panels:
        harmonics = np.asarray(panel["harmonics"], dtype=float)
        point = np.asarray(panel["point"])
        uncertainty_real, uncertainty_imag = panel["uncertainty"]
        uncertainty_real = np.asarray(uncertainty_real, dtype=float)
        uncertainty_imag = np.asarray(uncertainty_imag, dtype=float)
        if point.shape[0] != len(harmonics):
            raise ValueError("point must have axes (harmonic,)")
        for part_error in (uncertainty_real, uncertainty_imag):
            if part_error.shape != point.shape:
                raise ValueError(
                    "each uncertainty part must match the point shape"
                )
        checked.append(
            {
                "harmonics": harmonics,
                "point": point,
                "uncertainty": (uncertainty_real, uncertainty_imag),
                "label": panel.get("label", "component"),
            }
        )
    if block_values is not None:
        block_values = np.asarray(block_values)
        if block_values.ndim != 2 or not np.iscomplexobj(block_values):
            raise ValueError("block_values must be complex with axes (block, harmonic)")
    panel_count = len(checked) + (1 if block_values is not None else 0)
    n_columns = min(2, panel_count)
    n_rows = int(np.ceil(panel_count / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(6.4 * n_columns, 4.2 * n_rows),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for panel_index, panel in enumerate(checked):
        ax = flat_axes[panel_index]
        harmonics = panel["harmonics"]
        point = panel["point"]
        uncertainty_real, uncertainty_imag = panel["uncertainty"]
        for part_index, part in enumerate(("real", "imaginary")):
            estimate = point.real if part == "real" else point.imag
            part_error = uncertainty_real if part == "real" else uncertainty_imag
            offset = 0.06 * (1 if part == "real" else -1)
            ax.errorbar(
                harmonics + offset, estimate, yerr=part_error,
                color=SERIES[part_index], linestyle="", marker="o", markersize=3.5,
                capsize=2, label=part,
            )
        ax.set_xlabel("harmonic n")
        ax.set_ylabel(value_label)
        ax.set_xticks(harmonics, [str(int(value)) for value in harmonics])
        ax.set_title(f"{panel['label']} - estimates with uncertainty", loc="left")
        ax.legend(loc="upper right", fontsize=7.5)
    if block_values is not None:
        from matplotlib.patches import Ellipse

        ax = flat_axes[len(checked)]
        panel = checked[block_values_panel]
        if block_values_harmonic is None or block_values_harmonic not in panel["harmonics"]:
            raise ValueError(
                "block_values_harmonic must appear in the selected panel"
            )
        if block_values.shape != (block_values.shape[0], len(panel["harmonics"])):
            raise ValueError(
                "block_values must have axes (block, harmonic) matching the "
                "selected panel"
            )
        harmonic_index = list(panel["harmonics"]).index(block_values_harmonic)
        values = block_values[:, harmonic_index]
        mean = panel["point"][harmonic_index]
        ax.scatter(
            values.real, values.imag, color=CLOUD, s=8, alpha=0.6,
            rasterized=True, label=f"{len(values)} blocks",
        )
        ax.plot(
            [0.0], [0.0], marker="+", color=MUTED, markersize=8, linestyle="",
            label="origin",
        )
        ax.plot(
            [mean.real], [mean.imag], marker="o", color=COHERENT, markersize=5,
            linestyle="", label="block mean",
        )
        realified = np.column_stack((values.real, values.imag))
        covariance_of_mean = np.cov(realified, rowvar=False, ddof=1) / len(values)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_of_mean)
        if np.all(eigenvalues > 0):
            # 95% quantile of the chi^2_2 distribution.
            scale = np.sqrt(5.991)
            width, height = 2 * scale * np.sqrt(eigenvalues)
            angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
            ax.add_patch(
                Ellipse(
                    (mean.real, mean.imag), width, height, angle=angle,
                    fill=False, edgecolor=COHERENT, linewidth=1.0,
                    label="95% covariance ellipse of the block mean",
                )
            )
        ax.axhline(0, color=GRID, linewidth=0.5)
        ax.axvline(0, color=GRID, linewidth=0.5)
        ax.set_xlabel("Re(R_b)")
        ax.set_ylabel("Im(R_b)")
        ax.set_title(
            f"block distribution at n={block_values_harmonic:g}", loc="left"
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="upper right", fontsize=7.5)
    for extra in flat_axes[panel_count:]:
        extra.set_visible(False)
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, color=INK)
    fig.subplots_adjust(top=0.88, hspace=0.5)
    if caption:
        _footer(fig, caption)
    return fig
