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
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#4a3aa7", "#e34948"]
CLOUD = "#2a78d6"
COHERENT = "#0d366b"
COHERENT_BAND = "#86b6ef"
REFERENCE = "#898781"


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
            "grid.linewidth": 0.5,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "lines.markersize": 6,
            "errorbar.capsize": 2.5,
        }
    )


def save_figure(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            directory / f"{name}.{extension}",
            bbox_inches="tight",
            dpi=160,
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
    title="Individual block spectra and coherent ensemble mean",
    caption=None,
    x_range=None,
    y_scale="log",
    amplitude_label="|S_b(Omega)| (state)",
):
    """Fig 1: per-block amplitude spectra, block distribution, coherent mean.

    ``spectra`` has axes ``(block, frequency_bin)`` and holds the complex
    block-level coefficients ``S_b(Omega)`` on the explicit
    ``frequency_grid``.  Three distinct statistical objects are drawn:

    - thin low-opacity curves: ``|S_b(Omega)|`` for every block,
    - a central-quantile background band: the empirical block-to-block
      spectral distribution computed from all blocks,
    - the dominant dark curve: ``|mean_b S_b(Omega)|`` (complex ensemble
      averaging before taking the magnitude),
    - a separate lighter band: the resampling uncertainty of that coherent
      mean (bootstrap over whole blocks),
    - labeled vertical references at the forcing frequency and its harmonics
      that lie inside the displayed range.

    With ``coherent_mean=False`` only the per-block cloud and its quantile
    band are drawn; this mode exists to render a reconstruction-limited view
    when only per-block real amplitude spectra (e.g. Welch amplitudes) are
    retained, and must be labeled as such by the caption.
    """
    spectra = np.asarray(spectra)
    frequency_grid = np.asarray(frequency_grid, dtype=float)
    if spectra.ndim != 2 or spectra.shape[1] != len(frequency_grid):
        raise ValueError("spectra must have axes (block, frequency_bin)")
    if coherent_mean and not np.iscomplexobj(spectra):
        raise ValueError("spectra must be complex S_b(Omega) coefficients")
    if harmonics is None:
        harmonics = (1, 2, 3)
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    amplitude = np.abs(spectra)
    low, high = np.quantile(amplitude, quantile, axis=0)
    ax.fill_between(
        frequency_grid, low, high, color=CLOUD, alpha=0.18, linewidth=0,
        label=(
            f"central {100 * (quantile[1] - quantile[0]):.0f}% quantile band of "
            f"|S_b| across {len(spectra)} blocks"
        ),
    )
    for block_amplitude in amplitude:
        ax.plot(
            frequency_grid, block_amplitude, color=CLOUD, alpha=0.14,
            linewidth=0.6, rasterized=True,
        )
    ax.plot([], [], color=CLOUD, alpha=0.4, linewidth=0.6,
            label=f"individual blocks ({len(spectra)}), |S_b|")
    if coherent_mean:
        coherent = spectra.mean(axis=0)
        ax.plot(
            frequency_grid, np.abs(coherent), color=COHERENT, linewidth=2.2,
            zorder=4, label="coherent ensemble mean |mean_b S_b(Omega)|",
        )
        rng = np.random.default_rng(seed)
        resampled = np.empty((resamples, len(frequency_grid)))
        for index in range(resamples):
            draw = rng.integers(0, len(spectra), len(spectra))
            resampled[index] = np.abs(spectra[draw].mean(axis=0))
        lower, upper = np.quantile(resampled, (0.025, 0.975), axis=0)
        ax.fill_between(
            frequency_grid, lower, upper, color=COHERENT_BAND, alpha=0.45,
            linewidth=0, zorder=3,
            label="uncertainty of the coherent mean (block bootstrap, 95%)",
        )
    for harmonic in harmonics:
        reference = float(forcing_frequency) * harmonic
        if x_range is not None and not (x_range[0] <= reference <= x_range[1]):
            continue
        ax.axvline(
            reference, color=REFERENCE, linestyle=":", linewidth=1.0, zorder=1,
            label=(
                f"reference: {harmonic} x forcing frequency"
                if harmonic == harmonics[0]
                else None
            ),
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
    response (e.g. first-order susceptibility entries).  ``uncertainty`` has
    the same shape and holds the standard error of each real/imaginary part
    (one value per entry).  One row per entry, one column per real/imaginary
    part; curves are connected because ``frequencies`` is an ordered scan of
    one physical quantity.  No detection/validity classes are encoded.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    values = np.asarray(values)
    uncertainty = np.asarray(uncertainty, dtype=float)
    if values.shape[0] != len(frequencies) or values.shape != uncertainty.shape:
        raise ValueError("values and uncertainty must have axes (frequency, entry)")
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
            ax.plot(frequencies, estimate, color=SERIES[entry_index % len(SERIES)],
                    marker="o", markersize=4, linewidth=1.6, zorder=3)
            ax.fill_between(
                frequencies,
                estimate - uncertainty[:, entry_index],
                estimate + uncertainty[:, entry_index],
                color=SERIES[entry_index % len(SERIES)], alpha=0.2, linewidth=0,
                zorder=2, label="standard error of the block mean",
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
    strengths,
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

    ``first_order`` (``A1(h)/h``) and ``second_order`` (``A2(h)/h^2``) have
    axes ``(frequency, strength)`` and are complex; the uncertainties have
    the same shapes and hold the standard error of each real/imaginary part.
    One column per real frequency series (layout computed from the number of
    series), one row per quantity.  Real and imaginary parts are drawn as
    distinct curve styles; every available strength is plotted.  No h/h^2
    guide lines are drawn.
    """
    strengths = np.asarray(strengths, dtype=float)
    first_order = np.asarray(first_order)
    first_order_uncertainty = np.asarray(first_order_uncertainty, dtype=float)
    second_order = np.asarray(second_order)
    second_order_uncertainty = np.asarray(second_order_uncertainty, dtype=float)
    n_frequency = first_order.shape[0]
    if frequency_labels is None:
        frequency_labels = [f"omega {index}" for index in range(n_frequency)]
    if len(frequency_labels) != n_frequency:
        raise ValueError("frequency_labels must match the frequency axis")
    for values, se in (
        (first_order, first_order_uncertainty),
        (second_order, second_order_uncertainty),
    ):
        if values.shape != (n_frequency, len(strengths)) or values.shape != se.shape:
            raise ValueError(
                "strength arrays must have axes (frequency, strength)"
            )
    fig, axes = plt.subplots(
        2, n_frequency, figsize=(2.4 + 3.2 * n_frequency, 7.6), squeeze=False,
    )
    for quantity_row, (values, uncertainty, label, color) in enumerate(
        (
            (first_order, first_order_uncertainty, first_label, SERIES[0]),
            (second_order, second_order_uncertainty, second_label, SERIES[1]),
        )
    ):
        for column in range(n_frequency):
            ax = axes[quantity_row, column]
            for part_index, part in enumerate(("real", "imaginary")):
                estimate = values[column].real if part == "real" else values[column].imag
                line_style = "-" if part == "real" else "--"
                ax.errorbar(
                    strengths, estimate,
                    yerr=uncertainty[column],
                    color=color,
                    linestyle=line_style,
                    marker="o",
                    markersize=4,
                    linewidth=1.4,
                    capsize=2.5,
                    label=part,
                )
            ax.set_xlabel("forcing strength h")
            ax.set_ylabel(f"{label} (state units)")
            ax.set_title(f"omega={frequency_labels[column]}", loc="left")
            if quantity_row == 0 and column == 0:
                ax.legend(loc="upper right", fontsize=7.5)
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
    per_cycle=None,
    per_cycle_panel=0,
    per_cycle_harmonic=None,
    title="Harmonic content",
    caption=None,
    value_label="harmonic coefficient (state units)",
):
    """Fig 5: complete retained inferential harmonic set.

    ``panels`` is a sequence of dictionaries, each holding one physically
    relevant response component::

        {"harmonics": array(n,), "point": complex array(n,),
         "uncertainty": real array(n,), "label": str}

    Markers always sit at the estimated value of each part; uncertainty is
    drawn separately as error bars.  When ``per_cycle`` is given (axes
    ``(sample, harmonic)`` matching panel ``per_cycle_panel``), an
    accompanying panel shows the underlying cycle-level distribution for
    ``per_cycle_harmonic`` instead of only the final mean.  The layout is
    computed from the number of panels.
    """
    panels = list(panels)
    if not panels:
        raise ValueError("at least one harmonic panel is required")
    checked = []
    for panel in panels:
        harmonics = np.asarray(panel["harmonics"], dtype=float)
        point = np.asarray(panel["point"])
        uncertainty = np.asarray(panel["uncertainty"], dtype=float)
        if point.shape[0] != len(harmonics) or point.shape != uncertainty.shape:
            raise ValueError("point and uncertainty must have axes (harmonic,)")
        checked.append(
            {
                "harmonics": harmonics,
                "point": point,
                "uncertainty": uncertainty,
                "label": panel.get("label", "component"),
            }
        )
    if per_cycle is not None:
        per_cycle = np.asarray(per_cycle)
        if per_cycle.ndim != 2:
            raise ValueError("per_cycle must have axes (sample, harmonic)")
    panel_count = len(checked) + (1 if per_cycle is not None else 0)
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
        uncertainty = panel["uncertainty"]
        for part_index, part in enumerate(("real", "imaginary")):
            estimate = point.real if part == "real" else point.imag
            offset = 0.06 * (1 if part == "real" else -1)
            ax.errorbar(
                harmonics + offset, estimate, yerr=uncertainty,
                color=SERIES[part_index], linestyle="", marker="o", markersize=4,
                capsize=2.5, label=part,
            )
        ax.set_xlabel("harmonic n")
        ax.set_ylabel(value_label)
        ax.set_xticks(harmonics, [str(int(value)) for value in harmonics])
        ax.set_title(f"{panel['label']} - estimates with uncertainty", loc="left")
        ax.legend(loc="upper right", fontsize=7.5)
    if per_cycle is not None:
        ax = flat_axes[len(checked)]
        panel = checked[per_cycle_panel]
        if per_cycle_harmonic is None or per_cycle_harmonic not in panel["harmonics"]:
            raise ValueError(
                "per_cycle_harmonic must appear in the selected panel"
            )
        if per_cycle.shape != (per_cycle.shape[0], len(panel["harmonics"])):
            raise ValueError(
                "per_cycle must have axes (sample, harmonic) matching the "
                "selected panel"
            )
        harmonic_index = list(panel["harmonics"]).index(per_cycle_harmonic)
        cycle_values = per_cycle[:, harmonic_index]
        ax.scatter(
            np.arange(len(cycle_values)), cycle_values.real, color=SERIES[0],
            s=6, alpha=0.5, rasterized=True, label="real part",
        )
        ax.scatter(
            np.arange(len(cycle_values)), cycle_values.imag, color=SERIES[1],
            s=6, alpha=0.5, rasterized=True, label="imaginary part",
        )
        ax.axhline(
            panel["point"][harmonic_index].real, color=SERIES[0], linestyle="-",
            linewidth=1.4, label="block mean (real)",
        )
        ax.axhline(
            panel["point"][harmonic_index].imag, color=SERIES[1], linestyle="--",
            linewidth=1.4, label="block mean (imaginary)",
        )
        ax.set_xlabel("cycle sample index (blocks x cycles)")
        ax.set_ylabel(value_label)
        ax.set_title(
            f"per-cycle distribution at n={per_cycle_harmonic:g}", loc="left"
        )
        ax.legend(loc="upper right", fontsize=7.5)
    for extra in flat_axes[panel_count:]:
        extra.set_visible(False)
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, color=INK)
    fig.subplots_adjust(top=0.88, hspace=0.5)
    if caption:
        _footer(fig, caption)
    return fig
