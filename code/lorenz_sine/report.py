"""Final multi-page PDF report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def make_report(run_dir, cfg, spectrum=None, steady=None, responses=None, boundary=None):
    path = Path(run_dir) / "report.pdf"
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        text = json.dumps({
            "mode": "all",
            "n_seed": cfg["n_seed"],
            "frequencies": cfg["frequencies"],
            "amplitudes": cfg["amplitudes"],
            "n_phase": cfg["n_phase"],
            "bootstrap_samples": cfg["bootstrap_samples"],
        }, indent=2)
        ax.text(0.02, 0.98, "Configuration and run overview\n\n" + text,
                va="top", family="monospace", fontsize=8)
        pdf.savefig(fig); plt.close(fig)

        if spectrum is not None:
            fig, ax = plt.subplots(figsize=(8, 5))
            freqs = spectrum["freqs"]
            psd = spectrum["psd"]
            for i, lab in enumerate(["x", "y", "z"]):
                ax.loglog(freqs[1:], psd[i, 1:], label=lab)
            ax.set_title("Unforced natural spectrum")
            ax.set_xlabel("f")
            ax.set_ylabel("PSD")
            ax.legend(frameon=False)
            pdf.savefig(fig); plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 4))
            peaks = spectrum["peaks"]
            if len(peaks):
                ax.bar(np.arange(len(peaks)), peaks[:, 2])
                ax.set_xticks(np.arange(len(peaks)))
                ax.set_xticklabels([f"{w:.3g}" for w in peaks[:, 2]], rotation=45)
            ax.set_title("Natural spectrum candidate omega")
            pdf.savefig(fig); plt.close(fig)

        if steady is not None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.axis("off")
            ax.text(0.02, 0.98, "Periodic steady-state checks\n\n" + json.dumps(steady, indent=2, default=str),
                    va="top", family="monospace", fontsize=7)
            pdf.savefig(fig); plt.close(fig)

        if responses:
            r = responses[0]
            theta = np.linspace(0, 2 * np.pi, cfg["n_phase"], endpoint=False)
            fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
            for i, ax in enumerate(axs):
                ax.plot(theta, r["L"][i, 0], label="x forcing")
                ax.plot(theta, r["L"][i, 1], label="y forcing")
                ax.plot(theta, r["L"][i, 2], label="z forcing")
                ax.set_ylabel(["x", "y", "z"][i])
            axs[0].legend(frameon=False)
            axs[-1].set_xlabel("theta")
            fig.suptitle("First-order response phase curves")
            pdf.savefig(fig); plt.close(fig)

            fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
            for i, ax in enumerate(axs):
                ax.plot(theta, r["H"][i, 0, 0], label="xx")
                ax.plot(theta, r["H"][i, 0, 1], label="xy")
                ax.plot(theta, r["H"][i, 2, 2], label="zz")
                ax.set_ylabel(["x", "y", "z"][i])
            axs[0].legend(frameon=False)
            axs[-1].set_xlabel("theta")
            fig.suptitle("Second-order response phase curves")
            pdf.savefig(fig); plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot([rr["omega"] for rr in responses],
                    [np.linalg.norm(rr["L_amp"][..., 1]) for rr in responses], "o-", label="L k=1")
            ax.plot([rr["omega"] for rr in responses],
                    [np.linalg.norm(rr["H_amp"][..., 0]) for rr in responses], "o-", label="H k=0")
            ax.plot([rr["omega"] for rr in responses],
                    [np.linalg.norm(rr["H_amp"][..., 2]) for rr in responses], "o-", label="H k=2")
            ax.set_title("Frequency response scan")
            ax.set_xlabel("omega")
            ax.legend(frameon=False)
            pdf.savefig(fig); plt.close(fig)

        if boundary is not None:
            fig, ax = plt.subplots(figsize=(8, 4))
            rows = np.asarray(boundary, dtype=float)
            if rows.size:
                ax.plot(rows[:, 0], rows[:, 5], "o")
            ax.set_title("Linear boundary")
            ax.set_xlabel("omega")
            ax.set_ylabel("A*")
            pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.02, 0.98, "Convergence/statistical check page\n\nSee summary.csv and result.npz for diagnostics.",
                va="top")
        pdf.savefig(fig); plt.close(fig)
    return path
