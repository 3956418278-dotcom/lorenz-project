"""Linear boundary estimates and finite-amplitude validation summaries."""

from __future__ import annotations

import numpy as np


def l2_phase_norm(x):
    return np.sqrt(np.mean(np.asarray(x) ** 2))


def run(responses, cfg):
    rows = []
    for r in responses:
        L = r["L"]
        H = r["H"]
        omega = float(r["omega"])
        for d in cfg["boundary"]["directions"]:
            d = np.asarray(d, dtype=float)
            d = d / max(np.linalg.norm(d), 1e-300)
            lin = np.einsum("ijq,j->iq", L, d)
            quad = 0.5 * np.einsum("ijkq,j,k->iq", H, d, d)
            ratio_unit = l2_phase_norm(quad) / max(l2_phase_norm(lin), 1e-300)
            for thr in cfg["boundary"]["ratio_thresholds"]:
                a_star = thr / max(ratio_unit, 1e-300)
                rows.append([omega, *d.tolist(), thr, a_star, ratio_unit])
    return rows
