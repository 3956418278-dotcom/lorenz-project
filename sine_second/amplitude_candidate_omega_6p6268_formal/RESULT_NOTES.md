## Candidate amplitude scan artifact

This directory is the completed formal amplitude scan for the prior candidate
frequency:

- angular frequency: `omega = 6.62679700366597`
- cycle frequency: `f = 1.0546875`
- amplitude grid: `[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]`
- seed groups: `ngrp = 64`
- bootstrap samples: `2000`
- primary forcing direction for report figures: `z`

The interrupted low-frequency run (`omega = 0.2`) is intentionally not included
in this result artifact.

Interpretation caveat: this result should be treated as an audit artifact for
the current implementation. The SNR curves remain close to one across the
amplitude grid, so this run does not establish a clean amplitude window where
1ω and 2ω clearly emerge above the raw chaotic background.
