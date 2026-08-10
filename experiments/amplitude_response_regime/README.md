# Amplitude response regime experiment

Independent Lorenz-63 experiment for measuring how the actual first-, second-,
third-, and fourth-order harmonic response contributions change with forcing
amplitude.

Outputs are written only to:

```text
outputs/amplitude_response_regime/<run_name>/
```

Default frequencies:

```text
2.265625 1.0546875 0.765625 3.1484375
```

Default amplitudes:

```text
0.02 0.03 0.04 0.05 0.06 0.08 0.10 0.12
```

Quick run:

```bash
python -m experiments.amplitude_response_regime.run_scan
```

Formal run:

```bash
python -m experiments.amplitude_response_regime.run_formal --workers 12 --run-name formal_top4_n256
```

If the default `python` environment lacks numpy/scipy/matplotlib, activate the
Lorenz environment first or replace `python` with the environment interpreter.
