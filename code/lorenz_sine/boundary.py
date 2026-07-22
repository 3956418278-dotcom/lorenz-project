"""Retired ratio boundary implementation.

The former A*=threshold/(|H|/|L|) calculation is intentionally unavailable:
it did not establish detectability or truncation validity.  Use the independent
``validate`` stage instead.
"""


def run(*args, **kwargs):
    raise RuntimeError(
        "boundary.py is retired; run the validate stage with explicit statistical "
        "and truncation-error thresholds"
    )
