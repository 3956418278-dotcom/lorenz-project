"""Detailed rerun for harmonic-response candidates."""

from __future__ import annotations

from common import execute_experiment, parse_common_args


def main() -> None:
    args = parse_common_args(default_n_traj=24, default_run_name="detailed")
    execute_experiment(args, detailed=True)


if __name__ == "__main__":
    main()
