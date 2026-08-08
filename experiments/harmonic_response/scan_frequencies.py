"""Quick scan for ordinary forcing frequencies with low higher harmonics."""

from __future__ import annotations

from common import execute_experiment, parse_common_args


def main() -> None:
    args = parse_common_args(default_n_traj=4, default_run_name="scan")
    execute_experiment(args, detailed=False)


if __name__ == "__main__":
    main()
