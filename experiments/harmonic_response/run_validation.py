"""Independent validation run for selected harmonic-response frequencies."""

from __future__ import annotations

from common import execute_experiment, parse_common_args


def main() -> None:
    args = parse_common_args(default_n_traj=256, default_run_name="validation")
    if not args.run_name.startswith("validation_"):
        args.run_name = f"validation_{args.run_name}"
    if args.seed_offset == 0:
        args.seed_offset = 1000000
    execute_experiment(args, detailed=True)


if __name__ == "__main__":
    main()
