# Lorenz agent entry point

Follow [`.agents/core.md`](.agents/core.md) for the behavior that is always active.

Repository skills under `.agents/skills/` provide procedures only when their descriptions match the work. In particular:

- use `repository-orientation` when the current state or ownership is not yet clear;
- use `structural-implementation` for substantive code, configuration, interface, or ownership changes;
- use `debugging` to establish the cause of failures or inconsistent results;
- use `evidence-validation` to decide whether checks support an important claim;
- use `independent-review` when a consequential result needs a fresh challenge;
- use `lorenz-scientific-evidence` for experiment design, numerical or statistical analysis, scientific figures, artifact interpretation, or scientific-claim review.

Skills are selected by the task; they are not a mandatory sequence.

## Scientific authority

- [`PROJECT.md`](PROJECT.md) is authoritative for the project goal, mathematical definitions, estimands, conventions, identifiability statements, and replication semantics.
- [`STATE.md`](STATE.md) is authoritative for maintained capabilities and ownership, accepted numerical and statistical evidence, current scientific decisions, active constraints, and authorization status for further production work.
- [`RETENTION.md`](RETENTION.md) is authoritative for retained-data identity, chunk and checkpoint layout, retention choices, derived-summary status, and storage scaling.

Read the sections relevant to the task before changing a scientific definition, experiment, analysis, or artifact contract. Update these documents only when their owned knowledge has actually changed; do not use them as activity logs.

## Repository boundaries

- `src/lorenz/` owns reusable numerical, statistical, retention, and plotting capabilities. Keep `experiments/` as thin runners, focused analyses, or views that reuse those owners.
- Represent experiment instances in `configs/exploratory/` or `configs/production/`. Do not duplicate mechanisms to encode another parameter choice.
- Keep exploratory work distinct from production or inferential evidence. A production run requires the authorization and design status recorded in `STATE.md`; do not infer authorization from an existing config or old artifact.
- Treat completed simulation artifacts as immutable evidence. Write re-analysis, figures, and other derived products outside those artifacts, following `RETENTION.md` and `STATE.md`.
- Treat `reference/` and historical outputs as supporting material rather than maintained authority.
- `release/lorenz-high-order-dask-mpi/` is a standalone distribution of a deliberate runtime dependency closure, not a second owner of the scientific implementation. When an authorized change affects that distributed closure, update and verify the release deliberately, including its archive checksum; otherwise leave it unchanged.

Keep personal settings, caches, temporary arrays, and generated outputs in their ignored areas. Preserve unrelated working-tree state.
