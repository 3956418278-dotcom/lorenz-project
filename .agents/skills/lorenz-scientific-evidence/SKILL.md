---
name: lorenz-scientific-evidence
description: Design, run, analyze, visualize, or review Lorenz numerical experiments and scientific claims with the project's estimands, evidence classes, replication semantics, retention contract, and reproducibility requirements. Use only for scientific work, not ordinary repository engineering.
---

# Lorenz scientific evidence

Build evidence for a scientific decision without weakening the project's definitions or claim standards.

## Start from the scientific contract

Read the relevant parts of `PROJECT.md` and `STATE.md` before planning or interpreting work. Read `RETENTION.md` before choosing retained objects, storage layout, resumption behavior, or derived products.

Treat mathematical definitions, Fourier conventions, estimands, identifiability limits, and the replication unit in `PROJECT.md` as fixed unless the user explicitly approves a scientific redefinition. Treat the accepted results, current decision, active constraints, and production authorization in `STATE.md` as current. Code, old configs, historical branches, references, and prior artifacts do not override those documents.

## Frame the decision and evidence class

State the scientific question, the quantity being estimated or compared, and the decision the result can support. Classify the work before running it:

- **Exploratory:** debugging, reconnaissance, scale estimation, visualization development, or formal-design selection. It may guide later work but is not final inferential evidence.
- **Inferential or production:** evidence intended to support a scientific conclusion. It requires a compatible, authorized design and the project's replication, uncertainty, numerical-bias, and retention semantics.
- **Illustrative:** a cheap representative calculation that improves understanding but is explicitly separated from evidence for the claim.

Do not relabel an exploratory result after seeing it. If the existing evidence cannot answer the question, say so and identify the missing design or observation.

## Design and compare experiments

- Use the estimand and forcing/response conventions already defined in `PROJECT.md`.
- Choose conditions from the scientific question. Comparisons supporting a conclusion must use compatible numerical profiles, ensembles or block identities, sampling grids, contrasts, and retained quantities unless the difference is itself being studied.
- Preserve the initial-state block as the sampling replication unit. Do not treat cycles, samples, or distinct seed labels as independent replicates without a justified sampling model.
- Keep sampling uncertainty separate from spinup, transient, solver, phase-resolution, truncation, and other numerical or model biases. Check each with evidence suited to that source.
- Use paired or crossed structure in the uncertainty calculation when the design shares blocks or conditions. Preserve dependence rather than counting correlated observations as independent.
- Treat amplitude-window choice, polynomial order, tensor reconstruction, and direction design as scientific assumptions to challenge, not plotting choices.
- Do not turn diagnostic recurrence, signal strength, fit quality, or visual salience into a target statistic unless that interpretation is scientifically established.

Use exploratory configurations for pilot work and production configurations only for an accepted, authorized design. Estimate storage and choose retained objects from `RETENTION.md` before an expensive run. Never start production simulation merely because code and a config exist.

## Preserve reproducible evidence

For an important result, retain enough information to recover the code state, complete configuration, input and block identities, environment assumptions, numerical profile, output location, and interpretation. Follow the condition identity, chunk/checkpoint, manifest, and derived-summary rules in `RETENTION.md`.

Completed simulation artifacts are immutable. Resume only through their declared compatibility contract. Put read-only re-analysis, figures, tables, and summaries in designated derived-output areas and keep them reproducible from retained evidence. Keep raw or block-level evidence, derived data, computation, and presentation distinct when their authority differs.

## Interpret and review claims

Organize analyses and figures around the scientific decision. Reuse the project's shared contrast, covariance, inference, and plotting definitions rather than reimplementing them in a one-off view.

Check that uncertainty matches the replication structure; numerical convergence supports the protocol actually used; multiple comparisons or simultaneous claims receive appropriate treatment; and the tested frequency, strength, direction, output, and harmonic scope matches the wording.

Distinguish carefully among:

- **established:** directly supported under the stated design and uncertainty procedure;
- **inconsistent with:** challenged by evidence capable of testing the claim;
- **not established:** evidence is insufficient, incompatible, or outside the tested scope.

A failed bound does not establish a violation. A detected finite-strength response does not by itself establish a zero-strength tensor. A favorable point estimate, selected run, or visually persuasive figure does not replace uncertainty and robustness checks.

Before accepting a consequential scientific result, use the independent-review skill when available to challenge raw evidence, implementation, assumptions, and claim wording from a fresh context. Record accepted numerical/statistical evidence, the resulting decision, and material limitations in `STATE.md`; leave exploratory trails and rejected interpretations outside authoritative memory.
