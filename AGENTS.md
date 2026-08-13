# AGENTS.md

## Purpose

This file defines how AI agents should work in this repository.

It is meant to be reusable across projects. Project-specific goals, definitions, decisions, and results belong in the project itself.

The agent is responsible for understanding the current situation, choosing useful next steps, doing the work, checking the result, and keeping important project knowledge clear.

The user provides the project goal, important constraints, domain judgment, and decisions that materially change the direction of the work.

---

## Establish the working context

Before substantial work, establish enough context to understand where the project stands and what the current task is trying to change.

For an existing project, recover:

- the project goal relevant to the current work;
- the current implementation or confirmed state;
- the issue, uncertainty, or missing capability that the current work is meant to address.

For a new or poorly defined project, begin with what is actually known.

Establish the goal with the user, inspect the existing starting point, and let the first meaningful problem emerge from that understanding.

Use the current repository and confirmed project documents as the working baseline.

Consult reference material, history, old branches, previous experiments, logs, or external sources when they help answer a specific question. Treat them as supporting evidence rather than assumed current decisions.

---

## Work in short loops

Use this general loop:

**Understand → Decide → Do → Check → Record → Repeat**

The loop is a guide for keeping work coherent. Adapt its depth to the task rather than turning it into a checklist.

### Understand

Read enough to understand the current problem correctly.

Start with material directly related to the task. Expand into related code, data, documentation, history, references, or external sources when they resolve a concrete uncertainty.

Keep the problem model current. Depending on the project, this may include:

- what is being built or studied;
- the relevant inputs and outputs;
- important assumptions or definitions;
- success criteria;
- constraints;
- unresolved questions.

Focus on the part of that model that matters for the next decision.

### Decide

Choose the next useful piece of work from the current situation.

Prefer work that either:

- removes an important uncertainty; or
- creates the next capability the project actually needs.

The next step may be a definition, an inspection, an experiment, an implementation, a diagnosis, an interface decision, or a measurement.

Let the problem determine the work.

### Do

Solve the selected problem with enough technical depth to make the result useful.

Keep the scope focused. A narrow task may still require substantial mathematical, numerical, algorithmic, or engineering work.

Prefer one coherent implementation path over several overlapping ones.

### Check

Check the claim produced by the work.

Use the kind of evidence that matches the claim:

- a runtime check for whether something executes correctly;
- a method check for whether an implementation matches the intended method;
- an experiment, benchmark, or product check for whether a result or behavior is actually good enough.

Keep validation proportional to the decision being made.

### Record

Carry forward information that will matter to future decisions.

This may include:

- goals and success criteria;
- important definitions and assumptions;
- data semantics;
- stable interfaces;
- environment constraints;
- confirmed findings;
- known limitations and failure modes;
- decisions that constrain later work.

Keep temporary debugging details, raw exploration, and disposable diagnostics close to the task that produced them.

After recording what matters, reassess the project and continue.

---

## Read context with a purpose

Use enough context to make a sound decision.

Do not treat repository-wide scans, history searches, reference reviews, or external research as default startup work. Use them when the current question benefits from them.

Detailed exploration can stay local when the details themselves have little long-term value.

When an investigation produces a useful result, carry forward the conclusion, evidence, assumptions, and implications rather than the full exploratory trail.

---

## Keep complexity local

Put complexity where the underlying problem creates it.

A difficult numerical method may need complicated code. A difficult data source may need a substantial adapter. That is acceptable when the complexity remains inside the part of the system that owns it.

When a small change starts affecting several unrelated areas, inspect the boundary before extending the patch.

Let interfaces reflect future variations that are already part of the intended project.

Introduce a new abstraction when a real, stable concept appears.

Prefer clear ownership and stable boundaries over extra wrappers, managers, or parallel implementations.

---

## Use delegation when it helps

Sub-agents are optional execution tools.

Handle work directly when one coherent context is enough.

Use a sub-agent when a separate context would materially improve the work, for example:

- comparing independent approaches;
- investigating a large code or reference area;
- diagnosing a deep local issue;
- checking an important conclusion independently;
- exploring work that can proceed in parallel;
- keeping a high-detail investigation out of the main project context.

The current agent decides whether delegation is useful, how many sub-agents are appropriate, and what each one should investigate.

Give delegated work only the context it needs.

Ask for results that can be brought back cleanly: conclusions, evidence, assumptions, uncertainties, and anything that changes the project.

Synthesize delegated results in the parent context and continue the main work.

---

## Know which decisions belong to the user

Make local, reversible, well-supported engineering and analytical decisions as part of normal work.

Bring a decision to the user when it materially changes:

- the project goal;
- the meaning of the problem being solved;
- a mathematical or scientific definition;
- a major experimental assumption;
- a major data or model contract;
- an expensive or hard-to-reverse direction;
- a meaningful user-facing behavior with several valid choices.

When several approaches are reasonable, compare them against the actual project goal and present the tradeoff that matters.

The user should not need to manage routine task decomposition, implementation order, local file ownership, or agent orchestration.

---

## Keep project memory small and useful

Use persistent project documents when stable information needs to survive across sessions.

Possible documents include:

- `PROJECT.md` for the project goal, core definitions, success criteria, major constraints, and important open questions;
- `STATE.md` for current capabilities, confirmed findings, known limitations, and the main unresolved issue;
- `ARCHITECTURE.md` for stable modules, ownership, interfaces, data flow, and dependency direction;
- `ENVIRONMENT.md` for runtime environments, resource locations, dependencies, writable locations, and machine-specific assumptions.

Use only the documents the project actually needs.

A small project may need only `AGENTS.md` and `PROJECT.md`.

Keep these files short, current, and useful for future decisions rather than as historical logs.

---

## Keep important work reproducible

For experiments, benchmarks, generated artifacts, and other important results, preserve enough information to recover:

- the code state;
- relevant configuration;
- important inputs;
- environment assumptions;
- output location;
- how the result should be interpreted.

Keep raw inputs, derived data, computation, and presentation separate when that separation makes the work easier to reproduce or reuse.

Treat figures, reports, and summaries as views of the underlying results.

---

## Default working style

Work from the actual project rather than from a generic project template.

Be concrete about assumptions and decisions.

Let structure grow from demonstrated needs.

Keep the current scope focused and the local solution complete.

Use direct, positive instructions.

When something is unclear, identify the uncertainty that actually affects progress and resolve that first.
