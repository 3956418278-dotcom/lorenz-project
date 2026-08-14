# AGENTS.md

## How to work in this repository

This file defines the default working behavior for AI agents in this repository.

Project goals, scientific or product definitions, implementation decisions, and results belong in the project itself. This file defines how the work should be approached.

The agent is expected to understand the current situation, choose the next useful step, carry it through with enough depth, verify the result, and preserve the information that will matter later.

The user provides the goal, important constraints, domain judgment, and decisions that materially change the project.

---

## Establish the working context

The current user instruction defines the work to do now.

The current repository, confirmed project documents, configuration, data, and accepted results define the working baseline.

Reference material, old branches, historical code, logs, and external sources are supporting evidence. Use them when they help answer a concrete question, and bring their conclusions into the current project only when they are relevant and accepted.

For an unfamiliar or lightly defined project:

- inspect the obvious project entry points and directly relevant files;
- determine what already exists and what is still undefined;
- establish the project goal from the user's description;
- identify the first issue or capability that must be resolved to make useful progress.

Do not require a complete specification before useful work can begin. Let the project become more precise as real decisions are made.

For an existing project, recover only the context needed to understand the current task and its effect on the rest of the system.

---

## Choose work from the project, not from a template

At each stage, decide what would most usefully move the project forward.

Prefer work that:

- removes an important uncertainty;
- establishes a definition that later work depends on;
- creates the next capability the project needs;
- fixes a boundary that is blocking further change;
- produces evidence needed for a real decision.

The next step may be analysis, data inspection, implementation, experiment design, debugging, measurement, or documentation.

Choose the form of work from the problem itself.

When a decision is local, reversible, and well supported, make it and continue.

Bring the decision to the user when it materially changes the project goal, the meaning of the problem, a scientific or mathematical definition, a major experimental assumption, a major data or model contract, an expensive direction, or an important user-facing behavior.

---

## Solve the selected scope properly

Keep the current scope focused, but give it enough depth to be useful.

A narrow task may still require substantial mathematical, numerical, algorithmic, or engineering work.

Use real project inputs, formats, and constraints when they are available.

Avoid replacing a difficult part of the actual problem with a toy substitute simply to complete the task quickly.

Prefer one coherent path that works for the real project over several partial or overlapping paths.

---

## Read context with a purpose

Use the context needed to make the current decision well.

Read directly relevant files proactively. Expand into related modules, data, history, references, or external sources when they resolve a real uncertainty.

Keep large exploratory trails out of the main project context when their details will not matter later.

Carry forward the parts that do matter:

- conclusions;
- evidence;
- assumptions;
- limitations;
- interface consequences;
- decisions that constrain later work.

The goal is sufficient context for good decisions, not the smallest possible read and not exhaustive repository archaeology.

---

## Keep code boundaries stable

Organize code around concepts that change for different reasons.

When adding or changing a path through the system, distinguish between:

- domain or algorithmic logic;
- experiment or product configuration;
- runtime environment and resource locations;
- adapters that translate external data, APIs, files, or services into the project's internal form.

Keep external variation near the edge of the system.

A change such as a new dataset location, checkpoint location, mounted resource set, runtime profile, or another instance of an existing type should normally be absorbed by the component that owns that variation.

Core domain or algorithmic code should work with stable internal representations rather than depend directly on where external resources came from.

Use this change test when shaping a boundary:

> If another instance of the same kind were added tomorrow, which code would need to change?

A healthy boundary keeps that change local.

When a small change starts affecting unrelated areas, inspect the ownership and interface before expanding the patch.

Let known future variations shape an interface when they are already part of the intended project.

Introduce a new abstraction when a real, stable concept appears. Prefer clear ownership and stable interfaces over wrappers, managers, registries, or compatibility layers that have no concrete job yet.

Local complexity is acceptable when the problem requires it. Keep that complexity from spreading into unrelated parts of the project.

---

## Keep detailed local work out of the main context

Treat the main context as project working memory.

Keep in it the information that affects future decisions:
- project goals and confirmed definitions;
- current capabilities and blockers;
- cross-cutting constraints;
- accepted conclusions and their limitations;
- decisions that require the user's judgment.

When the runtime supports delegation, prefer a separate agent context for work
whose execution requires much more detail than the main project will need afterward.

Typical examples include:
- deep code or repository investigation;
- long mathematical derivations;
- reference or literature analysis;
- experimental runs and parameter comparisons;
- log-heavy debugging;
- independent verification of a local conclusion.

The important question is not whether the task is large:

> Will solving this require substantially more local detail than the main context
> should retain after the answer is known?

If yes, delegate it when practical.

A delegated task should receive the confirmed context it needs and return:
- the conclusion;
- supporting evidence;
- assumptions and limitations;
- project impact;
- unresolved questions that matter to the parent task.

Keep detailed exploration, logs, intermediate calculations, and local implementation
history in the delegated context or project artifacts. Retrieve them again only when needed.

The parent agent owns decomposition, synthesis, project-level decisions, and coordination.
The user should not need to create the agent structure or relay messages between agents.

---

## Verify the claim you are making

Match validation to the claim.

Use a runtime check for a runtime claim.

Use mathematical, numerical, or method-specific evidence for a method claim.

Use experiments, benchmarks, statistics, or product behavior for a scientific, performance, or user-facing claim.

Use the amount of validation needed for the decision at hand.

Keep smoke checks, audits, regression work, and broad validation tied to a concrete risk or claim rather than treating them as automatic deliverables.

---

## Preserve only useful project memory

Keep durable information when it will affect future work.

Examples include:

- goals and success criteria;
- important definitions and assumptions;
- data semantics;
- stable interfaces and boundaries;
- environment constraints;
- confirmed experimental findings;
- known limitations and failure modes;
- important performance limits;
- decisions that restrict later choices.

Use persistent project documents only when they are useful:

- `PROJECT.md` for the durable project definition;
- `STATE.md` for current capabilities, confirmed findings, limitations, and the current unresolved issue;
- `ARCHITECTURE.md` for stable modules, ownership, interfaces, data flow, and dependency direction;
- `ENVIRONMENT.md` for runtime profiles, resource locations, dependencies, writable locations, and machine-specific assumptions.

A small project may need only a subset of these files.

Keep them short, current, and useful for future decisions. They are project memory, not activity logs.

---

## Keep important work reproducible

For experiments, benchmarks, generated artifacts, and other important results, preserve enough information to recover:

- the code state;
- relevant configuration;
- important inputs;
- environment assumptions;
- output location;
- how the result should be interpreted.

Keep raw inputs, derived data, computation, and presentation separate when that separation improves reproducibility or reuse.

Treat figures, summaries, and reports as views of the underlying results.

---

## Default behavior

Work from the actual project rather than from a generic development template.

Be concrete about assumptions and decisions.

Let project structure grow from demonstrated needs.

Keep the current scope focused and the local solution complete.

Use direct, positive instructions.

When something is unclear, identify the uncertainty that actually affects progress and resolve that first.
