---
name: llm-inference-optimizer
description: Theory-guided LLM inference optimization across models, devices, kernels, runtimes and serving systems. Use to review historical attempts, diagnose bottlenecks, estimate optimization gains, choose experiments, implement authorized candidates, compare pre-frozen predictions with real results, and evolve a private cross-project theory library. Do not use for training the target LLM or unrelated generic coding.
---

# Self-evolving theory-guided LLM inference optimization

This skill connects project evidence, physical performance theory, constrained search, independent evaluation and persistent personal theory evolution. Do not merely list tricks. Do not change the task or quality contract to manufacture speedup.

## 1. Select mode and respect the task boundary

- **Review + Plan (default):** read project rules, history, source, logs and theory; return an evidence-based next-round plan. No project edits, GPU allocation, benchmark execution, remote transfers, oracle changes or commits without existing authorization.
- **Execute:** only for authorized implementation/experiment scope and budget; freeze a numerical/qualitative prediction before candidate measurements; implement in an isolated branch/worktree or equivalent; use the independent evaluator; complete the round artifacts below even when rejected.
- **Reflect:** compare new measurements against the frozen prediction; when old experiments lack a prospective prediction, label retrospective analysis, do not backdate or simulate a freeze. Preserve negative results.
- **Learn:** after an actual round, assess theory changes; persist consented evidence locally; construct a candidate theory when justified, not an automatic rewrite. Promotion/rollback require the user's or designated independent reviewer's actual authorization. Never forge an approval reference.

The user can combine modes. All modes inherit existing project permissions. A recommendation is not authority to start experiments. The skill contains no GPU runner or external agent service.

## 2. Locate the theory and knowledge state

Read [the knowledge router](references/knowledge-router.md), [structured theory usage](references/structured-theory.md), and [six-layer theory index](knowledge/structured/INDEX.md). The 33 structured theory units and 37 source documents are an immutable seed; the user's installed skill directory is not the evolving database.

Resolve the skill's own directory (not the current project). Run the offline CLI with Python 3.10+:

```bash
python3 "$SKILL_DIR/scripts/skill.py" resolve
# Only when establishing the local user library:
python3 "$SKILL_DIR/scripts/skill.py" init
```

The CLI uses LLM_OPTIMIZER_HOME or ~/.local/share/llm-inference-optimizer, overridable with --home before the subcommand. resolve returns active version, documents and INDEX path; read the active version's relevant documents first, including the applicability of local revisions. If uninitialized and local persistence is not authorized, use knowledge/structured and knowledge/initial read-only; catalog/lookup/impact/calculate/validate-library also read the packaged seed without creating a store. For an existing v1 library, resolve reports migration_required; retain the old version and request authorization before explicit migrate. Never silently replace private edits. Use catalog with the observed issue, lookup the model ID and transitive prerequisites, and read its applicability and limitations. Calculate only supported narrow evaluators using explicit units and assumptions; the result is conditional, not measured. Read the original research repository at a pinned revision when supplemental context is required; its availability is optional. Record actual theory version, file paths/sections and the seed source commit. Never claim a document was read if inaccessible.

Read the project's own AGENTS.md/README before acting; its constraints apply. Existing project measurements take precedence over example data. Do not import old archived Qwen/H800 constraints into unrelated projects.

## 3. Audit the project and form mechanisms

Follow [project audit](references/project-audit.md). Establish fixed original baseline, current best accepted parent, candidate identities, task/quality/precision contract, device and topology, engine version, workload and metric definitions. Read actual raw evidence when available, not just summaries.

Create a chronological ledger with parent, measured change, implementation status, evidence, comparability, cost and decision. Separate facts, interpretations, counterexamples and unknowns. Never merge incomparable measurements or turn failed/blocked trials into zero gain. For prior work lacking prospective prediction, retain historical evidence separately; do not use it in prediction-accuracy statistics.

Route observed problems to theory using [knowledge router](references/knowledge-router.md). Map each optimization to FLOPs, hierarchical traffic, communication, capacity, dependency path, effective committed tokens or scheduling. Include introduced costs. Use [diagnosis](references/diagnosis.md) to retain discriminating hypotheses, and [decision making](references/decision-making.md) to rank feasible next actions, including measurements. Prefer two or three substantive directions, not a fixed count. Check whether old failures become relevant after a bottleneck changes. Do not claim a bound is achievable or a model estimate a strict bound.

## 4. Before each authorized candidate measurement

1. Fix the baseline and comparator, quality oracle, workload and measurement protocol. Record metric units, statistic, direction, resource/execution/service scope and baseline drift tolerance **before** the test. Set the tolerance from evidence/project policy, not an invented threshold.
2. Derive a point, conditional interval, strict lower/upper bound or explicitly unknown prediction for each relevant metric. Predict resource and execution changes as well as service-level effect when identifiable. Include mechanism, assumptions, costs, expected bottleneck migration, uncertainty and what would falsify the explanation. If baseline is unknown, first recover it or propose the necessary baseline experiment; do not fill an unknown with zero.
3. Prepare [prediction JSON](assets/prediction.json) using [prediction contract](references/prediction-comparison.md). The file is a synthetic example, not a project prediction. Set stable project/round IDs and candidate ID, actual parent ref, reproducible context ID, active version and theory refs. The candidate ID labels the planned intervention; the implemented commit is recorded separately in the measurement. In structured versions each metric must bind model_ref (stable id, revision, output, optional parameter_sets) or an explicit model_gap. Check context, units and bounds; do not fabricate a link. The shadow must bind its own model_refs for all target points.
4. Run `skill.py freeze <new-round-dir> <prediction-input.json>`. It creates immutable prediction JSON, Markdown and hash receipt. The directory must not already exist. Never overwrite or backdate it; revise a plan by opening a new candidate/round. If a candidate is materially changed, freeze a new prediction before measuring it.
5. A holdout round uses validation_role=holdout and a preregistered candidate theory as shadow; freeze both versions' predictions on exactly the same metric definition. Never fit using this holdout before comparing.

A genuinely unknown numerical prediction is acceptable; a missing prospective prediction is not a completed prediction-validation round. Read [evaluation](references/evaluation.md).

## 5. After every candidate, including a failed one

In the project-local round folder, leave:

- prediction.json / prediction.md / freeze.json (generated before candidate observations)
- result/measurement.json / comparison.json / comparison.md / receipt.json (generated by compare)
- theory-update.md (agent explanation; use [template](assets/theory-update.md))
- theory-update.json (structured versions: fill every primary model/metric decision and rationale from result/module-review-template.json; review gaps and optional dependency modules)
- next-plan.md (ranked follow-up/stop decision; use [template](assets/next-plan.md))

Use independent evaluation; record raw evidence references, actual implementation identity, evaluator identity, correctness, comparability and evidence class. Include resource, execution and service metrics when available. Missing measurements are omitted with explanation; failed, inconclusive and blocked rounds still have terminal measurement records. Never claim a measurement that was not run.

Run `skill.py compare <round-dir> <measurement-input.json>`. The CLI checks identity, metric definitions, baseline drift, correctness and evidence; calculates prediction/actual gains, signed errors, interval coverage and physical bound satisfaction separately. A result directory is immutable. If a data error is discovered preserve it, document the issue and make a new superseding round rather than overwriting the receipt. Use the output plus project logs to explain **which physical mechanism changed, which prediction was wrong and why, what remains unidentified**; do not use discrepancy alone as proof of a cause.

If local user-level accumulation is authorized, use `skill.py record <round-dir> <round-dir>/theory-update.md`. It copies evidence and theory note into the private local store, refuses simulated records and duplicate round IDs, and checks hashes. Unavailable local storage is a blocker for durable learning, not for returning the report. Store private data only locally; do not publish it, push to GitHub or transfer across users by default.

## 6. Evolve the theory, not just the system

Follow [theory evolution](references/theory-evolution.md). Every round must assess: keep existing theory, calibrate parameters, propose a correction, extension, empirical law or structural revision; record a no-change reason when warranted. Separate a conditional empirical finding from a mathematical law. Failed candidates and unknowns inform hypothesis generation, but are not fabricated numeric training samples.

- `summary` and `module-summary`: compare errors by version/context/metric and primary module ID, revision, output and selected parameter sets. Report coverage, unlinked gaps and sample counts; primary prediction provenance is not causal attribution.
- `calibrate <metric> <project/round> ...`: optional robust multiplicative fit from at least two compatible measured search rounds; a proposed fit, not evidence of improvement. Check known bounds and scope.
- `propose <proposal.json>`: create a full immutable candidate snapshot, including Markdown and structured files. For v2, revise model.json + theory.md together (increment revision) or add new immutable context-specific parameter sets; use evidence-linked search data. Rebuild registry/dependencies, record changed and impacted modules and retire outdated sets. Formula changes must disable an unchanged executable evaluator until independently updated. Require preregistered targets and holdouts.
- Evaluate all preregistered unseen rounds against the active and shadow versions (paired, same metric).
- `promote <candidate-version> <real-review-reference> <all-holdout-ids...>`: require no overlap, matching context coverage, comparable measured holdout counts, no MAE regression on declared targets and at least one improvement. For v2, also check per-target context, coverage of changed modules in the paired prediction chains and actual use of new parameter set IDs in eligible shadow targets. This numeric gate is not a sufficient evaluation of nonnumeric methods; use independent research/benchmark review and keep such proposals pending if they cannot satisfy this gate. This is only a mechanical minimum; independent reviewer must verify scope, meaningful improvement, physical invariants, leakage and safety. No fabricated approval or autonomous promotion.
- `rollback <ancestor-version> <real-review-reference>`: explicit rollback of active pointer, retain all evidence, versions and review receipts.

Keep version manifests and generated indices. See [structured theory usage](references/structured-theory.md) for model types, migration, schemas and proposal examples. Skill updates must not replace a user's active knowledge; new upstream seeds need deliberate migration/review. On incompatible contexts transfer mechanism/priors, not raw calibrated rates. No cross-user sharing unless explicitly authorized and sanitized.

## 7. Deliver

For Review/Plan use [optimization report](assets/optimization-report.md); for Execute/Reflect return the round artifacts and summarize prediction vs measurement, correctness, accepted/rejected/undecided, remaining bottleneck, proposed model update and next decision. Clearly state missing measurements, budget, authorization and theoretical scope. Link actual evidence paths and the exact theory revision. The analysis remains useful if the CLI or GPU is unavailable; report the blocker instead of claiming files were saved or measured.
