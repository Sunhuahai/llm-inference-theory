#!/usr/bin/env python3
"""Offline, synthetic candidate-search replay. No LLM, code execution or GPU use.

Policies receive only immutable public candidates and completed observations.
The evaluator's outcome table is kept out of the selection interface. This is
an information-flow demonstration, not an OS/security isolation mechanism.
All candidates are independent variants of the same supplied baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import random
import sys
from typing import Any


POLICIES = ("random", "predicted", "cost-aware")
PUBLIC_FIELDS = {
    "id", "parent_id", "mechanism", "predicted_latency_ms", "uncertainty_ms",
    "feasibility_probability", "estimated_cost_units", "cost_cap_units",
}


class InputError(ValueError):
    """The fixture violates the deliberately small replay contract."""


def number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise InputError(f"{name} cannot be represented as a finite number") from error
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise InputError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def cost_number(value: Any, name: str, *, positive: bool = False) -> Decimal:
    """Use canonical decimal inputs for costs, retaining exact integer values.

    Validation imposes the same finite numeric range as other inputs. Decimal
    avoids rejecting 0.1 + 0.2 against a 0.3 budget and does not round large
    integral costs to binary float before comparing them with their limits.
    """
    number(value, name, positive=positive)
    return Decimal(value) if isinstance(value, int) else Decimal(str(value))


def json_cost(value: Decimal) -> int | float:
    """JSON keeps ordinary numeric fields; exact cumulative text is also emitted."""
    if value == value.to_integral_value():
        return int(value)
    return number(float(value), "serialized cost")


def fraction(value: Any, name: str) -> float:
    value = number(value, name)
    if value > 1:
        raise InputError(f"{name} must be in [0, 1]")
    return value


def text_field(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{name} must be nonempty text")
    return value


def object_field(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputError(f"{name} must be an object")
    return value


@dataclass(frozen=True)
class Candidate:
    id: str
    parent_id: str
    mechanism: str
    predicted_latency_ms: float
    uncertainty_ms: float
    feasibility_probability: float
    estimated_cost_units: Decimal
    cost_cap_units: Decimal


@dataclass(frozen=True)
class Observation:
    candidate_id: str
    evaluation_status: str
    correctness_pass: bool | None
    observed_latency_ms: float | None
    cost_units: Decimal
    decision: str
    reason: str
    predicted_latency_ms: float
    estimated_cost_units: Decimal
    cost_cap_units: Decimal
    cumulative_cost_units: Decimal
    incumbent_id: str
    incumbent_latency_ms: float


class ReplayEvaluator:
    """Reveal only a selected outcome; cap overruns become censored timeouts."""

    def __init__(self, outcomes: dict[str, Any]) -> None:
        self._outcomes = outcomes

    def evaluate(self, candidate: Candidate) -> dict[str, Any]:
        outcome = self._outcomes[candidate.id]
        if outcome["cost_units"] > candidate.cost_cap_units:
            return {
                "status": "timeout", "correctness_pass": None,
                "latency_ms": None, "cost_units": candidate.cost_cap_units,
                "reason": "Evaluation did not complete within its reserved cost cap.",
            }
        return dict(outcome)


def parse_fixture(data: Any) -> tuple[dict[str, Any], tuple[Candidate, ...], ReplayEvaluator]:
    data = object_field(data, "fixture")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise InputError("schema_version must be integer 1")
    if data.get("evidence_origin") != "synthetic":
        raise InputError("this replay accepts synthetic fixtures only")
    text_field(data.get("task_id"), "task_id")
    baseline = object_field(data.get("baseline"), "baseline")
    baseline_id = text_field(baseline.get("id"), "baseline.id")
    baseline_ms = number(baseline.get("latency_ms"), "baseline.latency_ms", positive=True)
    setup_cost = cost_number(baseline.get("setup_cost_units"), "baseline.setup_cost_units")
    threshold = number(data.get("minimum_improvement_ms", 0), "minimum_improvement_ms")
    rows = data.get("candidates")
    if not isinstance(rows, list):
        raise InputError("candidates must be a list")
    candidates: list[Candidate] = []
    for i, raw in enumerate(rows):
        row = object_field(raw, f"candidates[{i}]")
        if set(row) != PUBLIC_FIELDS:
            raise InputError(f"candidates[{i}] must contain exactly public candidate fields")
        parent_id = text_field(row["parent_id"], "parent_id")
        if parent_id != baseline_id:
            raise InputError("replay supports independent candidates of one baseline only")
        candidate = Candidate(
            id=text_field(row["id"], "id"), parent_id=parent_id,
            mechanism=text_field(row["mechanism"], "mechanism"),
            predicted_latency_ms=number(row["predicted_latency_ms"], "predicted_latency_ms", positive=True),
            uncertainty_ms=number(row["uncertainty_ms"], "uncertainty_ms"),
            feasibility_probability=fraction(row["feasibility_probability"], "feasibility_probability"),
            estimated_cost_units=cost_number(row["estimated_cost_units"], "estimated_cost_units", positive=True),
            cost_cap_units=cost_number(row["cost_cap_units"], "cost_cap_units", positive=True),
        )
        if candidate.estimated_cost_units > candidate.cost_cap_units:
            raise InputError("estimated cost cannot exceed the declared per-candidate hard cap")
        candidates.append(candidate)
    ids = [candidate.id for candidate in candidates]
    if len(set(ids)) != len(ids) or baseline_id in ids:
        raise InputError("candidate ids must be unique and different from baseline.id")
    raw_outcomes = object_field(data.get("outcomes"), "outcomes")
    if set(raw_outcomes) != set(ids):
        raise InputError("outcomes must match candidate ids exactly")
    outcomes: dict[str, Any] = {}
    for candidate_id, raw in raw_outcomes.items():
        row = object_field(raw, f"outcomes.{candidate_id}")
        if set(row) != {"status", "correctness_pass", "latency_ms", "cost_units", "reason"}:
            raise InputError("each outcome must have status/correctness_pass/latency_ms/cost_units/reason")
        status = row["status"]
        if status not in ("completed", "compile_error", "runtime_error", "timeout"):
            raise InputError("unsupported outcome status")
        correct, latency = row["correctness_pass"], row["latency_ms"]
        if status == "completed":
            if type(correct) is not bool:
                raise InputError("completed outcome correctness_pass must be boolean")
            latency = number(latency, "outcome.latency_ms", positive=True)
        elif correct is not None or latency is not None:
            raise InputError("failed outcomes must not contain correctness or latency claims")
        outcomes[candidate_id] = {
            "status": status, "correctness_pass": correct, "latency_ms": latency,
            "cost_units": cost_number(row["cost_units"], "outcome.cost_units"),
            "reason": text_field(row["reason"], "outcome.reason"),
        }
    metadata = {
        "task_id": data["task_id"], "baseline_id": baseline_id,
        "baseline_latency_ms": baseline_ms, "setup_cost_units": setup_cost,
        "minimum_improvement_ms": threshold,
    }
    return metadata, tuple(sorted(candidates, key=lambda x: x.id)), ReplayEvaluator(outcomes)


def select_candidate(
    candidates: tuple[Candidate, ...], history: tuple[Observation, ...],
    incumbent_ms: float, *, policy: str, rng: random.Random, exploration: float,
) -> Candidate:
    """Public selection boundary. No evaluator, raw fixture or future outcomes."""
    if not candidates:
        raise InputError("selection requires at least one affordable candidate")
    if policy == "random":
        return rng.choice(candidates)
    if policy == "predicted":
        return min(candidates, key=lambda x: (x.predicted_latency_ms, x.id))
    if policy != "cost-aware":
        raise InputError(f"unknown policy: {policy}")
    if rng.random() < exploration:
        return rng.choice(candidates)

    def score(candidate: Candidate) -> float:
        optimistic_gain = number(
            max(0.0, incumbent_ms - candidate.predicted_latency_ms + candidate.uncertainty_ms),
            f"optimistic gain for {candidate.id}",
        )
        return number(
            candidate.feasibility_probability * optimistic_gain / float(candidate.estimated_cost_units),
            f"cost-aware score for {candidate.id}",
        )

    # The history parameter supports future policies; this simple policy only
    # adapts to the incumbent, which itself comes from completed observations.
    return min(candidates, key=lambda x: (-score(x), x.id))


def run_replay(
    data: Any, *, policy: str = "cost-aware", budget: float = 12,
    seed: int = 0, exploration: float = 0.2,
) -> dict[str, Any]:
    budget_decimal = cost_number(budget, "budget")
    exploration = fraction(exploration, "exploration")
    if policy not in POLICIES:
        raise InputError(f"policy must be one of {POLICIES}")
    if type(seed) is not int:
        raise InputError("seed must be an integer")
    meta, candidates, evaluator = parse_fixture(data)
    best_id, best_ms = meta["baseline_id"], meta["baseline_latency_ms"]
    spent = Decimal(0)
    history: list[Observation] = []
    pending = {candidate.id: candidate for candidate in candidates}
    rng = random.Random(seed)
    stop_reason = "insufficient_setup_budget"
    # Finite binary floats have decimal exponents between roughly -324 and
    # +308; accepted integers are also bounded by finite-float range. Eight
    # hundred digits therefore preserve every admitted ledger addition and
    # subtraction exactly, without modifying the caller's decimal context.
    with localcontext() as context:
        context.prec = 800
        if meta["setup_cost_units"] <= budget_decimal:
            spent = meta["setup_cost_units"]
            stop_reason = "exhausted_candidates"
            while pending:
                remaining = budget_decimal - spent
                affordable = tuple(x for x in pending.values() if x.cost_cap_units <= remaining)
                if not affordable:
                    stop_reason = "no_affordable_candidates"
                    break
                selected = select_candidate(
                    affordable, tuple(history), best_ms,
                    policy=policy, rng=rng, exploration=exploration,
                )
                if selected not in affordable:
                    raise InputError("policy selected an unavailable or unaffordable candidate")
                del pending[selected.id]
                outcome = evaluator.evaluate(selected)
                # Reserve the full public cap; settle only the selected actual
                # cost. Never clip a true overrun to conceal overspending.
                next_spent = spent + outcome["cost_units"]
                if outcome["cost_units"] > selected.cost_cap_units or next_spent > budget_decimal:
                    raise InputError("evaluator cost exceeds the reserved or remaining budget")
                spent = next_spent
                decision = "rejected"
                status = outcome["status"]
                reason = outcome["reason"]
                if status == "completed":
                    if not outcome["correctness_pass"]:
                        status = "incorrect"
                    elif best_ms - outcome["latency_ms"] > meta["minimum_improvement_ms"]:
                        # Validate only the selected observed result, never
                        # future outcomes. Extreme positive inputs can still
                        # produce an unrepresentable derived speedup.
                        number(
                            meta["baseline_latency_ms"] / outcome["latency_ms"],
                            f"replay speedup for {selected.id}", positive=True,
                        )
                        best_id, best_ms = selected.id, outcome["latency_ms"]
                        decision = "accepted"
                    else:
                        status = "not_improved"
                history.append(Observation(
                    selected.id, status, outcome["correctness_pass"], outcome["latency_ms"],
                    outcome["cost_units"], decision, reason, selected.predicted_latency_ms,
                    selected.estimated_cost_units, selected.cost_cap_units, spent, best_id, best_ms,
                ))
    serialized_history = []
    for observation in history:
        row = {key: json_cost(value) if isinstance(value, Decimal) else value
               for key, value in asdict(observation).items()}
        row["cumulative_cost_units_exact"] = str(observation.cumulative_cost_units)
        serialized_history.append(row)
    return {
        "schema_version": 1, "implementation": "offline-search-replay-v1",
        "status": "simulated", "evidence_origin": "synthetic",
        "scope": "finite independent same-baseline candidates; no LLM or GPU execution",
        "task_id": meta["task_id"], "policy": policy, "seed": seed,
        "exploration": exploration if policy == "cost-aware" else None,
        "budget_cost_units": json_cost(budget_decimal), "spent_cost_units": json_cost(spent),
        "budget_cost_units_exact": str(budget_decimal), "spent_cost_units_exact": str(spent),
        "baseline_latency_ms": meta["baseline_latency_ms"],
        "best_candidate_id": best_id, "best_replay_latency_ms": best_ms,
        "replay_speedup": number(meta["baseline_latency_ms"] / best_ms, "replay_speedup", positive=True),
        "attempted_candidates": len(history), "stop_reason": stop_reason,
        "unevaluated_candidate_ids": list(pending),
        "history": serialized_history,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--policy", choices=POLICIES, default="cost-aware")
    parser.add_argument("--budget", type=float, default=12, help="Synthetic cost-unit budget, including setup.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exploration", type=float, default=0.2, help="Random exploration fraction for cost-aware.")
    args = parser.parse_args()
    try:
        data = json.loads(args.fixture.read_text(encoding="utf-8"))
        result = run_replay(data, policy=args.policy, budget=args.budget, seed=args.seed, exploration=args.exploration)
    except (InputError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
