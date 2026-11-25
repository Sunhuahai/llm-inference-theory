#!/usr/bin/env python3
"""Small resource/DAG calculator. No GPU, fitting, or serving simulation.

All input times are computed from work / rates. Physical upper rates give
conditional lower bounds; effective rates give model estimates, never bounds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


class InputError(ValueError):
    """A statement cannot be interpreted under this calculator's contract."""


def number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{path}: expected a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise InputError(f"{path}: numeric value exceeds finite range") from error
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise InputError(f"{path}: expected {'positive' if positive else 'nonnegative'} finite value")
    return result


def finite_derived(value: float, path: str) -> float:
    """Check computed ratios/errors without rejecting legitimate negative errors."""
    if not math.isfinite(value):
        raise InputError(f"{path}: derived value exceeds the finite numeric range")
    return value


def obj(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputError(f"{path}: expected an object")
    return value


def label(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{path}: expected a nonempty string")
    return value


def string_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
        raise InputError(f"{path}: expected a list of nonempty strings")
    if len(set(value)) != len(value):
        raise InputError(f"{path}: duplicate identifiers")
    return value


def _rates(data: dict[str, Any], mode: str) -> dict[str, dict[str, Any]]:
    rates = obj(data.get("resources"), "resources")
    if not rates:
        raise InputError("resources: at least one resource is required")
    expected = "physical_upper" if mode == "bound" else "effective_forecast"
    parsed = {}
    for name, raw in rates.items():
        label(name, "resource id")
        raw = obj(raw, f"resources.{name}")
        if raw.get("rate_kind") != expected:
            raise InputError(f"resources.{name}: {mode} mode requires {expected}; do not mix rate semantics")
        if raw.get("work_unit") not in ("FLOP", "byte", "operation"):
            raise InputError(f"resources.{name}.work_unit: use FLOP, byte, or operation")
        if raw.get("rate_unit") != raw["work_unit"] + "/s":
            raise InputError(f"resources.{name}: work/rate units do not match")
        parsed[name] = {
            "rate": number(raw.get("rate"), f"resources.{name}.rate", positive=True),
            "work_unit": raw["work_unit"],
            "source": label(raw.get("source"), f"resources.{name}.source"),
        }
    return parsed


def _scenario(raw: Any, resources: dict[str, dict[str, Any]], mode: str) -> dict[str, Any]:
    scenario = obj(raw, "scenario")
    sid = label(scenario.get("id"), "scenario.id")
    semantics = scenario.get("graph_semantics")
    if semantics not in ("dependencies_only", "fixed_schedule"):
        raise InputError(f"{sid}: graph_semantics must be dependencies_only or fixed_schedule")
    nodes_raw = scenario.get("nodes")
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise InputError(f"{sid}.nodes: expected a nonempty list")
    nodes: dict[str, dict[str, Any]] = {}
    resource_totals = {r: 0.0 for r in resources}
    for raw_node in nodes_raw:
        node = obj(raw_node, f"{sid}.node")
        nid = label(node.get("id"), f"{sid}.node.id")
        if nid in nodes:
            raise InputError(f"{sid}: duplicate node {nid}")
        deps = string_list(node.get("after", []), f"{sid}.{nid}.after")
        demands = obj(node.get("demands"), f"{sid}.{nid}.demands")
        if not demands:
            raise InputError(f"{sid}.{nid}: at least one demand is required")
        times, amounts = {}, {}
        for resource, raw_demand in demands.items():
            if resource not in resources:
                raise InputError(f"{sid}.{nid}: unknown resource {resource}")
            demand = obj(raw_demand, f"{sid}.{nid}.{resource}")
            if demand.get("unit") != resources[resource]["work_unit"]:
                raise InputError(f"{sid}.{nid}.{resource}: demand unit mismatch")
            amount = number(demand.get("amount"), f"{sid}.{nid}.{resource}.amount")
            amounts[resource] = amount
            times[resource] = number(amount / resources[resource]["rate"] * 1000,
                                     f"{sid}.{nid}.{resource}.computed_ms")
            resource_totals[resource] = number(resource_totals[resource] + amount,
                                               f"{sid}.{resource}.total_work")
        nodes[nid] = {"after": set(deps), "times": times, "amounts": amounts}

    for nid, node in nodes.items():
        unknown = node["after"] - nodes.keys()
        if unknown:
            raise InputError(f"{sid}.{nid}: missing dependencies {sorted(unknown)}")
    orders = scenario.get("serialization_orders", [])
    if not isinstance(orders, list):
        raise InputError(f"{sid}.serialization_orders: expected a list")
    if semantics == "dependencies_only" and orders:
        raise InputError(f"{sid}: serialization_orders require fixed_schedule semantics")
    for index, raw_order in enumerate(orders):
        order = string_list(raw_order, f"{sid}.serialization_orders[{index}]")
        if any(nid not in nodes for nid in order):
            raise InputError(f"{sid}: serialization order references a missing node")
        for first, second in zip(order, order[1:]):
            nodes[second]["after"].add(first)

    # Topological pass both detects cycles and computes transitive ancestry.
    pending = set(nodes)
    ancestors: dict[str, set[str]] = {}
    finish: dict[str, float] = {}
    paths: dict[str, list[str]] = {}
    detail = []
    while pending:
        ready = sorted(n for n in pending if nodes[n]["after"] <= finish.keys())
        if not ready:
            raise InputError(f"{sid}: graph contains a cycle")
        for nid in ready:
            node = nodes[nid]
            parents = node["after"]
            ancestors[nid] = set(parents)
            for parent in parents:
                ancestors[nid].update(ancestors[parent])
            critical_parent = max(sorted(parents), key=lambda p: finish[p]) if parents else None
            start = finish[critical_parent] if critical_parent else 0.0
            duration = max(node["times"].values())
            finish[nid] = number(start + duration, f"{sid}.{nid}.computed_finish_ms")
            paths[nid] = (paths[critical_parent] if critical_parent else []) + [nid]
            bottlenecks = sorted(r for r, t in node["times"].items() if math.isclose(t, duration))
            detail.append({"id": nid, "resource_times_ms": node["times"],
                           "node_time_ms": duration, "limiting_resources_in_model": bottlenecks,
                           "earliest_start_in_model_ms": start, "earliest_finish_in_model_ms": finish[nid]})
            pending.remove(nid)

    unordered_shared = []
    ids = sorted(nodes)
    for index, first in enumerate(ids):
        for second in ids[index + 1:]:
            if first in ancestors[second] or second in ancestors[first]:
                continue
            shared = sorted(r for r in nodes[first]["amounts"]
                            if nodes[first]["amounts"][r] > 0 and nodes[second]["amounts"].get(r, 0) > 0)
            if shared:
                unordered_shared.append({"nodes": [first, second], "resources": shared})
    if semantics == "fixed_schedule" and unordered_shared:
        raise InputError(f"{sid}: fixed_schedule has unordered users of shared resources; serialize them or use dependencies_only")

    terminal = max(sorted(finish), key=lambda n: finish[n])
    path_time = finish[terminal]
    aggregate_times = {r: number(amount / resources[r]["rate"] * 1000, f"{sid}.{r}.aggregate_ms")
                       for r, amount in resource_totals.items()}
    aggregate_time = max(aggregate_times.values())
    memory = obj(scenario.get("memory"), f"{sid}.memory")
    required = number(memory.get("required_peak_bytes"), f"{sid}.memory.required_peak_bytes")
    available = number(memory.get("available_bytes"), f"{sid}.memory.available_bytes", positive=True)
    memory_note = label(memory.get("source"), f"{sid}.memory.source")
    if mode == "bound":
        value_kind = "fixed_schedule_lower_bound" if semantics == "fixed_schedule" else "resource_dependency_lower_bound"
        time_value = max(path_time, aggregate_time)
    else:
        value_kind = "fixed_schedule_model_estimate" if semantics == "fixed_schedule" else "dependency_model_estimate"
        time_value = path_time
    return {
        "id": sid, "graph_semantics": semantics, "value_kind": value_kind,
        "time_ms": time_value, "critical_path": paths[terminal], "critical_path_time_ms": path_time,
        "aggregate_resource_times_ms": aggregate_times,
        "unordered_shared_resources": unordered_shared,
        "capacity_feasible_from_supplied_values": required <= available,
        "memory_headroom_bytes": available - required, "memory_source": memory_note,
        "nodes": detail,
    }


def analyze(data: Any) -> dict[str, Any]:
    data = obj(data, "input")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise InputError("schema_version must be 1")
    mode = data.get("mode")
    if mode not in ("bound", "forecast"):
        raise InputError("mode must be bound or forecast")
    task = label(data.get("task_signature"), "task_signature")
    origin = data.get("input_origin")
    if origin not in ("synthetic", "literature", "user_report", "measured"):
        raise InputError("input_origin must identify synthetic/literature/user_report/measured inputs")
    resources = _rates(data, mode)
    scenarios = data.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise InputError("scenarios: expected a nonempty list")
    results = [_scenario(s, resources, mode) for s in scenarios]
    by_id = {r["id"]: r for r in results}
    if len(by_id) != len(results):
        raise InputError("duplicate scenario ids")
    baseline_id = label(data.get("baseline_id"), "baseline_id")
    if baseline_id not in by_id:
        raise InputError("baseline_id must identify a scenario")
    baseline = by_id[baseline_id]
    for result in results:
        comparable = (baseline["capacity_feasible_from_supplied_values"]
                      and result["capacity_feasible_from_supplied_values"]
                      and result["time_ms"] > 0
                      and baseline["graph_semantics"] == result["graph_semantics"])
        ratio_key = "bound_ratio_not_speedup" if mode == "bound" else "model_time_ratio_not_measured_speedup"
        result[ratio_key] = (finite_derived(baseline["time_ms"] / result["time_ms"],
                                          f"{result['id']}.{ratio_key}") if comparable else None)

    observations = data.get("observations", [])
    if not isinstance(observations, list):
        raise InputError("observations: expected a list")
    comparisons = []
    for index, raw_observation in enumerate(observations):
        obs = obj(raw_observation, f"observations[{index}]")
        sid = label(obs.get("scenario_id"), "observation.scenario_id")
        if sid not in by_id:
            raise InputError("observation references a missing scenario")
        observed = number(obs.get("duration_ms"), "observation.duration_ms", positive=True)
        split = obs.get("split")
        if split not in ("calibration", "holdout"):
            raise InputError("observation.split must be calibration or holdout")
        obs_origin = obs.get("origin")
        if obs_origin not in ("synthetic", "literature", "user_report", "measured"):
            raise InputError("observation.origin must identify its evidence")
        item = {"scenario_id": sid, "split": split, "origin": obs_origin,
                "artifact_ref": label(obs.get("artifact_ref"), "observation.artifact_ref")}
        value = by_id[sid]["time_ms"]
        if mode == "forecast":
            item["signed_relative_error"] = finite_derived(
                (value - observed) / observed, f"observations[{index}].signed_relative_error")
        else:
            item["observation_below_supplied_bound"] = observed < value - 1e-9
        comparisons.append(item)
    return {
        "schema_version": 1, "evidence_status": "simulated" if origin == "synthetic" else "derived",
        "input_origin": origin, "mode": mode, "task_signature": task,
        "calculator_version": "1.0", "resource_parameters": data["resources"],
        "baseline_id": baseline_id, "scenarios": results, "observation_checks": comparisons,
        "limitations": [
            "No GPU workload was executed; supplied origins and resource ceilings are not independently verified.",
            "Bounds condition on supplied work, resource scope and graph; effective rates produce estimates only.",
            "Fixed schedules serialize shared-resource users; arbitrary overlap and interference are unsupported.",
            "Memory peaks are supplied, not derived; TTFT/TPOT, queueing and tail latencies are not predicted.",
            "Ratios of two lower bounds are not actual speedups or speedup upper bounds.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(json.loads(args.input.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, InputError) as error:
        parser.exit(2, f"Input error: {error}\n")
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
