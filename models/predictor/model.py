"""Profile-calibrated, deliberately bounded inference benefit predictor.

Only Python's standard library is needed. All time values are milliseconds.
This is a scenario calculator, not a trained model or a latency simulator.
"""

from __future__ import annotations

import math
from typing import Any


STAGES = ("prefill", "first_decode", "decode")
EFFECT_FIELDS = {
    "stage", "block", "serial_speedup", "compute_speedup", "comm_speedup",
    "overlap_fraction", "added_ms",
}


class InputError(ValueError):
    """The scenario lacks a meaningful or internally consistent input."""


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputError(f"{path} must be an object")
    return value


def _number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise InputError(f"{path} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        word = "positive" if positive else "nonnegative"
        raise InputError(f"{path} must be finite and {word}")
    return number


def _fraction(value: Any, path: str) -> float:
    number = _number(value, path)
    if number > 1:
        raise InputError(f"{path} must be in [0, 1]")
    return number


def _block_values(block: dict[str, Any], stage: str, index: int) -> dict[str, float | str]:
    path = f"profile.{stage}[{index}]"
    name = block.get("name")
    if not isinstance(name, str) or not name.strip():
        raise InputError(f"{path}.name must be a nonempty string")
    fields = {
        key: _number(block.get(key, 0), f"{path}.{key}")
        for key in ("serial_ms", "compute_ms", "comm_ms", "compute_floor_ms", "comm_floor_ms")
    }
    fields["overlap_fraction"] = _fraction(block.get("overlap_fraction", 1), f"{path}.overlap_fraction")
    for lane in ("compute", "comm"):
        if fields[f"{lane}_floor_ms"] > fields[f"{lane}_ms"] + 1e-9:
            raise InputError(f"{path}.{lane}_floor_ms exceeds the profiled lane time")
    if sum(fields[k] for k in ("serial_ms", "compute_ms", "comm_ms")) == 0:
        raise InputError(f"{path} has no measured cost")
    return {"name": name, **fields}


def _duration(b: dict[str, Any], e: dict[str, Any] | None = None) -> tuple[float, list[str]]:
    e = e or {}
    serial = b["serial_ms"] / _number(e.get("serial_speedup", 1), "effect.serial_speedup", positive=True)
    compute_requested = b["compute_ms"] / _number(
        e.get("compute_speedup", 1), "effect.compute_speedup", positive=True
    )
    comm_requested = b["comm_ms"] / _number(e.get("comm_speedup", 1), "effect.comm_speedup", positive=True)
    compute = max(compute_requested, b["compute_floor_ms"])
    comm = max(comm_requested, b["comm_floor_ms"])
    overlap = _fraction(e.get("overlap_fraction", b["overlap_fraction"]), "effect.overlap_fraction")
    added = _number(e.get("added_ms", 0), "effect.added_ms")
    warnings = []
    if compute > compute_requested + 1e-8:
        warnings.append("计算 lane 已触及所提供的性能下界")
    if comm > comm_requested + 1e-8:
        warnings.append("通信 lane 已触及所提供的性能下界")
    # overlap=0: serial sum. overlap=1: fully overlapped max(compute, comm).
    return serial + compute + comm - overlap * min(compute, comm) + added, warnings


def expected_speculative_tokens(draft_tokens: int, acceptance: float) -> float:
    """1 + p + ... + p**draft_tokens, assuming constant conditional acceptance."""
    if isinstance(draft_tokens, bool) or not isinstance(draft_tokens, int) or not 1 <= draft_tokens <= 32:
        raise InputError("speculation.draft_tokens must be an integer in [1, 32]")
    p = _fraction(acceptance, "speculation.acceptance_probability")
    return sum(p ** i for i in range(draft_tokens + 1))


def _validate_profile(data: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    profile = _object(data.get("profile"), "profile")
    blocks: dict[str, list[dict[str, Any]]] = {}
    totals: dict[str, float] = {}
    if set(profile) != set(STAGES):
        raise InputError("profile must contain exactly prefill, first_decode and decode")
    for stage in STAGES:
        items = profile[stage]
        if not isinstance(items, list) or not items:
            raise InputError(f"profile.{stage} must be a nonempty list")
        blocks[stage] = [_block_values(_object(v, f"profile.{stage}[{i}]"), stage, i) for i, v in enumerate(items)]
        names = [b["name"] for b in blocks[stage]]
        if len(set(names)) != len(names):
            raise InputError(f"profile.{stage} block names must be unique")
        totals[stage] = sum(_duration(block)[0] for block in blocks[stage])
        if totals[stage] <= 0:
            raise InputError(f"profile.{stage} must have a positive duration")
    return blocks, totals


def _parse_effects(
    scenario: dict[str, Any], blocks: dict[str, list[dict[str, Any]]]
) -> dict[tuple[str, str], dict[str, Any]]:
    items = scenario.get("effects", [])
    if not isinstance(items, list):
        raise InputError("scenario.effects must be a list")
    known = {(stage, b["name"]) for stage in STAGES for b in blocks[stage]}
    effects: dict[tuple[str, str], dict[str, Any]] = {}
    for i, item in enumerate(items):
        e = _object(item, f"effects[{i}]")
        unknown = set(e) - EFFECT_FIELDS
        if unknown:
            raise InputError(f"effects[{i}] unknown keys: {sorted(unknown)}")
        key = (e.get("stage"), e.get("block"))
        if key not in known:
            raise InputError(f"effects[{i}] targets a missing stage/block: {key}")
        if key in effects:
            raise InputError(f"effects[{i}] duplicates {key}; combine changes in one effect")
        effects[key] = e
    return effects


def _capacity(
    baseline: dict[str, Any], scenario: dict[str, Any]
) -> tuple[float | None, str | None, float | None, float | None]:
    factors = scenario.get("resource_demand_factors")
    if factors is None:
        return None, None, None, None
    factors = _object(factors, "resource_demand_factors")
    utilization = _object(baseline.get("resource_busy_fraction"), "baseline.resource_busy_fraction")
    if not utilization:
        raise InputError("resource_busy_fraction must be nonempty for capacity prediction")
    if set(factors) != set(utilization):
        raise InputError("resource_demand_factors must specify every measured resource")
    u = {k: _fraction(v, f"baseline.resource_busy_fraction.{k}") for k, v in utilization.items()}
    if max(u.values()) <= 0:
        raise InputError("resource_busy_fraction must include an active resource")
    f = {k: _number(v, f"resource_demand_factors.{k}", positive=True) for k, v in factors.items()}
    pressure = {k: u[k] * f[k] for k in u}
    new_bottleneck = max(pressure, key=pressure.get)
    capacity = _number(baseline.get("saturated_output_tps"), "baseline.saturated_output_tps", positive=True)
    # Relative resource-demand proxy normalized to reproduce the saturated baseline.
    predicted_capacity = capacity * max(u.values()) / max(pressure.values())
    arrival = baseline.get("arrival_rps")
    if arrival is None:
        return predicted_capacity, new_bottleneck, None, None
    offered = _number(arrival, "baseline.arrival_rps") * _number(
        baseline.get("avg_output_tokens"), "baseline.avg_output_tokens", positive=True
    )
    return predicted_capacity, new_bottleneck, min(offered, predicted_capacity), offered / predicted_capacity


def _speculation(
    scenario: dict[str, Any], baseline: dict[str, Any], normal_decode_ms: float
) -> tuple[float, float | None]:
    spec = scenario.get("speculation")
    if spec is None:
        return normal_decode_ms, None
    spec = _object(spec, "speculation")
    if baseline.get("decode_tokens_per_step") != 1:
        raise InputError("speculation requires a baseline with decode_tokens_per_step=1")
    tokens = expected_speculative_tokens(
        spec.get("draft_tokens"), spec.get("acceptance_probability")
    )
    verify_multiplier = _number(
        spec.get("verification_multiplier"), "speculation.verification_multiplier", positive=True
    )
    overhead = _number(spec.get("extra_per_round_ms"), "speculation.extra_per_round_ms")
    return (normal_decode_ms * verify_multiplier + overhead) / tokens, tokens


def _observations(scenario: dict[str, Any], predicted: dict[str, Any]) -> dict[str, float]:
    measured = scenario.get("observed", {})
    if not isinstance(measured, dict):
        raise InputError("scenario.observed must be an object")
    result = {}
    for metric in ("ttft_mean_ms", "tpot_mean_ms", "saturated_output_tps"):
        if metric not in measured:
            continue
        actual = _number(measured[metric], f"observed.{metric}", positive=True)
        estimate = predicted[metric]
        if estimate is not None:
            result[f"{metric}_error_pct"] = 100 * (estimate - actual) / actual
    return result


def predict(data: dict[str, Any]) -> dict[str, Any]:
    """Return baseline-reproducing critical-path and capacity scenarios.

    Does not predict p95/p99, loaded TTFT, scheduling or prefix-cache hit rates.
    """
    data = _object(data, "root")
    if data.get("schema_version") != 1:
        raise InputError("schema_version must be 1")
    data_kind = data.get("data_kind")
    if data_kind not in ("synthetic", "measured"):
        raise InputError("data_kind must be synthetic or measured")
    baseline = _object(data.get("baseline"), "baseline")
    ttft0 = _number(baseline.get("ttft_mean_ms"), "baseline.ttft_mean_ms", positive=True)
    tpot0 = _number(baseline.get("tpot_mean_ms"), "baseline.tpot_mean_ms", positive=True)
    cap0 = _number(baseline.get("saturated_output_tps"), "baseline.saturated_output_tps", positive=True)
    measured = _object(baseline.get("stage_mean_ms"), "baseline.stage_mean_ms")
    if set(measured) != {"prefill", "first_decode"}:
        raise InputError("baseline.stage_mean_ms requires exactly prefill and first_decode")
    measured_stage = {
        "prefill": _number(measured["prefill"], "stage_mean_ms.prefill", positive=True),
        "first_decode": _number(measured["first_decode"], "stage_mean_ms.first_decode", positive=True),
        "decode": tpot0,
    }
    fixed_ttft = ttft0 - measured_stage["prefill"] - measured_stage["first_decode"]
    if fixed_ttft < -1e-7:
        raise InputError("TTFT is smaller than measured prefill + first_decode")
    fixed_ttft = max(0.0, fixed_ttft)
    blocks, base_model_stage = _validate_profile(data)
    calibration = {k: measured_stage[k] / base_model_stage[k] for k in STAGES}
    scenarios = data.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise InputError("scenarios must be a nonempty list")
    output = []
    seen_names = set()
    for i, obj in enumerate(scenarios):
        sc = _object(obj, f"scenarios[{i}]")
        name = sc.get("name")
        if not isinstance(name, str) or not name.strip() or name in seen_names:
            raise InputError("scenario names must be nonempty and unique")
        seen_names.add(name)
        effects = _parse_effects(sc, blocks)
        stage_costs = {}
        floor_hits = []
        for stage in STAGES:
            uncalibrated_without_new_overhead = 0.0
            exposed_new_overhead = 0.0
            for b in blocks[stage]:
                effect = effects.get((stage, b["name"]))
                cost, hit = _duration(b, {k: v for k, v in (effect or {}).items() if k != "added_ms"})
                uncalibrated_without_new_overhead += cost
                if effect:
                    exposed_new_overhead += _number(effect.get("added_ms", 0), "effect.added_ms")
                floor_hits += [f"{stage}/{b['name']}: {msg}" for msg in hit]
            # New exposed overhead is already measured in milliseconds; do not rescale it.
            stage_costs[stage] = uncalibrated_without_new_overhead * calibration[stage] + exposed_new_overhead
        candidate_normal_decode = stage_costs["decode"]
        tpot, tokens_per_round = _speculation(sc, baseline, candidate_normal_decode)
        predicted = {
            "name": name,
            "assumption": sc.get("assumption", "unspecified"),
            "ttft_mean_ms": fixed_ttft + stage_costs["prefill"] + stage_costs["first_decode"],
            "tpot_mean_ms": tpot,
            "saturated_output_tps": None,
            "offered_output_tps": None,
            "utilization_ratio": None,
            "capacity_bottleneck": None,
            "expected_tokens_per_verification": tokens_per_round,
            "stage_mean_ms": stage_costs,
            "floor_hits": floor_hits,
            "warnings": [],
        }
        capacity, bottleneck, served, rho = _capacity(baseline, sc)
        predicted["saturated_output_tps"] = capacity
        predicted["offered_output_tps"] = served
        predicted["utilization_ratio"] = rho
        predicted["capacity_bottleneck"] = bottleneck
        predicted["ttft_speedup"] = ttft0 / predicted["ttft_mean_ms"]
        predicted["tpot_speedup"] = tpot0 / tpot
        predicted["capacity_speedup"] = (capacity / cap0) if capacity else None
        if capacity is None:
            predicted["warnings"].append("缺少资源需求变化因子，暂不预测饱和吞吐容量")
        if rho is not None and rho >= 1:
            predicted["warnings"].append("到达负载达到或超过预测容量，队列可能持续增长")
        elif rho is not None and rho >= 0.85:
            predicted["warnings"].append("接近饱和：排队 TTFT 和尾延迟不在本模型范围")
        if tokens_per_round is not None:
            predicted["warnings"].append("MTP 对首 token 和批处理的影响未计入 TTFT")
        predicted["observed_error_pct"] = _observations(sc, predicted)
        output.append(predicted)
    return {
        "schema_version": 1,
        "data_kind": data_kind,
        "baseline_id": baseline.get("id", "unregistered"),
        "baseline": {
            "ttft_mean_ms": ttft0,
            "tpot_mean_ms": tpot0,
            "saturated_output_tps": cap0,
            "fixed_ttft_mean_ms": fixed_ttft,
        },
        "stage_profile_model_ms": base_model_stage,
        "stage_calibration_factor": calibration,
        "scenarios": output,
        "model_scope": [
            "TTFT/TPOT 仅预测低负载均值，不预测 p50/p95/p99 与线上排队延迟",
            "吞吐使用饱和资源需求代理，不从 TPOT 倒数推断",
            "缓存命中、batch 变化、跨卡拓扑及生成质量需要外部测量",
            "任何干预在固定 workload / GPU 预算的 A/B 验证前均属假设",
        ],
    }
