#!/usr/bin/env python3
"""Trace-calibrated prefix reuse and prompt compression scenario ceilings.

Scope: chronological exact-key FULL-prefix objects, one sequential LRU cache.
This does not emulate vLLM's block trie, hybrid state groups, batching or p99.
Only stdlib is required. All reported numbers are conditional on input evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


class InputError(ValueError):
    pass


def number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{label} must be a finite number")
    x = float(value)
    if not math.isfinite(x) or (x <= 0 if positive else x < 0):
        raise InputError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return x


def integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0):
        raise InputError(f"{label} must be an integer >= {1 if positive else 0}")
    return value


def fraction(value: Any, label: str) -> float:
    x = number(value, label)
    if x > 1:
        raise InputError(f"{label} must be <= 1")
    return x


def object_(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputError(f"{label} must be an object")
    return value


def parse_variant(value: Any, label: str, *, compressed: bool) -> dict[str, Any]:
    obj = object_(value, label)
    tokens = integer(obj.get("prompt_tokens"), f"{label}.prompt_tokens", positive=True)
    prefix_tokens = integer(obj.get("prefix_tokens"), f"{label}.prefix_tokens")
    if prefix_tokens > tokens:
        raise InputError(f"{label}.prefix_tokens exceeds prompt_tokens")
    key = obj.get("prefix_key")
    if prefix_tokens:
        if not isinstance(key, str) or not key.strip():
            raise InputError(f"{label}.prefix_key must be nonempty for reusable prefix")
    elif key is not None:
        raise InputError(f"{label}.prefix_key must be null for zero reusable tokens")
    state_mib = number(obj.get("state_mib"), f"{label}.state_mib")
    if (prefix_tokens > 0) != (state_mib > 0):
        raise InputError(f"{label}.state_mib must be positive exactly when prefix_tokens > 0")
    cold = number(obj.get("cold_prefill_ms"), f"{label}.cold_prefill_ms")
    hit = number(obj.get("hit_prefill_ms"), f"{label}.hit_prefill_ms")
    parsed: dict[str, Any] = {
        "prompt_tokens": tokens,
        "prefix_tokens": prefix_tokens,
        "prefix_key": key,
        "state_mib": state_mib,
        "cold_prefill_ms": cold,
        "hit_prefill_ms": hit,
    }
    if compressed:
        parsed["compressor_ms"] = number(obj.get("compressor_ms"), f"{label}.compressor_ms")
        quality = obj.get("quality_pass")
        if quality is not None and not isinstance(quality, bool):
            raise InputError(f"{label}.quality_pass must be bool or null")
        parsed["quality_pass"] = quality
    return parsed


def check_prefix_identity(rows: list[dict[str, Any]], label: str) -> None:
    identity = {}
    for i, row in enumerate(rows):
        key = row["prefix_key"]
        if key is None:
            continue
        spec = (row["prefix_tokens"], row["state_mib"])
        if key in identity and identity[key] != spec:
            raise InputError(f"{label}[{i}] reuses a key with different prefix size/state")
        identity[key] = spec


def replay(
    rows: list[dict[str, Any]], *,
    capacity_mib: float,
    lookup_ms: float,
    restore_ms_per_mib: float,
    free_hit: bool = False,
) -> dict[str, Any]:
    """Replay exact-key LRU. Use inf capacity for a no-eviction reference.

    A miss becomes cacheable only after its cold prefill completes. All requests
    are serialized in arrival order; no concurrent in-use pinning is simulated.
    """
    cache: OrderedDict[str, float] = OrderedDict()
    used = 0.0
    peak = 0.0
    hit_requests = hit_tokens = evictions = 0
    total_ms = total_cold_ms = total_prompt_tokens = 0.0
    potential_reuse_ms = realized_reuse_ms = 0.0
    hits_by_key = {}
    for row in rows:
        key = row["prefix_key"]
        size = row["state_mib"]
        total_cold_ms += row["cold_prefill_ms"]
        total_prompt_tokens += row["prompt_tokens"]
        eligible_ms = max(0.0, row["cold_prefill_ms"] - row["hit_prefill_ms"])
        potential_reuse_ms += eligible_ms
        hit = key is not None and key in cache
        if hit:
            hit_requests += 1
            hit_tokens += row["prefix_tokens"]
            hits_by_key[key] = hits_by_key.get(key, 0) + 1
            realized_reuse_ms += eligible_ms
            cache.move_to_end(key)
            total_ms += (
                0.0 if free_hit else
                row["hit_prefill_ms"] + lookup_ms + size * restore_ms_per_mib
            )
        else:
            total_ms += row["cold_prefill_ms"] + (0.0 if free_hit else lookup_ms)
            if key is not None and size <= capacity_mib:
                while used + size > capacity_mib + 1e-9 and cache:
                    _, released = cache.popitem(last=False)
                    used -= released
                    evictions += 1
                cache[key] = size
                used += size
                peak = max(peak, used)
    n = len(rows)
    return {
        "request_hit_rate": hit_requests / n,
        "cached_token_fraction": hit_tokens / total_prompt_tokens,
        "reusable_compute_hit_fraction": (
            realized_reuse_ms / potential_reuse_ms if potential_reuse_ms else 0.0
        ),
        "hit_requests": hit_requests,
        "reused_prefix_tokens": hit_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "mean_prefill_path_ms": total_ms / n,
        "mean_cold_prefill_ms": total_cold_ms / n,
        "gross_saved_prefill_ms": total_cold_ms - total_ms,
        "evictions": evictions,
        "peak_state_mib": peak,
        "ending_state_mib": used,
        "hits_by_key": hits_by_key,
    }


def predict(data: dict[str, Any]) -> dict[str, Any]:
    data = object_(data, "root")
    if data.get("schema_version") != 1 or data.get("data_kind") not in ("synthetic", "measured"):
        raise InputError("schema_version=1 and data_kind=synthetic|measured required")
    config = object_(data.get("config"), "config")
    fixed = number(config.get("fixed_ttft_ms"), "config.fixed_ttft_ms", positive=True)
    capacity = number(config.get("cache_capacity_mib"), "config.cache_capacity_mib")
    lookup = number(config.get("lookup_ms"), "config.lookup_ms")
    restore_rate = number(config.get("restore_ms_per_mib"), "config.restore_ms_per_mib")
    min_quality = fraction(config.get("minimum_quality_pass_rate", 1), "config.minimum_quality_pass_rate")
    requests = data.get("requests")
    if not isinstance(requests, list) or not requests:
        raise InputError("requests must be a nonempty chronological list")
    baseline_rows = []
    compressed_rows = []
    ids = set()
    compressed_count = 0
    for i, item in enumerate(requests):
        req = object_(item, f"requests[{i}]")
        req_id = req.get("id")
        if not isinstance(req_id, str) or not req_id.strip() or req_id in ids:
            raise InputError(f"requests[{i}].id must be unique and nonempty")
        ids.add(req_id)
        baseline_rows.append(parse_variant(req.get("baseline"), f"requests[{i}].baseline", compressed=False))
        if "compressed" in req:
            compressed_count += 1
            compressed_rows.append(parse_variant(req["compressed"], f"requests[{i}].compressed", compressed=True))
    if compressed_count not in (0, len(requests)):
        raise InputError("compressed variants must be supplied for every request or none")
    check_prefix_identity(baseline_rows, "baseline")
    if compressed_rows:
        check_prefix_identity(compressed_rows, "compressed")

    n = len(baseline_rows)
    baseline_prefill = sum(r["cold_prefill_ms"] for r in baseline_rows) / n
    baseline_ttft = fixed + baseline_prefill
    configurations = {
        "no_cache": None,
        "lru_budget": capacity,
        "unbounded_exact_key": float("inf"),
        "ideal_free_hit_exact_key": float("inf"),
    }
    scenarios = {}
    for name, budget in configurations.items():
        if budget is None:
            result = {
                "mean_prefill_path_ms": baseline_prefill,
                "request_hit_rate": 0.0,
                "cached_token_fraction": 0.0,
                "reusable_compute_hit_fraction": 0.0,
                "hit_requests": 0,
                "evictions": 0,
                "peak_state_mib": 0.0,
            }
        else:
            result = replay(
                baseline_rows, capacity_mib=budget, lookup_ms=lookup,
                restore_ms_per_mib=restore_rate,
                free_hit=(name == "ideal_free_hit_exact_key"),
            )
        result["ttft_mean_ms"] = fixed + result["mean_prefill_path_ms"]
        result["ttft_speedup"] = baseline_ttft / result["ttft_mean_ms"]
        scenarios[name] = result
    # Omitting ALL prefill is an Amdahl ceiling, not an implementable cache strategy.
    absolute_ceiling = baseline_ttft / fixed

    compression = None
    if compressed_rows:
        compressor_mean = sum(r["compressor_ms"] for r in compressed_rows) / n
        cold_mean = sum(r["cold_prefill_ms"] for r in compressed_rows) / n
        total_prompt_orig = sum(r["prompt_tokens"] for r in baseline_rows)
        total_prompt_new = sum(r["prompt_tokens"] for r in compressed_rows)
        quality = [r["quality_pass"] for r in compressed_rows]
        pass_rate = sum(quality) / n if all(isinstance(q, bool) for q in quality) else None
        quality_ok = pass_rate >= min_quality if pass_rate is not None else None
        results = {}
        for name, budget in {
            "compressed_no_cache": None,
            "compressed_lru_budget": capacity,
            "compressed_unbounded_exact_key": float("inf"),
        }.items():
            if budget is None:
                result = {"mean_prefill_path_ms": cold_mean, "request_hit_rate": 0.0,
                          "cached_token_fraction": 0.0, "evictions": 0}
            else:
                result = replay(
                    compressed_rows, capacity_mib=budget, lookup_ms=lookup,
                    restore_ms_per_mib=restore_rate,
                )
            ttft = fixed + compressor_mean + result["mean_prefill_path_ms"]
            result["ttft_mean_ms"] = ttft
            result["conditional_ttft_speedup"] = baseline_ttft / ttft
            result["relative_to_uncompressed_lru_speedup"] = (
                scenarios["lru_budget"]["ttft_mean_ms"] / ttft
            )
            result["qualified_ttft_speedup"] = (
                result["conditional_ttft_speedup"] if quality_ok is True else None
            )
            results[name] = result
        compression = {
            "input_token_reduction": 1 - total_prompt_new / total_prompt_orig,
            "mean_compressor_ms": compressor_mean,
            "mean_compressed_cold_prefill_ms": cold_mean,
            "prefill_only_break_even_ms": baseline_prefill - cold_mean,
            "compressor_break_even": compressor_mean < baseline_prefill - cold_mean,
            "quality_pass_rate": pass_rate,
            "required_quality_pass_rate": min_quality,
            "quality_gate_ok": quality_ok,
            "results": results,
            "warning": (
                "Context compression changes token IDs and may destroy exact prefix reuse; "
                "combined gains use explicit compressed keys, not summed speedups."
            ),
        }
    return {
        "schema_version": 1,
        "data_kind": data["data_kind"],
        "scope": "sequential exact-key full-prefix trace; not a block-level cache simulator",
        "request_count": n,
        "baseline_ttft_mean_ms": baseline_ttft,
        "fixed_ttft_mean_ms": fixed,
        "absolute_all_prefill_removed_speedup": absolute_ceiling,
        "cache_capacity_mib": capacity,
        "cache": scenarios,
        "compression": compression,
        "not_predicted": [
            "p95/p99, queueing or concurrent prefix pressure",
            "GPU L2 cache hit rate or block-level partial-prefix reuse",
            "saturated output TPS / goodput without separate resource-demand measurements",
            "Qwen hybrid-state correctness, multi-tenant isolation and score parity",
        ],
    }


def markdown(result: dict[str, Any]) -> str:
    label = "合成演示 / 非 H800 实测" if result["data_kind"] == "synthetic" else "用户标注的测量输入"
    lines = [
        f"# Prefix Cache / Context Compression 天花板情景（{label}）",
        "",
        f"请求数：{result['request_count']}；基线 TTFT 均值：{result['baseline_ttft_mean_ms']:.2f} ms。",
        f"全部 Prefill 凭空消失的 Amdahl **极松上界**：{result['absolute_all_prefill_removed_speedup']:.3f}×，不能作为工程预测。",
        "",
        "| 缓存策略 | 请求命中率 | 命中输入 Token 比例 | 命中可省 Prefill 比例 | TTFT 均值 ms | 相对无缓存加速 | 淘汰次数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, v in result["cache"].items():
        lines.append(
            f"| {name} | {v['request_hit_rate']:.1%} | {v['cached_token_fraction']:.1%} | "
            f"{v['reusable_compute_hit_fraction']:.1%} | "
            f"{v['ttft_mean_ms']:.2f} | {v['ttft_speedup']:.3f}× | {v['evictions']} |"
        )
    if result["compression"] is not None:
        c = result["compression"]
        gate = "待验证" if c["quality_gate_ok"] is None else ("通过" if c["quality_gate_ok"] else "未通过")
        lines += [
            "", f"## 上下文压缩（质量门槛：{gate}）", "",
            f"输入 Token 缩减：{c['input_token_reduction']:.1%}；平均压缩开销：{c['mean_compressor_ms']:.2f} ms。",
            f"仅算 Prefill 的压缩盈亏阈值：压缩器耗时 < {c['prefill_only_break_even_ms']:.2f} ms；"
            f"当前情景：{'满足' if c['compressor_break_even'] else '不满足'}。",
            "",
            "| 压缩情景 | 请求命中率 | TTFT 均值 ms | vs 无缓存原文 | vs 原文 LRU | 达质量门槛可采纳加速 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, v in c["results"].items():
            approved = (
                f"{v['qualified_ttft_speedup']:.3f}×"
                if v["qualified_ttft_speedup"] is not None else "未获质量证明"
            )
            lines.append(
                f"| {name} | {v['request_hit_rate']:.1%} | {v['ttft_mean_ms']:.2f} | "
                f"{v['conditional_ttft_speedup']:.3f}× | "
                f"{v['relative_to_uncompressed_lru_speedup']:.3f}× | {approved} |"
            )
    lines += [
        "",
        "**注意**：无限缓存只在同一精确前缀键模型内构成无淘汰参照；不是块级部分前缀复用的全局最优上界。",
        "压缩与缓存不能直接把收益相加；压缩后的 prefix_key 必须单独测量。所有参数若来自合成示例，均不能宣称 H800 加速。",
        "本模型不输出饱和输出 tok/s；该指标必须回到资源容量模型进行校准。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefix-cache LRU trace and context compression scenarios")
    parser.add_argument("input", type=Path)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = predict(json.loads(args.input.read_text(encoding="utf-8")))
    except (InputError, OSError, ValueError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2
    body = json.dumps(result, ensure_ascii=False, indent=2) + "\n" if args.format == "json" else markdown(result)
    if args.output:
        args.output.write_text(body, encoding="utf-8")
    else:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
