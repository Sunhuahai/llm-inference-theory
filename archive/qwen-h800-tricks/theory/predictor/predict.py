#!/usr/bin/env python3
"""CLI: python3 predict.py examples/synthetic-baseline.json --format markdown"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model import InputError, predict


def _cell(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def as_markdown(result: dict) -> str:
    kind = "合成演示数据，非 H800 实测" if result["data_kind"] == "synthetic" else "用户提供的测量数据"
    baseline_label = "合成基线" if result["data_kind"] == "synthetic" else "基线测量"
    baseline = result["baseline"]
    lines = [
        f"# 优化收益情景预测（{kind}）",
        "",
        f"基线：{result['baseline_id']}；TTFT/TPOT 预测为低负载均值，不是 p95。",
        "",
        "| 情景 | TTFT ms | TTFT 加速 | TPOT ms | TPOT 加速 | 饱和输出 tok/s | 容量加速 | 新瓶颈 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        f"| baseline | {_cell(baseline['ttft_mean_ms'])} | 1.00× | "
        f"{_cell(baseline['tpot_mean_ms'])} | 1.00× | "
        f"{_cell(baseline['saturated_output_tps'])} | 1.00× | {baseline_label} |",
    ]
    for s in result["scenarios"]:
        lines.append(
            f"| {s['name']} | {_cell(s['ttft_mean_ms'])} | {s['ttft_speedup']:.3f}× | "
            f"{_cell(s['tpot_mean_ms'])} | {s['tpot_speedup']:.3f}× | "
            f"{_cell(s['saturated_output_tps'])} | "
            f"{_cell(s['capacity_speedup'], 3)}× | {s['capacity_bottleneck'] or '未预测'} |"
        )
    lines += [
        "",
        "## 适用范围",
        "",
        "- 输出吞吐模型要求固定 workload、固定 GPU 数量，以及饱和基线与资源繁忙比例。",
        "- 到达率不足时，实际输出 tok/s 不会因为容量增加而自动增加。",
        "- 分块调度、多请求排队、p95/p99 和 P/D 分离需补充 trace 级仿真。",
        "- 输入参数为情景假设；合成示例不构成 H800 加速结论。",
    ]
    warnings = [(s["name"], w) for s in result["scenarios"] for w in s["warnings"]]
    if warnings:
        lines += ["", "## 预警", ""]
        lines += [f"- {name}：{warning}" for name, warning in warnings]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile-calibrated latency and capacity scenarios")
    parser.add_argument("input", type=Path, help="JSON with baseline, profile and scenarios")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", type=Path, help="optional output path (otherwise stdout)")
    args = parser.parse_args()
    try:
        source = json.loads(args.input.read_text(encoding="utf-8"))
        result = predict(source)
    except (OSError, ValueError, InputError) as exc:
        print(f"Prediction input error: {exc}", file=sys.stderr)
        return 2
    data = (
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.format == "json"
        else as_markdown(result)
    )
    if args.output:
        args.output.write_text(data, encoding="utf-8")
    else:
        sys.stdout.write(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
