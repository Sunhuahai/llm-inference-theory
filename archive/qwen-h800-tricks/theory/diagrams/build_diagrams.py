#!/usr/bin/env python3
"""Build three dense, professionally titled native draw.io pages and SVGs.

Both formats use the same geometric primitives. The source is deterministic.
"""

from __future__ import annotations

import html
import xml.etree.ElementTree as ET
from pathlib import Path


DIR = Path(__file__).resolve().parent
FILE = DIR / "inference-optimization-map.drawio"
PREVIEWS = DIR / "previews"
W, H = 1920, 1080
INK = "#14243B"
MUTED = "#60718A"
BG = "#F6F8FC"
WHITE = "#FFFFFF"
BLUE = "#2765D8"
BLUE_TINT = "#E8F0FF"
TEAL = "#078E87"
TEAL_TINT = "#E3F5F1"
VIOLET = "#7655C2"
VIOLET_TINT = "#F0EBFA"
ORANGE = "#C9771A"
ORANGE_TINT = "#FFF1DB"
ROSE = "#C64F64"
ROSE_TINT = "#FBEAF0"
GREEN = "#27875B"
GREEN_TINT = "#E5F5EC"
BORDER = "#DCE5F0"
FONT = "Noto Sans CJK SC"
SVG_FONT = "Noto Sans CJK SC, PingFang SC, Microsoft YaHei, sans-serif"


def fmt(value: float | int) -> str:
    return str(int(value)) if float(value).is_integer() else str(round(value, 3))


class Page:
    def __init__(self, slug: str, name: str, number: int, section: str, title: str, subtitle: str):
        self.slug, self.name, self.number = slug, name, number
        self.diagram = ET.Element("diagram", {"id": slug, "name": name})
        self.graph = ET.SubElement(self.diagram, "mxGraphModel", {
            "dx": str(W), "dy": str(H), "grid": "1", "gridSize": "10",
            "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
            "fold": "1", "page": "1", "pageScale": "1",
            "pageWidth": str(W), "pageHeight": str(H), "math": "0",
            "shadow": "0", "background": BG,
        })
        self.root = ET.SubElement(self.graph, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})
        self.svg = []
        self.index = 0
        self.rect(0, 0, W, H, BG, stroke=BG, radius=0)
        self.rect(0, 0, W, 9, BLUE, stroke=BLUE, radius=0)
        self.text(64, 26, 1400, 31, section.upper(), 17, BLUE, bold=True)
        self.text(64, 65, 1740, 58, title, 39, INK, bold=True)
        self.text(64, 136, 1750, 36, subtitle, 19, MUTED)
        self.line(64, 188, 1856, 188, BORDER, 2)
        self.text(64, 1035, 1660, 30,
                  "QWEN · H800    固定 checkpoint / 固定精度 / 固定 GPU 预算    ·    研究模型 ≠ H800 实测",
                  13, MUTED)
        self.text(1790, 1032, 65, 33, f"{number:02d}", 22, BLUE, bold=True, align="right")

    def _id(self):
        self.index += 1
        return f"{self.slug}-{self.index}"

    def rect(self, x, y, w, h, fill=WHITE, *, stroke=BORDER, radius=16, width=1.4):
        style = (f"rounded={1 if radius else 0};arcSize={min(30, round(radius / max(1, min(w,h)) * 100))};"
                 f"whiteSpace=wrap;html=0;fillColor={fill};strokeColor={stroke};strokeWidth={width};")
        cell = ET.SubElement(self.root, "mxCell", {
            "id": self._id(), "parent": "1", "vertex": "1", "style": style,
        })
        ET.SubElement(cell, "mxGeometry", {
            "x": fmt(x), "y": fmt(y), "width": fmt(w), "height": fmt(h), "as": "geometry",
        })
        self.svg.append(
            f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" '
            f'rx="{fmt(radius)}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
        )

    def text(self, x, y, w, h, value, size=18, color=INK, *, bold=False, align="left"):
        a = {"left": "left", "center": "center", "right": "right"}[align]
        style = (f"text;whiteSpace=wrap;html=0;overflow=visible;align={a};"
                 f"verticalAlign=middle;spacing=0;fontFamily={FONT};fontSize={size};"
                 f"fontStyle={1 if bold else 0};fontColor={color};strokeColor=none;fillColor=none;")
        cell = ET.SubElement(self.root, "mxCell", {
            "id": self._id(), "parent": "1", "vertex": "1", "style": style,
            "value": value,
        })
        ET.SubElement(cell, "mxGeometry", {
            "x": fmt(x), "y": fmt(y), "width": fmt(w), "height": fmt(h), "as": "geometry",
        })
        lines = value.split("\n")
        lead = size * 1.45
        total = lead * len(lines)
        yy = y + (h - total) / 2 + size * 1.12
        anchor = {"left": "start", "center": "middle", "right": "end"}[align]
        xx = {"left": x, "center": x + w / 2, "right": x + w}[align]
        self.svg.append(
            f'<text x="{fmt(xx)}" y="{fmt(yy)}" text-anchor="{anchor}" '
            f'font-family="{html.escape(SVG_FONT)}" font-size="{size}" '
            f'font-weight="{700 if bold else 400}" fill="{color}">'
            + "".join(
                f'<tspan x="{fmt(xx)}" dy="{fmt(0 if j == 0 else lead)}">{html.escape(line)}</tspan>'
                for j, line in enumerate(lines)
            ) + "</text>"
        )

    def line(self, x1, y1, x2, y2, color=BORDER, width=2):
        cell = ET.SubElement(self.root, "mxCell", {
            "id": self._id(), "parent": "1", "edge": "1",
            "style": f"endArrow=none;html=0;strokeColor={color};strokeWidth={width};",
        })
        g = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        ET.SubElement(g, "mxPoint", {"x": fmt(x1), "y": fmt(y1), "as": "sourcePoint"})
        ET.SubElement(g, "mxPoint", {"x": fmt(x2), "y": fmt(y2), "as": "targetPoint"})
        self.svg.append(
            f'<line x1="{fmt(x1)}" y1="{fmt(y1)}" x2="{fmt(x2)}" '
            f'y2="{fmt(y2)}" stroke="{color}" stroke-width="{width}"/>'
        )

    def arrow(self, x1, y1, x2, y2, color=BLUE, width=3):
        cell = ET.SubElement(self.root, "mxCell", {
            "id": self._id(), "parent": "1", "edge": "1",
            "style": f"endArrow=block;endFill=1;html=0;strokeColor={color};strokeWidth={width};",
        })
        g = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        ET.SubElement(g, "mxPoint", {"x": fmt(x1), "y": fmt(y1), "as": "sourcePoint"})
        ET.SubElement(g, "mxPoint", {"x": fmt(x2), "y": fmt(y2), "as": "targetPoint"})
        self.svg.append(
            f'<line x1="{fmt(x1)}" y1="{fmt(y1)}" x2="{fmt(x2)}" y2="{fmt(y2)}" '
            f'stroke="{color}" stroke-width="{width}" marker-end="url(#arrow)"/>'
        )

    def chip(self, x, y, w, label, tint=BLUE_TINT, color=BLUE):
        self.rect(x, y, w, 33, tint, stroke=tint, radius=16)
        self.text(x + 10, y + 1, w - 20, 30, label, 14, color, bold=True, align="center")

    def card(self, x, y, w, h, title, body, *, accent=BLUE, tint=BLUE_TINT, tag=None, size=18):
        self.rect(x, y, w, h, WHITE, stroke=BORDER, radius=18)
        self.rect(x, y, 8, h, accent, stroke=accent, radius=0)
        self.text(x + 25, y + 16, w - 48, 38, title, 23, INK, bold=True)
        if tag:
            self.chip(x + w - 126, y + 18, 108, tag, tint, accent)
        self.text(x + 25, y + 60, w - 48, h - 73, body, size, MUTED)

    def svg_file(self, path: Path):
        defs = ('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
                'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
                '<path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>')
        content = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(self.name)}">'
            + defs + "".join(self.svg) + "</svg>\n"
        )
        path.write_text(content, encoding="utf-8")

def page_performance():
    p = Page("performance-decomposition", "01｜性能分解与瓶颈识别", 1,
             "PERFORMANCE MODEL / LATENCY & CAPACITY",
             "推理性能分解与瓶颈识别",
             "TTFT、TPOT 与饱和吞吐使用不同的成本口径；从请求生命周期与资源利用两条路径同时定位瓶颈。")

    p.text(64, 206, 1500, 37, "请求生命周期与关键时间区间", 24, INK, bold=True)
    steps = [
        ("接入 / 调度", "Tokenize、排队、batch 决策", MUTED, "#E9EDF3"),
        ("Prefill", "Prefix 命中 / 上下文压缩 / 状态生成", BLUE, BLUE_TINT),
        ("首 token Decode", "首次 target step；首个有效输出", TEAL, TEAL_TINT),
        ("迭代 Decode", "后续 token、MoE/QSA/GDN/MTP", VIOLET, VIOLET_TINT),
        ("输出与完成", "Streaming、实际生成长度", ORANGE, ORANGE_TINT),
    ]
    for i, (name, detail, color, tint) in enumerate(steps):
        x = 64 + i * 370
        p.rect(x, 254, 312, 118, tint, stroke=tint)
        p.text(x + 14, 267, 284, 47, name, 24, INK, bold=True, align="center")
        p.text(x + 11, 318, 289, 39, detail, 17, color, align="center")
        if i < 4:
            p.arrow(x + 321, 313, x + 361, 313, color)
    p.arrow(88, 400, 1334, 400, BLUE, 4)
    p.text(290, 405, 890, 29, "TTFT：请求发送 → 首个有效生成 token 到达", 19, BLUE, bold=True, align="center")
    p.arrow(1390, 400, 1840, 400, VIOLET, 4)
    p.text(1389, 405, 455, 29, "TPOT：后续 token 平均间隔", 19, VIOLET, bold=True, align="center")

    p.rect(64, 452, 864, 296, WHITE, stroke=BORDER)
    p.text(89, 465, 816, 43, "低负载：关键路径分解", 27, BLUE, bold=True)
    p.text(92, 516, 813, 45,
           "TTFT = Tqueue + Tfront + Tprefill + Tfirst + Tnet", 21, INK, bold=True)
    p.text(92, 564, 802, 43,
           "TPOT = (tlast − tfirst) / (O − 1)  （O > 1）", 21, INK, bold=True)
    p.line(90, 622, 900, 622, BORDER)
    p.text(92, 632, 800, 96,
           "关键路径 block： t = s + c + m − η·min(c,m)\n"
           "s：串行；c：计算；m：通信；η：重叠比例。\n"
           "η=1 时并行段为 max(c,m)，不可重复计费。",
           19, MUTED)

    p.rect(970, 452, 885, 296, WHITE, stroke=BORDER)
    p.text(996, 465, 820, 42, "高负载：资源容量与瓶颈迁移", 27, TEAL, bold=True)
    p.text(997, 510, 825, 44, "GPU Compute / HBM / Interconnect / CPU", 19, MUTED)
    for i, (label, fraction, color) in enumerate([
        ("GPU Compute", .91, BLUE), ("HBM", .70, TEAL),
        ("Interconnect", .47, ORANGE), ("CPU", .60, VIOLET),
    ]):
        y = 556 + 35 * i
        p.text(998, y, 193, 31, label, 18, INK, bold=True)
        p.rect(1215, y + 7, 500, 17, "#EDF2F7", stroke="#EDF2F7", radius=9)
        p.rect(1215, y + 7, 500 * fraction, 17, color, stroke=color, radius=9)
        p.text(1724, y, 93, 31, f"{round(fraction * 100)}%", 18, color, bold=True, align="right")
    p.text(996, 706, 800, 28,
           "忙碌比例仅为合成示意；瓶颈依实际饱和 profile 判定。", 17, MUTED)

    cards = [
        (64, BLUE, "TTFT / 首 token 延迟",
         "低负载均值用于结构预测；线上还受队列影响。\n"
         "按 p50/p95/p99 报告；冷/热缓存、压缩与\n"
         "前缀命中成本单独分组，质量不能退化。"),
        (674, VIOLET, "TPOT / 生成节拍",
         "后续有效 token 的平均步长；MTP 必须按\n"
         "每次验证实际交付 token 数摊销，不能用 chunk 间隔。"),
        (1284, TEAL, "吞吐 / Goodput",
         "输出 tok/s 与输入+输出总 tok/s 分开；\n"
         "容量与实际到达负载不同，SLO 下再看 goodput。"),
    ]
    for x, accent, title, body in cards:
        p.rect(x, 778, 572, 211, WHITE, stroke=BORDER)
        p.rect(x, 778, 9, 211, accent, stroke=accent, radius=0)
        p.text(x + 24, 793, 528, 45, title, 23, INK, bold=True)
        p.text(x + 25, 846, 520, 119, body, 18, MUTED)
    p.text(70, 997, 1780, 28,
           "比较约束：固定模型 revision、精度、H800 总数、业务任务与采样；分层记录实际输出、输入压缩、冷热缓存、并发 / RPS。",
           17, ROSE, bold=True, align="center")
    return p


def page_strategies():
    p = Page("optimization-matrix", "02｜优化策略与架构映射", 2,
             "OPTIMIZATION MATRIX / T001–T013",
             "优化策略与架构映射",
             "按 13 个研究候选对齐机制、影响指标、版本前提与代价；Prefix 命中与上下文压缩必须联合评估质量及缓存策略。")
    xs = [64, 379, 1021, 1281, 1856]
    p.rect(64, 210, 1792, 46, "#DCE5F1", stroke="#DCE5F1", radius=9)
    for x, w, label in zip([78, 390, 1031, 1292], [281, 613, 228, 544],
                            ["ID / 优化项", "关键机制与作用路径", "主指标", "前提 / 潜在代价"]):
        p.text(x, 214, w, 38, label, 20, INK, bold=True)
    for x in xs[1:-1]:
        p.line(x, 211, x, 1008, BORDER, 1)

    def group(y, label, color, tint):
        p.rect(64, y, 1792, 27, tint, stroke=tint, radius=0)
        p.text(79, y + 1, 1745, 24, label, 17, color, bold=True)

    def row(y, label, mechanism, metric, requirement, color, even):
        p.rect(64, y, 1792, 50, "#FFFFFF" if even else "#F1F5FA", stroke="#E2E9F1", radius=0, width=0.8)
        p.rect(64, y, 7, 50, color, stroke=color, radius=0)
        p.text(80, y + 4, 293, 42, label, 18, INK, bold=True)
        p.text(393, y + 4, 612, 42, mechanism, 18, MUTED)
        p.text(1034, y + 4, 235, 42, metric, 18, color, bold=True)
        p.text(1293, y + 4, 550, 42, requirement, 17, MUTED)

    group(263, "A  ·  Prefill / 首 token 路径", BLUE, BLUE_TINT)
    data_a = [
        ("T001 · 分块 Prefill", "分块 token budget + 连续批处理，控制长输入阻塞", "TTFT · TPOT 尾部", "需混合负载 trace；切块增加轮次"),
        ("T005 · GR/PLE 序列并行", "长序列分片与依赖展开，换取更短 Prefill", "长输入 TTFT", "依赖框架 / 跨卡通信支持"),
        ("T008 · Prefix / KV 缓存", "复用兼容 QSA / GDN 状态；按请求/token/成本测命中", "多轮 TTFT · TPS", "两池容量、驱逐、恢复成本、数值一致性"),
        ("T011 · Host / Tokenizer", "减少 CPU、分词和 launch 的暴露等待", "短输入 TTFT · 尾延迟", "需 CPU/GPU 时间线证明瓶颈"),
        ("T013 · 上下文压缩", "结构去重 / 检索剪裁 / 摘要；重算前缀 key", "TTFT · 质量", "压缩开销、证据遗漏、缓存命中变化"),
    ]
    for i, item in enumerate(data_a):
        row(290 + i * 50, *item, BLUE, i % 2 == 0)

    group(543, "B  ·  Decode / 单 token 步长", VIOLET, VIOLET_TINT)
    data_b = [
        ("T002 · CUDA Graph", "消除部分 Host 提交与调度空隙", "TPOT · 短请求", "确认 graph capture / shape 覆盖"),
        ("T003 · QSA Prepare 融合", "合并 QSA 前处理、访存与启动次数", "TPOT", "先核对 upstream build 已包含部分"),
        ("T004 · QSA 投影融合", "SM90 小 GEMM 与投影算子调度优化", "TPOT · 输出 TPS", "看真实 kernel shape / 数值一致性"),
        ("T006 · PLE 异步预取", "利用 overlap 隐藏 embedding 搬运延迟", "TTFT · TPOT", "CPU / PCIe / NUMA 有暴露等待"),
        ("T009 · 原生 MTP", "降低每最终 token 的验证轮次成本", "TPOT · 输出 TPS", "接受分布、验证成本、质量门槛"),
        ("T010 · MoE 通信", "All-to-All 合并、专家负载平衡与重叠", "TPOT · 输出 TPS", "通信占比、路由和拓扑证据"),
    ]
    for i, item in enumerate(data_b):
        row(570 + i * 50, *item, VIOLET, i % 2 == 0)

    group(873, "C  ·  多卡并行 / 端到端服务", TEAL, TEAL_TINT)
    row(900, "T007 · TP / EP / DP", "同预算重排计算、专家路由与服务副本",
        "TTFT · TPOT · TPS", "总 H800 数固定；collective 与每卡吞吐", TEAL, True)
    row(950, "T012 · Prefill / Decode 分离", "降低混合干扰，分配 P/D 资源与状态通道",
        "尾延迟 · Goodput", "状态传输、规模、SLO、框架支持", TEAL, False)
    p.text(74, 1009, 1773, 23,
           "优先顺序由 profile 决定：局部算子加速不等于端到端收益；已经由基线覆盖的优化不得重复归功。",
           16, ROSE, bold=True, align="center")
    return p


def page_prediction():
    p = Page("benefit-predictor", "03｜收益预测与实验校准", 3,
             "CEILING / CACHE HIT / COMPRESSION / VALIDATION",
             "收益天花板与缓存—压缩联合预测",
             "关键路径、资源容量与前缀命中分账；将有限缓存和输入压缩的质量代价纳入同一请求 trace。")

    inputs = [
        (64, BLUE, BLUE_TINT, "输入 A · 实测基线",
         "TTFT/TPOT 与 Prefill 冷热配对；trace 顺序、\n"
         "输入/输出长度与饱和输出 tok/s。"),
        (673, TEAL, TEAL_TINT, "输入 B · Profile",
         "关键路径、资源忙碌与 Roofline 下界；\n"
         "每前缀 state MiB、restore、eviction 与来源层级。"),
        (1282, VIOLET, VIOLET_TINT, "输入 C · 候选方案",
         "cache MiB/前缀键/命中；压缩后 token、耗时、\n"
         "质量门槛；k_s/k_c/η、MTP 与资源因子。"),
    ]
    for x, accent, tint, title, body in inputs:
        p.rect(x, 220, 575, 123, WHITE, stroke=BORDER)
        p.rect(x, 220, 8, 123, accent, stroke=accent, radius=0)
        p.text(x + 22, 232, 531, 36, title, 22, INK, bold=True)
        p.text(x + 23, 272, 527, 55, body, 17, MUTED)

    p.rect(64, 369, 1102, 466, WHITE, stroke=BORDER)
    p.text(89, 377, 1049, 41, "计算式：延迟、缓存、质量与容量各自设限", 26, BLUE, bold=True)
    formulas = [
        (87, 428, BLUE_TINT, BLUE, "01  ·  关键路径 + Roofline",
         "t′_b = s/k_s + c′ + m′ − η′ min(c′,m′) + h\n"
         "c′ ≥ max(c/k_c, FLOPs/P_eff, Bytes/BW_eff)"),
        (637, 428, TEAL_TINT, TEAL, "02  ·  MTP + 饱和容量",
         "TPOT_MTP ≈ T_verify/E[K]；E[K]=Σ(i=0..d)p^i\n"
         "R′cap = R0 · max(u_r) / max(u_r f_r)\n"
         "R_served ≤ min(λ × E[O], R′cap)"),
        (87, 626, VIOLET_TINT, VIOLET, "03  ·  Prefix 命中与缓存成本",
         "h_req = ΣH_i / N；h_tok = Σ(H_i P_i)/ΣL_i\n"
         "T_cache = F + ℓ + (1−H)C + H(S+R)"),
        (637, 626, ORANGE_TINT, ORANGE, "04  ·  上下文压缩 + 质量门槛",
         "T_combo = F+A+ℓ′+(1−H′)C′+H′(S′+R′)\n"
         "A < C−C′（冷态盈亏）；Quality ≥ Q_min\n"
         "S_abs = T0/F（全 Prefill 消失的极松上界）"),
    ]
    for x, y, tint, accent, title, formula in formulas:
        p.rect(x, y, 516, 178, tint, stroke=tint)
        p.text(x + 20, y + 10, 474, 38, title, 21, accent, bold=True)
        p.text(x + 20, y + 54, 478, 110, formula, 17, INK)

    p.rect(1199, 369, 656, 466, WHITE, stroke=BORDER)
    p.text(1221, 379, 609, 42, "合成情景：缓存容量与压缩交互", 25, VIOLET, bold=True)
    p.chip(1221, 428, 215, "SYNTHETIC · 非实测", ROSE_TINT, ROSE)
    p.rect(1222, 482, 610, 47, "#E9EEF5", stroke="#E9EEF5", radius=0)
    for x, w, txt in [(1233,215,"情景"),(1450,113,"TTFT ms"),
                      (1580,98,"请求 hit"),(1682,139,"vs 冷原文")]:
        p.text(x, 488, w, 34, txt, 18, INK, bold=True)
    rows = [
        ("原文：无缓存", "121.50", "0%", "1.000×", BLUE),
        ("原文：40 MiB LRU", "114.62", "12.5%", "1.060×", TEAL),
        ("原文：无限完整键", "86.00", "62.5%", "1.413×", VIOLET),
        ("原文：免费命中", "60.62", "62.5%", "2.004×", VIOLET),
        ("压缩+LRU*", "75.92", "50.0%", "1.600×", ORANGE),
    ]
    for i, (name, ttft, hit, speedup, accent) in enumerate(rows):
        y = 529 + i * 47
        p.rect(1222, y, 610, 47, WHITE if i % 2 == 0 else "#F4F7FA", stroke=BORDER, radius=0)
        for x, w, txt in [(1233,215,name),(1450,113,ttft),
                          (1580,98,hit),(1682,139,speedup)]:
            p.text(x, y + 2, w, 43, txt, 18, accent if x == 1233 else INK,
                   bold=(x == 1233))
    p.text(1225, 774, 595, 54,
           "*压缩质量未验证；8 个虚构请求，非 H800 实测。\n"
           "仅预测低载 TTFT；饱和 TPS 仍需另测资源需求。",
           17, MUTED)

    p.rect(64, 859, 1791, 152, "#EDF2F9", stroke=BORDER)
    p.text(85, 869, 1760, 30, "模型校准与验证闭环", 23, INK, bold=True)
    flow = [
        ("固定基线", "模型 / 精度 / GPU / 有序 trace"),
        ("三层天花板", "免费 oracle / 无淘汰 / 有限容量"),
        ("单变量 A/B", "均值 + 分位数 + 质量"),
        ("误差更新", "(预测−实测)/实测；留出集"),
    ]
    for i, (title, detail) in enumerate(flow):
        x = 88 + i * 443
        p.rect(x, 908, 383, 81, WHITE, stroke=BORDER, radius=12)
        p.text(x + 12, 915, 354, 28, title, 20, INK, bold=True, align="center")
        p.text(x + 11, 948, 361, 31, detail, 16, MUTED, align="center")
        if i < 3:
            p.arrow(x + 389, 947, x + 431, 947, BLUE)
    p.text(72, 1011, 1776, 21,
           "模型：predict.py（关键路径/容量） + cache_context.py（Prefix/压缩）    ·    无真实 H800 trace / 质量结果时不宣称收益。",
           16, ROSE, bold=True, align="center")
    return p


def main():
    pages = [page_performance(), page_strategies(), page_prediction()]
    mxfile = ET.Element("mxfile", {
        "host": "app.diagrams.net", "modified": "2026-11-17T00:00:00.000Z",
        "agent": "native-xml", "version": "24.7.17",
        "type": "device", "pages": "3",
    })
    for page in pages:
        mxfile.append(page.diagram)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    for page, name in zip(pages, ("01-performance", "02-strategies", "03-prediction")):
        page.svg_file(PREVIEWS / (name + ".svg"))
    ET.indent(mxfile, space="  ")
    ET.ElementTree(mxfile).write(FILE, encoding="utf-8", xml_declaration=True)
    print("Generated 3 native draw.io pages + 3 matching SVG previews")


if __name__ == "__main__":
    main()
