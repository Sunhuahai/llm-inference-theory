#!/usr/bin/env python3
"""Build approachable native draw.io pages and matching SVG previews.

Edits to the two original dense reference pages are preserved by ID.
The new pages are generated from the same geometric primitives as the SVGs.
"""

from __future__ import annotations

import html
import xml.etree.ElementTree as ET
from pathlib import Path


DIR = Path(__file__).resolve().parent
FILE = DIR / "inference-optimization-map.drawio"
PREVIEWS = DIR / "previews"
W, H = 1600, 900
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
        self.text(56, 28, 1130, 32, section.upper(), 16, BLUE, bold=True)
        self.text(56, 62, 1460, 62, title, 37, INK, bold=True)
        self.text(56, 135, 1480, 36, subtitle, 18, MUTED)
        self.line(56, 188, 1544, 188, BORDER, 2)
        self.text(56, 854, 1350, 28,
                  "QWEN · H800    固定 checkpoint / 固定精度 / 固定 GPU 预算    ·    研究模型 ≠ H800 实测",
                  13, MUTED)
        self.text(1460, 849, 83, 32, f"{number:02d}", 21, BLUE, bold=True, align="right")

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


def page01():
    p = Page("human-roadmap", "01｜阅读路线", 1, "先看这页",
             "先看懂系统，再预测优化效果", "每页只回答一个问题；把未知参数留给真实 H800 profile。")
    labels = [
        ("01", "理解请求", "一次输入怎样变成 token", BLUE, BLUE_TINT),
        ("02", "读懂指标", "TTFT / TPOT / 吞吐", TEAL, TEAL_TINT),
        ("03", "定位瓶颈", "关键路径与资源上限", VIOLET, VIOLET_TINT),
        ("04", "选择优化", "12 张 trick 有条件地选", ORANGE, ORANGE_TINT),
        ("05", "先预测", "范围、上限与新瓶颈", GREEN, GREEN_TINT),
    ]
    for i, (n, title, body, accent, tint) in enumerate(labels):
        x = 57 + i * 306
        p.rect(x, 248, 252, 166, WHITE, stroke=BORDER)
        p.chip(x + 17, 264, 58, n, tint, accent)
        p.text(x + 17, 309, 220, 41, title, 24, INK, bold=True)
        p.text(x + 17, 354, 220, 39, body, 16, MUTED)
        if i < 4:
            p.arrow(x + 261, 329, x + 296, 329, accent, 3)
    p.rect(58, 460, 714, 249, BLUE_TINT, stroke=BLUE_TINT)
    p.text(87, 480, 670, 45, "给第一次看图的人", 26, BLUE, bold=True)
    p.text(88, 537, 650, 141,
           "先看 02–04 页：请求旅程、指标与瓶颈。\n再看 05–07 页：Prefill / Decode / 多卡。\n最后看 08–09 页：预测器与校准闭环。",
           21, INK)
    p.rect(818, 460, 724, 249, TEAL_TINT, stroke=TEAL_TINT)
    p.text(850, 480, 660, 45, "给实施 Agent 的最短路径", 26, TEAL, bold=True)
    p.text(849, 537, 670, 141,
           "固定 checkpoint、精度、GPU 预算与 workload。\n采集同一组请求的关键路径 + 饱和容量。\n每次只改一个变量，先预测再 A/B。",
           21, INK)
    p.rect(58, 742, 1484, 78, WHITE, stroke=BORDER)
    p.text(83, 757, 1420, 48,
           "核心原则：预测“可消除的暴露成本”，而不是直接搬用上游 benchmark 的加速倍数。",
           22, INK, bold=True, align="center")
    return p


def page02():
    p = Page("human-lifecycle", "02｜请求生命周期", 2, "请求旅程",
             "一个 token 从哪里来？", "TTFT 关注首 token 之前，TPOT 关注后续 token 的节拍。")
    steps = [
        ("请求到达", "tokenize / 接入", MUTED, "#EDF1F6"),
        ("等待调度", "队列 / batch", ORANGE, ORANGE_TINT),
        ("Prefill", "处理输入 token", BLUE, BLUE_TINT),
        ("首个 Decode", "产生首 token", TEAL, TEAL_TINT),
        ("后续 Decode", "循环逐 token", VIOLET, VIOLET_TINT),
    ]
    for i, (title, body, color, tint) in enumerate(steps):
        x = 58 + 309 * i
        p.rect(x, 265, 257, 154, tint, stroke=tint)
        p.text(x + 17, 280, 222, 53, title, 25, INK, bold=True, align="center")
        p.text(x + 18, 337, 220, 48, body, 18, color, align="center")
        if i < 4:
            p.arrow(x + 264, 341, x + 298, 341, color)
    p.arrow(75, 477, 1107, 477, BLUE, 5)
    p.text(330, 489, 720, 44, "TTFT = 到达请求 → 收到首个有效生成 token", 22, BLUE, bold=True, align="center")
    p.arrow(1153, 477, 1518, 477, VIOLET, 5)
    p.text(1137, 492, 379, 45, "TPOT：后续 token 的平均间隔", 18, VIOLET, bold=True, align="center")
    p.card(58, 588, 466, 185, "TTFT", "排队 + 预处理 + Prefill\n+ 首个 Decode + 传输", accent=BLUE, tint=BLUE_TINT, size=21)
    p.card(565, 588, 467, 185, "TPOT", "后续 N 个 token 的总时间\n除以 N；多 token chunk 要摊开", accent=VIOLET, tint=VIOLET_TINT, size=20)
    p.card(1072, 588, 469, 185, "输出吞吐", "在给定并发/到达率下\n单位时间实际交付的 token 数", accent=TEAL, tint=TEAL_TINT, size=20)
    return p


def page03():
    p = Page("human-metrics", "03｜指标与公平比较", 3, "指标定义",
             "三个数字，不是一个“加速比”", "同一优化可能减少 TTFT、增加 TPOT，或只提高饱和容量。")
    items = [
        (62, BLUE, BLUE_TINT, "TTFT", "用户等首字", "首 token 到达 − 请求发送", "重点：长输入、队列、prefix"),
        (566, VIOLET, VIOLET_TINT, "TPOT", "用户看生成", "生成段时长 / 后续 token 数", "重点：串行 decode step"),
        (1070, TEAL, TEAL_TINT, "TPS", "系统总容量", "固定 workload 下的输出 tok/s", "重点：饱和压测、每卡效率"),
    ]
    for x, accent, tint, tag, title, formula, caveat in items:
        p.rect(x, 240, 464, 303, WHITE, stroke=BORDER)
        p.chip(x + 22, 261, 108, tag, tint, accent)
        p.text(x + 26, 310, 415, 47, title, 27, INK, bold=True)
        p.text(x + 26, 368, 414, 68, formula, 19, accent, bold=True)
        p.line(x + 22, 459, x + 442, 459)
        p.text(x + 26, 472, 413, 54, caveat, 18, MUTED)
    p.rect(60, 590, 1484, 190, "#EDF2F9", stroke="#EDF2F9")
    p.text(92, 611, 1418, 42, "公平 A/B 必须锁定", 26, INK, bold=True)
    for i, (title, body) in enumerate([
        ("模型与精度", "checkpoint / dtype"),
        ("GPU 预算", "卡数 / 拓扑 / 功耗"),
        ("请求分布", "输入/真实输出长度"),
        ("负载条件", "RPS / 并发 / 缓存"),
    ]):
        x = 91 + i * 360
        p.rect(x, 670, 324, 86, WHITE, stroke=BORDER, radius=12)
        p.text(x + 14, 681, 295, 27, title, 19, INK, bold=True)
        p.text(x + 14, 713, 295, 25, body, 15, MUTED)
    return p


def page04():
    p = Page("human-bottlenecks", "04｜诊断瓶颈", 4, "理论到 profile",
             "延迟看关键路径；吞吐看最忙的资源", "把两套账分开，是建立收益预测模型的第一步。")
    p.rect(58, 225, 712, 447, WHITE, stroke=BORDER)
    p.text(86, 247, 650, 44, "A  ·  单请求关键路径", 27, BLUE, bold=True)
    p.text(88, 299, 650, 48, "串行：Host  →  计算 / 通信重叠  →  输出", 18, MUTED)
    p.rect(94, 372, 130, 81, BLUE_TINT, stroke=BLUE_TINT)
    p.text(105, 387, 106, 45, "Host\n串行", 18, BLUE, bold=True, align="center")
    p.arrow(233, 413, 284, 413, BLUE)
    p.rect(300, 353, 254, 61, TEAL_TINT, stroke=TEAL_TINT)
    p.text(309, 365, 232, 37, "GPU 计算 C", 20, TEAL, bold=True, align="center")
    p.rect(300, 430, 191, 61, ORANGE_TINT, stroke=ORANGE_TINT)
    p.text(310, 443, 172, 36, "通信 M", 20, ORANGE, bold=True, align="center")
    p.arrow(567, 414, 643, 414, BLUE)
    p.rect(651, 372, 94, 81, VIOLET_TINT, stroke=VIOLET_TINT)
    p.text(655, 390, 85, 42, "输出", 18, VIOLET, bold=True, align="center")
    p.text(90, 540, 638, 74, "重叠块：C + M − η·min(C,M)\nη = 1 时取 max(C,M)，不是 C + M", 20, INK, bold=True)
    p.rect(807, 225, 734, 447, WHITE, stroke=BORDER)
    p.text(839, 247, 662, 44, "B  ·  饱和容量的资源压力", 27, TEAL, bold=True)
    p.text(842, 300, 636, 35, "以下数值仅用于示意，非 H800 测量", 16, MUTED)
    for i, (name, u, color) in enumerate([
        ("GPU 计算", .91, BLUE), ("HBM", .70, TEAL),
        ("互联通信", .47, ORANGE), ("CPU", .60, VIOLET),
    ]):
        y = 356 + 64 * i
        p.text(841, y, 151, 34, name, 18, INK, bold=True)
        p.rect(1015, y + 7, 414, 18, "#EEF2F8", stroke="#EEF2F8", radius=9)
        p.rect(1015, y + 7, round(414 * u), 18, color, stroke=color, radius=9)
        p.text(1441, y, 69, 34, f"{round(u*100)}%", 17, color, bold=True, align="right")
    p.rect(58, 709, 1483, 107, "#FFF6E9", stroke="#FFE3B7")
    p.text(88, 724, 1410, 80,
           "局部 kernel 2×  ≠  整体 2×。\n先问：原本它占关键路径多少？是否已有重叠？新瓶颈会不会转移？",
           22, INK, bold=True)
    return p


def page05():
    p = Page("human-prefill", "05｜Prefill 优化", 5, "先拿首 token",
             "缩短 Prefill：四种常见抓手", "先从真实 profile 选择；长输入、短输入、多轮复用的优先级不一样。")
    cards = [
        (62, 234, "T001 · Chunked prefill", "把长输入切成可调度的块。\n看 TTFT、混合 batch 的 TPOT；\n可能增加调度轮次。", BLUE, BLUE_TINT),
        (820, 234, "T005 · GR/PLE 序列并行", "把部分序列依赖跨卡展开。\n长输入更可能受益；\n先核对实际框架是否支持。", VIOLET, VIOLET_TINT),
        (62, 485, "T008 · Prefix 状态复用", "复用已验证的前缀状态。\n命中率、状态兼容性与内存\n决定净收益；不能只看 cache hit。", TEAL, TEAL_TINT),
        (820, 485, "T011 · Host / Tokenizer", "减少 CPU 排队、分词和提交空隙。\n短请求与高并发可能敏感；\n须用 CPU/GPU timeline 证明。", ORANGE, ORANGE_TINT),
    ]
    for x, y, title, body, color, tint in cards:
        p.card(x, y, 720, 216, title, body, accent=color, tint=tint, size=21)
    p.rect(62, 736, 1479, 86, "#E8F0FF", stroke="#D5E4FF")
    p.text(92, 749, 1399, 58,
           "预测时分别填写：输入长度分布、暴露的 Prefill 成本、额外调度开销、prefix 命中与状态搬运成本。",
           20, BLUE, bold=True, align="center")
    return p


def page06():
    p = Page("human-decode", "06｜Decode 优化", 6, "逐 token 生成",
             "Decode：小节拍被重复成大开销", "先量每次 token step 的暴露成本；MTP 要额外算“一轮交付几个 token”。")
    cards = [
        ("T002 · CUDA Graph", "削减 Host launch 间隙\n同时检查覆盖形状与预热", BLUE, BLUE_TINT),
        ("T003 · QSA prepare", "融合准备算子和访存\n先查当前 build 是否已覆盖", TEAL, TEAL_TINT),
        ("T004 · QSA 投影", "投影融合与小 GEMM 调度\n关心 SM90 实际 shape", VIOLET, VIOLET_TINT),
        ("T006 · PLE 预取", "异步搬运与 NUMA 放置\n只节省暴露的等待窗口", ORANGE, ORANGE_TINT),
        ("T009 · 原生 MTP", "接受长度 / 验证 / draft 成本\n低 TPOT 不等于高吞吐容量", GREEN, GREEN_TINT),
        ("T010 · MoE 通信", "减少 All-to-All 与专家倾斜\n看通信计算能否重叠", ROSE, ROSE_TINT),
    ]
    for i, (title, body, color, tint) in enumerate(cards):
        col, row = i % 3, i // 3
        x, y = 60 + col * 503, 238 + row * 243
        p.card(x, y, 475, 214, title, body, accent=color, tint=tint, size=19)
    p.rect(60, 740, 1480, 83, "#FFF4E8", stroke="#F6DBB3")
    p.text(88, 749, 1422, 62,
           "MTP 用单位最终 token 时间 = (验证轮次成本 + draft 开销) / 期望交付 token 数；必须测接受分布。",
           20, ORANGE, bold=True, align="center")
    return p


def page07():
    p = Page("human-scale", "07｜多卡与服务", 7, "系统层优化",
             "更多实例，不等于随意增加 GPU", "比较 T007 / T012 时，始终保持整套服务的 H800 总数量相同。")
    p.rect(57, 240, 876, 457, WHITE, stroke=BORDER)
    p.text(84, 262, 800, 42, "相同 GPU 预算下的两种部署形态", 25, BLUE, bold=True)
    p.chip(92, 330, 142, "单实例混跑", BLUE_TINT, BLUE)
    p.rect(91, 393, 272, 130, BLUE_TINT, stroke=BLUE_TINT)
    p.text(111, 408, 236, 96, "同一实例\nPrefill + Decode", 22, BLUE, bold=True, align="center")
    p.arrow(374, 458, 466, 458, INK)
    p.chip(485, 330, 154, "P/D 分离", TEAL_TINT, TEAL)
    p.rect(485, 391, 182, 128, TEAL_TINT, stroke=TEAL_TINT)
    p.text(504, 410, 143, 92, "P 节点\n长输入", 22, TEAL, bold=True, align="center")
    p.arrow(674, 455, 725, 455, TEAL)
    p.rect(737, 391, 163, 128, VIOLET_TINT, stroke=VIOLET_TINT)
    p.text(751, 410, 135, 92, "D 节点\n逐 token", 22, VIOLET, bold=True, align="center")
    p.text(95, 564, 796, 99,
           "P/D 分离还要核算状态传输、负载比例与跨节点链路。\n未确认引擎支持前，不把它当已经可用的功能。",
           20, MUTED)
    p.rect(971, 240, 571, 457, WHITE, stroke=BORDER)
    p.text(999, 262, 507, 43, "并行布局 T007", 25, VIOLET, bold=True)
    for i, (title, desc, color, tint) in enumerate([
        ("TP", "张量并行：算子与 collective", BLUE, BLUE_TINT),
        ("EP", "专家并行：路由与 all-to-all", TEAL, TEAL_TINT),
        ("DP", "数据并行：副本与请求分发", ORANGE, ORANGE_TINT),
    ]):
        y = 327 + 110 * i
        p.rect(1003, y, 506, 84, tint, stroke=tint)
        p.text(1020, y + 12, 64, 56, title, 23, color, bold=True, align="center")
        p.text(1107, y + 14, 375, 54, desc, 18, INK)
    p.rect(57, 733, 1486, 87, "#EDF2F9", stroke=BORDER)
    p.text(84, 749, 1434, 55,
           "服务吞吐的公允比较：总 tok/s、每卡 tok/s、TTFT/TPOT 尾延迟、跨卡通信与实际 GPU 账。",
           20, INK, bold=True, align="center")
    return p


def page08():
    p = Page("human-model", "08｜可计算的预测器", 8, "最终目标",
             "把优化直觉，变成可检验的数字", "理论下界 + 关键路径校准 + 资源瓶颈容量；不是用一个倍数解释所有指标。")
    for x, title, body, accent, tint in [
        (67, "输入 A · 基线", "TTFT / TPOT 均值\n饱和输出 tok/s", BLUE, BLUE_TINT),
        (583, "输入 B · Profile", "互斥阶段与重叠\n计算 / HBM / 通信 / CPU", TEAL, TEAL_TINT),
        (1098, "输入 C · Trick", "局部提速 / 新开销\n资源需求变化因子", VIOLET, VIOLET_TINT),
    ]:
        p.card(x, 230, 441, 171, title, body, accent=accent, tint=tint, size=19)
    p.arrow(311, 415, 311, 469, BLUE)
    p.arrow(807, 415, 807, 469, TEAL)
    p.arrow(1310, 415, 1310, 469, VIOLET)
    p.rect(68, 484, 697, 172, BLUE_TINT, stroke=BLUE_TINT)
    p.text(92, 494, 648, 45, "模型 ① · 低负载关键路径", 25, BLUE, bold=True)
    p.text(91, 548, 641, 82,
           "固定部分 + Σ 阶段校准成本\n串行 + 计算 + 通信 − 重叠 + 新开销",
           21, INK)
    p.rect(809, 484, 728, 172, TEAL_TINT, stroke=TEAL_TINT)
    p.text(836, 494, 667, 45, "模型 ② · 饱和资源需求", 25, TEAL, bold=True)
    p.text(835, 548, 652, 82,
           "原容量 × 原最大资源压力 / 新最大压力\n瓶颈可能转移，且 TPS ≠ 1/TPOT",
           21, INK)
    p.arrow(404, 667, 404, 705, BLUE)
    p.arrow(1172, 667, 1172, 705, TEAL)
    p.rect(72, 719, 1459, 94, WHITE, stroke=BORDER)
    p.text(88, 731, 1428, 62,
           "输出：预测 TTFT / TPOT / 饱和 TPS + 上限 + 新瓶颈 + 假设清单\n运行入口：theory/predictor/predict.py",
           21, INK, bold=True, align="center")
    return p


def page09():
    p = Page("human-validation", "09｜从预测到校准", 9, "模型闭环",
             "每晚反馈，都会让下一次预测更可信", "先从可复核的 A/B 建立校准，再谈统计学习；没有数据就保留区间。")
    steps = [
        ("01", "先预测", "保守 / 中心 / 乐观\n记录局部成本与适用条件", BLUE, BLUE_TINT),
        ("02", "再实验", "同版本、同 GPU、同 trace\n留出测试负载", TEAL, TEAL_TINT),
        ("03", "看误差", "(预测 − 实测) / 实测\n逐指标分析原因", ORANGE, ORANGE_TINT),
        ("04", "更新模型", "修正重叠、额外成本\n和饱和资源因子", VIOLET, VIOLET_TINT),
    ]
    for i, (n, title, body, color, tint) in enumerate(steps):
        x = 60 + i * 378
        p.rect(x, 247, 337, 222, WHITE, stroke=BORDER)
        p.chip(x + 22, 263, 55, n, tint, color)
        p.text(x + 25, 311, 290, 43, title, 25, INK, bold=True)
        p.text(x + 25, 364, 290, 85, body, 18, MUTED)
        if i < 3:
            p.arrow(x + 343, 359, x + 370, 359, color)
    p.rect(61, 527, 1477, 223, "#EDF2F9", stroke=BORDER)
    p.text(94, 544, 1410, 43, "合成算例：为什么不能只报一个指标？", 26, INK, bold=True)
    for x, tag, value, color, tint in [
        (97, "QSA 保守情景", "TPOT 22 → 20.25 ms", BLUE, BLUE_TINT),
        (568, "MTP 假设情景", "TPOT 22 → 15.87 ms", VIOLET, VIOLET_TINT),
        (1041, "但 MTP 容量", "1800 → 1579 tok/s", ROSE, ROSE_TINT),
    ]:
        p.rect(x, 618, 426, 105, WHITE, stroke=BORDER)
        p.text(x + 16, 628, 394, 36, tag, 19, color, bold=True)
        p.text(x + 16, 666, 390, 42, value, 21, INK, bold=True)
    p.rect(60, 773, 1477, 53, "#FFF1DB", stroke="#FFE1B0")
    p.text(83, 779, 1430, 39,
           "上面都是合成数字，不是 H800 成绩。真实性 = 可追溯版本 + 原始结果 + 误差记录。",
           18, ORANGE, bold=True, align="center")
    return p


def main():
    if not FILE.exists():
        raise SystemExit(f"Missing editable legacy file: {FILE}")
    old = ET.parse(FILE).getroot()
    legacy_ids = ("qwen-h800-overview", "qwen-h800-prediction")
    legacy = []
    for ident in legacy_ids:
        found = old.find(f"./diagram[@id='{ident}']")
        if found is None:
            raise SystemExit(f"Cannot preserve legacy page {ident}; refusing overwrite")
        legacy.append(found)
    pages = [page01(), page02(), page03(), page04(), page05(),
             page06(), page07(), page08(), page09()]
    mxfile = ET.Element("mxfile", {
        "host": "app.diagrams.net", "modified": "2026-11-17T00:00:00.000Z",
        "agent": "native-xml", "version": "24.7.17",
        "type": "device", "pages": str(len(pages) + len(legacy)),
    })
    for page in pages:
        mxfile.append(page.diagram)
    for i, diagram in enumerate(legacy, start=10):
        diagram.attrib["name"] = f"{i:02d}｜技术附录（原 {'优化总览' if i == 10 else '收益预测'}）"
        mxfile.append(diagram)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    for page in pages:
        page.svg_file(PREVIEWS / f"{page.number:02d}-{page.slug.removeprefix('human-')}.svg")
    ET.indent(mxfile, space="  ")
    ET.ElementTree(mxfile).write(FILE, encoding="utf-8", xml_declaration=True)
    print(f"Generated {len(pages)} approachable pages + {len(legacy)} preserved appendices")
    print(f"Editable draw.io: {FILE}")


if __name__ == "__main__":
    main()
