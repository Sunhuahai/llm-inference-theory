#!/usr/bin/env python3
"""Generate matching SVG and editable draw.io research maps; standard library only."""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
FONT = 'Noto Sans CJK SC, Microsoft YaHei, sans-serif'


class Figure:
    def __init__(self, name, title, subtitle, width, height):
        self.name, self.width, self.height = name, width, height
        self.svg = ET.Element('svg', {'xmlns': 'http://www.w3.org/2000/svg', 'viewBox': f'0 0 {width} {height}', 'width': str(width), 'height': str(height), 'role': 'img'})
        ET.SubElement(self.svg, 'title').text = title
        ET.SubElement(self.svg, 'desc').text = subtitle
        defs = ET.SubElement(self.svg, 'defs')
        marker = ET.SubElement(defs, 'marker', {'id': 'arrow', 'viewBox': '0 0 10 10', 'refX': '8', 'refY': '5', 'markerWidth': '7', 'markerHeight': '7', 'orient': 'auto-start-reverse'})
        ET.SubElement(marker, 'path', {'d': 'M0 0 L10 5 L0 10 Z', 'fill': '#64748b'})
        ET.SubElement(self.svg, 'rect', {'x': '0', 'y': '0', 'width': str(width), 'height': str(height), 'fill': '#f8fafc'})
        self.mxfile = ET.Element('mxfile', {'host': 'app.diagrams.net', 'version': '24.7.17'})
        diagram = ET.SubElement(self.mxfile, 'diagram', {'name': title, 'id': name})
        model = ET.SubElement(diagram, 'mxGraphModel', {'dx': str(width), 'dy': str(height), 'grid': '1', 'page': '1', 'pageWidth': str(width), 'pageHeight': str(height)})
        self.mxroot = ET.SubElement(model, 'root')
        ET.SubElement(self.mxroot, 'mxCell', {'id': '0'})
        ET.SubElement(self.mxroot, 'mxCell', {'id': '1', 'parent': '0'})
        self.counter = 1
        self.text(60, 65, title, 36, '#0f172a', True)
        self.text(60, 108, subtitle, 21, '#475569')

    def cell(self, value, style, x, y, w, h):
        self.counter += 1
        cell = ET.SubElement(self.mxroot, 'mxCell', {'id': str(self.counter), 'value': value, 'style': style, 'vertex': '1', 'parent': '1'})
        ET.SubElement(cell, 'mxGeometry', {'x': str(x), 'y': str(y), 'width': str(w), 'height': str(h), 'as': 'geometry'})

    def text(self, x, y, value, size=22, color='#334155', bold=False):
        ET.SubElement(self.svg, 'text', {'x': str(x), 'y': str(y), 'font-family': FONT, 'font-size': str(size), 'fill': color, 'font-weight': '700' if bold else '400'}).text = value
        self.cell(value, f'text;html=0;align=left;verticalAlign=middle;whiteSpace=wrap;overflow=hidden;fontFamily=Noto Sans CJK SC;fontSize={size};fontColor={color};fontStyle={1 if bold else 0};', x-2, y-size-3, max(100, len(value)*size+8), size*1.6)

    def rect(self, x, y, w, h, fill='#ffffff', stroke='#cbd5e1', radius=16):
        ET.SubElement(self.svg, 'rect', {'x': str(x), 'y': str(y), 'width': str(w), 'height': str(h), 'rx': str(radius), 'fill': fill, 'stroke': stroke, 'stroke-width': '2'})
        self.cell('', f'rounded=1;whiteSpace=wrap;html=0;fillColor={fill};strokeColor={stroke};strokeWidth=2;arcSize=8;', x, y, w, h)

    def path(self, points, dashed=False):
        attrs = {'d': 'M'+' L'.join(f'{x} {y}' for x, y in points), 'fill': 'none', 'stroke': '#64748b', 'stroke-width': '2.5', 'marker-end': 'url(#arrow)'}
        if dashed:
            attrs['stroke-dasharray'] = '8 6'
        ET.SubElement(self.svg, 'path', attrs)
        self.counter += 1
        cell = ET.SubElement(self.mxroot, 'mxCell', {'id': str(self.counter), 'edge': '1', 'parent': '1', 'style': f'endArrow=block;html=0;strokeColor=#64748b;strokeWidth=2;dashed={int(dashed)};'})
        geo = ET.SubElement(cell, 'mxGeometry', {'relative': '1', 'as': 'geometry'})
        ET.SubElement(geo, 'mxPoint', {'x': str(points[0][0]), 'y': str(points[0][1]), 'as': 'sourcePoint'})
        ET.SubElement(geo, 'mxPoint', {'x': str(points[-1][0]), 'y': str(points[-1][1]), 'as': 'targetPoint'})
        if len(points) > 2:
            arr = ET.SubElement(geo, 'Array', {'as': 'points'})
            for x, y in points[1:-1]:
                ET.SubElement(arr, 'mxPoint', {'x': str(x), 'y': str(y)})

    def save(self):
        ET.indent(self.svg)
        ET.indent(self.mxfile)
        ET.ElementTree(self.svg).write(ROOT / f'{self.name}.svg', encoding='utf-8', xml_declaration=True)
        ET.ElementTree(self.mxfile).write(ROOT / f'{self.name}.drawio', encoding='utf-8', xml_declaration=True)


def research_map():
    f = Figure('research-map', 'LLM 推理优化：理论、建模与 Agent 自主优化', '三条核心研究主线，以独立实验连接解释、预测、决策和反馈', 1920, 1190)
    specs = [
        (60, '#2563eb', '#eff6ff', '01  推理性能理论', '理解成本、边界和优化成立的条件', [
            ('资源与性能边界', 'FLOPs / 多级访存 / 通信 / 串行依赖'),
            ('状态与生成效率', '容量、复用、有效提交数与生成周期'),
            ('执行与服务', '关键路径、并行、动态批处理与 SLO'),
            ('瓶颈诊断', '观测可辨识性、敏感性与区分性实验'),
            ('优化机制与耦合', '干预资源量、边际收益和瓶颈迁移'),
            ('输出：可证伪的条件判断', '明确假设、反例、适用域和测量需求')]),
        (680, '#0d9488', '#f0fdfa', '02  性能建模与预测', '把理论转化为可计算、可校准的模型', [
            ('资源账本与算子模型', 'shape / dtype / 路径 / 硬件速率来源'),
            ('执行图模型', '依赖、资源互斥、重叠与完成时间'),
            ('服务与负载模型', '到达、长度、缓存和调度的共同作用'),
            ('校准与不确定性', '噪声、模型误差、留出与分布外识别'),
            ('候选收益与反事实', '重算变化后的成本，保留预测误差'),
            ('输出：带来源的预测', '区分严格下界、情景值和真实测量')]),
        (1300, '#7c3aed', '#f5f3ff', '03  Agent 自主优化', '有限预算内，寻找并验证更好的实现', [
            ('感知与假设', 'profile → 诊断 → 机制不同的候选'),
            ('优化空间', '配置 / Kernel / Runtime / Serving'),
            ('规划与搜索', '约束、收益、实验成本与探索价值'),
            ('代码进化与组合', '独立工作区、版本谱系、组合回归'),
            ('独立验证与经验记忆', '冻结 oracle、复测、正负结果复用'),
            ('输出：可复现的改进', 'patch + 适用域 + 证据 + 搜索成本')])
    ]
    for x, color, tint, title, subtitle, rows in specs:
        f.rect(x, 160, 560, 650, '#ffffff', color)
        f.rect(x+1, 161, 558, 114, tint, tint)
        f.text(x+26, 211, title, 29, color, True)
        f.text(x+26, 248, subtitle, 21, '#475569')
        for i, (heading, detail) in enumerate(rows):
            y = 318 + 82*i
            f.text(x+26, y, heading, 23, '#0f172a', True)
            f.text(x+26, y+32, detail, 20, '#475569')
        f.path([(x+280, 810), (x+280, 862)])
    f.path([(620, 225), (674, 225)])
    f.path([(1240, 225), (1294, 225)])
    f.rect(60, 866, 1800, 188, '#fff7ed', '#d97706')
    f.text(88, 910, '共享实验与评估平台', 28, '#9a3412', True)
    labels = [('合成验证', '检验数学与控制逻辑'), ('微基准与校准', '测量真实执行成本'), ('模型留出验证', '检验干预与迁移误差'), ('Agent / 系统对照', '同预算、同质量、独立复测')]
    for i, (heading, detail) in enumerate(labels):
        x = 88 + i*442
        f.text(x, 961, heading, 24, '#0f172a', True)
        f.text(x, 999, detail, 20, '#475569')
    f.text(60, 1100, '反馈方向：实验误差修正模型，模型修正诊断与搜索，负结果限定理论适用范围。', 22, '#334155')
    f.text(60, 1146, '结构示意 · CPU 示例为合成验证 · 通用 GPU 优化器与真实 Agent 对照仍需接入和实测', 20, '#64748b')
    f.save()


def optimization_loop():
    f = Figure('optimization-loop', '理论引导的 Agent 自主优化：决策与独立证据', '控制器负责预算与接受规则；Agent 负责提出假设和实现候选', 1660, 1040)
    f.rect(60, 165, 300, 716, '#f1f5f9', '#64748b')
    f.text(85, 210, '确定性控制器', 28, '#334155', True)
    rules = [('固定任务契约', '模型、精度、资源、质量'), ('管理实验预算', '预留、实际消耗、超时'), ('保持评价独立', '冻结 oracle 与留出规则'), ('管理候选状态', '基线、最佳、失败、回退'), ('执行停止条件', '预算耗尽或继续价值不足')]
    for i, (a, b) in enumerate(rules):
        y = 288+i*117
        f.text(85, y, a, 23, '#0f172a', True)
        f.text(85, y+36, b, 18, '#475569')
    cards = [
        (420, 180, '#2563eb', '1  感知与观测', ['服务时间线、计数器', '环境 / 负载 / 执行版本']),
        (830, 180, '#2563eb', '2  理论诊断', ['资源账本、关键路径', '竞争假设与区分性实验']),
        (1240, 180, '#7c3aed', '3  规划与选择', ['预测收益、成本和不确定性', '可行性、探索与候选依赖']),
        (1240, 466, '#7c3aed', '4  独立实现', ['Coding Agent / 配置变更', '可审查 patch 与 parent 版本']),
        (830, 466, '#d97706', '5  独立评估', ['正确性 → 微基准 → 服务', '所有尝试计费，留出复测']),
        (420, 466, '#0d9488', '6  更新与归因', ['接受、拒绝或证据不足', '重新诊断，更新模型与记忆'])
    ]
    for x, y, color, title, body in cards:
        f.rect(x, y, 350, 176, '#ffffff', color)
        f.text(x+22, y+43, title, 27, color, True)
        for i, line in enumerate(body):
            f.text(x+22, y+91+i*36, line, 20, '#475569')
    f.path([(770, 268), (824, 268)])
    f.path([(1180, 268), (1234, 268)])
    f.path([(1415, 356), (1415, 460)])
    f.text(1266, 416, '候选及预算契约', 20, '#64748b')
    f.path([(1240, 554), (1186, 554)])
    f.path([(830, 554), (776, 554)])
    f.path([(595, 466), (595, 362)])
    f.text(433, 416, '下一轮观测', 20, '#64748b')
    f.rect(420, 742, 1170, 139, '#f0fdfa', '#0d9488')
    f.text(446, 788, '共享知识与证据库', 27, '#0f766e', True)
    f.text(446, 831, '成功与失败、机制与适用域、预测与偏差、候选谱系、实验原始证据', 22, '#334155')
    f.path([(1005, 642), (1005, 736)], True)
    f.path([(595, 742), (595, 648)], True)
    f.text(1040, 699, '记录观测', 19, '#64748b')
    f.text(433, 699, '条件化复用', 19, '#64748b')
    f.text(60, 938, '理论决定哪些实验值得做；独立评价决定哪些改进可信；历史结果只在其适用条件下复用。', 23, '#334155')
    f.text(60, 991, '架构设计示意 · 当前离线原型只回放合成候选，不调用 LLM、编译器或 GPU', 20, '#64748b')
    f.save()


if __name__ == '__main__':
    research_map()
    optimization_loop()
    print('Generated 2 SVGs and 2 editable draw.io sources.')
