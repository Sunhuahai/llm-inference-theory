#!/usr/bin/env python3
"""Build a linked, editable draw.io atlas and matching native-text SVG pages.

Content and geometry live in pages/*.json. Only the Python standard library is
required. draw.io links and edge terminals follow the official format reference:
https://www.drawio.com/docs/reference/diagram-generation/style-reference/
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path
import unicodedata
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = 'https://github.com/Sunhuahai/llm-inference-theory/blob/main/'
FONT = 'Noto Sans CJK SC, Microsoft YaHei, PingFang SC, sans-serif'
NAV = [('overview', '00 全局关系'), ('theory', '01 性能与诊断'),
       ('mechanisms', '02 机制与耦合'), ('models', '03 性能模型'),
       ('agent', '04 自主优化'), ('evidence', '05 实验与证据'),
       ('project', '06 项目使用')]
PALETTE = {
    'theory': ('#2459a6', '#edf4ff', '#8caee0'),
    'models': ('#087a78', '#eaf8f5', '#80beb5'),
    'agent': ('#7042a6', '#f4effb', '#b39ace'),
    'evidence': ('#a96612', '#fff6e7', '#dfbb7c'),
    'neutral': ('#465b70', '#f1f5f9', '#b5c3d0'),
    'project': ('#245f56', '#eef8ee', '#98b9a0'),
}
EDGE = {'flow': '#5b748d', 'feedback': '#b87527', 'constraint': '#8b6baf'}


def n(value):
    return str(int(value)) if float(value).is_integer() else f'{value:g}'


def text_width(text, size):
    return sum(1 if unicodedata.east_asian_width(c) in 'WF' else .57 for c in text) * size


def midpoint(points):
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    rest = sum(lengths) / 2
    for a, b, length in zip(points, points[1:], lengths):
        if rest <= length and length:
            t = rest / length
            return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        rest -= length
    return points[-1]


def validate(page, page_ids):
    width, height = page['size']
    ids = set()
    for node in page['nodes']:
        assert node['id'] not in ids, f"duplicate node: {node['id']}"
        ids.add(node['id'])
        x, y, w, h = node['rect']
        assert x >= 20 and y >= 285 and x+w <= width-20 and y+h <= height-115, (page['id'], node['id'], 'bounds')
        assert len(node.get('lines', [])) <= 4, node['id']
        size = node.get('font', 24)
        for line in node.get('lines', []):
            assert text_width(line, size) <= w-48, (page['id'], node['id'], 'long body', line)
        assert text_width(node['title'], node.get('title_font', 29)) <= w-64, (page['id'], node['id'], 'long title')
        if node.get('doc'):
            assert (REPO / node['doc']).exists(), f"missing document: {node['doc']}"
        if node.get('page'):
            assert node['page'] in page_ids, f"missing page: {node['page']}"
    for edge in page.get('edges', []):
        assert edge['source'] in ids and edge['target'] in ids, (page['id'], 'edge reference', edge)
        assert len(edge['points']) >= 2, edge
        by_id = {node['id']: node for node in page['nodes']}
        for key, index in [('source', 0), ('target', -1)]:
            x, y, w, h = by_id[edge[key]]['rect']
            px, py = edge['points'][index]
            assert x <= px <= x+w and y <= py <= y+h and (px in (x, x+w) or py in (y, y+h)), (page['id'], 'edge port', edge[key])
        for a, b in zip(edge['points'], edge['points'][1:]):
            assert a[0] == b[0] or a[1] == b[1], (page['id'], 'non-orthogonal edge', edge)
            for node in page['nodes']:
                x, y, w, h = node['rect']
                vertical_cross = a[0] == b[0] and x < a[0] < x+w and max(min(a[1], b[1]), y) < min(max(a[1], b[1]), y+h)
                horizontal_cross = a[1] == b[1] and y < a[1] < y+h and max(min(a[0], b[0]), x) < min(max(a[0], b[0]), x+w)
                assert not (vertical_cross or horizontal_cross), (page['id'], 'edge crosses card', edge['source'], edge['target'], node['id'])
    for i, a in enumerate(page['nodes']):
        ax, ay, aw, ah = a['rect']
        for b in page['nodes'][i+1:]:
            bx, by, bw, bh = b['rect']
            assert not (max(ax,bx)<min(ax+aw,bx+bw) and max(ay,by)<min(ay+ah,by+bh)), (page['id'], 'overlap', a['id'], b['id'])


class Renderer:
    def __init__(self, page, all_pages):
        self.page = page
        self.all_pages = {p['id']: p for p in all_pages}
        self.width, self.height = page['size']
        self.seq = 0
        self.svg = ET.Element('svg', {'xmlns': 'http://www.w3.org/2000/svg', 'width': n(self.width),
            'height': n(self.height), 'viewBox': f'0 0 {self.width} {self.height}', 'role': 'img'})
        ET.SubElement(self.svg, 'title').text = page['title']
        ET.SubElement(self.svg, 'desc').text = page['subtitle'] + '。节点链接对应文档；导航链接对应子图。'
        defs = ET.SubElement(self.svg, 'defs')
        for name, color in EDGE.items():
            marker = ET.SubElement(defs, 'marker', {'id': f'arrow-{name}', 'viewBox': '0 0 10 10',
                'refX': '8.5', 'refY': '5', 'markerWidth': '8', 'markerHeight': '8', 'orient': 'auto-start-reverse'})
            ET.SubElement(marker, 'path', {'d': 'M 0 0 L 10 5 L 0 10 z', 'fill': color})
        self.layers = {key: ET.SubElement(self.svg, 'g', {'id': key}) for key in ['background', 'edges', 'cards', 'labels']}
        self.diagram = ET.Element('diagram', {'id': page['id'], 'name': page['tab'], 'alt': page['title']})
        model = ET.SubElement(self.diagram, 'mxGraphModel', {'dx': n(self.width), 'dy': n(self.height),
            'grid': '1', 'gridSize': '10', 'guides': '1', 'tooltips': '1', 'connect': '1',
            'arrows': '1', 'fold': '1', 'page': '1', 'pageScale': '1',
            'pageWidth': n(self.width), 'pageHeight': n(self.height), 'background': '#f8fafc'})
        self.mx = ET.SubElement(model, 'root')
        ET.SubElement(self.mx, 'mxCell', {'id': '0'})
        ET.SubElement(self.mx, 'mxCell', {'id': '1', 'parent': '0'})
        self.nodes = {node['id']: node for node in page['nodes']}

    def cell(self, ident, value, style, rect, link=None, tooltip=None):
        attrs = {'style': style, 'vertex': '1', 'parent': '1'}
        if link:
            wrapper = ET.SubElement(self.mx, 'UserObject', {'id': ident, 'label': value,
                'link': link, 'linkTarget': '_blank' if not link.startswith('data:') else '',
                'tooltip': tooltip or ''})
            cell = ET.SubElement(wrapper, 'mxCell', attrs)
        else:
            cell = ET.SubElement(self.mx, 'mxCell', {'id': ident, 'value': value, **attrs})
        x, y, w, h = rect
        ET.SubElement(cell, 'mxGeometry', {'x': n(x), 'y': n(y), 'width': n(w), 'height': n(h), 'as': 'geometry'})
        return cell

    def fresh(self):
        self.seq += 1
        return 'decor-' + str(self.seq)

    def rect(self, rect, fill, stroke='none', radius=16, layer='background'):
        x, y, w, h = rect
        ET.SubElement(self.layers[layer], 'rect', {'x': n(x), 'y': n(y), 'width': n(w), 'height': n(h),
            'rx': n(radius), 'fill': fill, 'stroke': stroke, 'stroke-width': '2'})
        self.cell(self.fresh(), '', f'rounded=1;arcSize=8;html=0;fillColor={fill};strokeColor={stroke};strokeWidth=2;connectable=0;', rect)

    def text(self, x, y, value, size=24, color='#394b5d', bold=False, layer='labels'):
        ET.SubElement(self.layers[layer], 'text', {'x': n(x), 'y': n(y), 'font-family': FONT,
            'font-size': n(size), 'font-weight': '700' if bold else '400', 'fill': color}).text = value
        self.cell(self.fresh(), value,
            f'text;html=0;align=left;verticalAlign=middle;whiteSpace=wrap;overflow=hidden;'
            f'fontFamily=Noto Sans CJK SC;fontSize={size};fontColor={color};fontStyle={int(bold)};connectable=0;',
            [x-2, y-size-3, min(self.width-x-20, text_width(value,size)+25), size*1.5])

    def chrome(self):
        self.rect([0, 0, self.width, self.height], '#f8fafc', radius=0)
        self.rect([0, 0, self.width, 165], '#112b43', radius=0)
        self.text(65, 66, self.page['title'], 46, '#ffffff', True)
        self.text(67, 119, self.page['subtitle'], 25, '#c9d8e5')
        nav_width = (self.width-130-6*16)/7
        for i, (ident, title) in enumerate(NAV):
            x, y, w, h = 65+i*(nav_width+16), 194, nav_width, 53
            active = ident == self.page['id']
            fill, color = ('#dfeaf4', '#173f64') if active else ('#ffffff', '#536b80')
            href = BASE + 'visualizations/diagrams/research-atlas/' + self.all_pages[ident]['file'] + '.svg'
            a = ET.SubElement(self.layers['labels'], 'a', {'href': href, 'target': '_blank'})
            ET.SubElement(a, 'rect', {'x': n(x), 'y': n(y), 'width': n(w), 'height': n(h), 'rx': '12', 'fill': fill, 'stroke': '#c9d6e1'})
            ET.SubElement(a, 'text', {'x': n(x+w/2), 'y': n(y+35), 'text-anchor': 'middle',
                'font-family': FONT, 'font-size': '23', 'font-weight': '700' if active else '400', 'fill': color}).text = title
            self.cell(self.fresh(), title, f'rounded=1;arcSize=18;html=0;fillColor={fill};strokeColor=#c9d6e1;'
                f'fontFamily=Noto Sans CJK SC;fontSize=23;fontColor={color};fontStyle={int(active)};',
                [x,y,w,h], link='data:page/id,'+ident)
        y = self.height-75
        self.text(65, y, '实线：构成 / 推导 / 数据流', 22, EDGE['flow'], True)
        self.text(525, y, '虚线：反馈 / 约束 / 条件', 22, EDGE['feedback'], True)
        self.text(1000, y, self.page.get('footer', '结构与机制示意；完整系统能力以对应文档为准。'), 22, '#52697c')
        self.text(65, self.height-28, 'LLM Inference Theory  ·  2026-11-25  ·  带 ↗ 的节点可打开对应文档；draw.io 顶部导航可切换页面', 19, '#718497')

    def panel(self, p):
        color, tint, border = PALETTE[p.get('kind','neutral')]
        self.rect(p['rect'], tint, border)
        x,y,w,h = p['rect']
        self.text(x+24,y+43,p['title'],30,color,True)
        if p.get('subtitle'):
            self.text(x+24,y+77,p['subtitle'],21,'#536b80')

    def node(self, node):
        x,y,w,h = node['rect']
        color,tint,border = PALETTE[node.get('kind','neutral')]
        title_size, body_size = node.get('title_font',29), node.get('font',24)
        doc = node.get('doc')
        target_page = node.get('page')
        href = BASE+doc if doc else (BASE+'visualizations/diagrams/research-atlas/'+self.all_pages[target_page]['file']+'.svg' if target_page else None)
        mxlink = 'data:page/id,'+target_page if target_page else href
        group = ET.SubElement(self.layers['cards'], 'a' if href else 'g', {'id': node['id'], **({'href':href,'target':'_blank'} if href else {})})
        ET.SubElement(group,'title').text = node['title'] + (' — '+doc if doc else '')
        ET.SubElement(group,'rect', {'x':n(x),'y':n(y),'width':n(w),'height':n(h),'rx':'14',
            'fill':'#ffffff','stroke':border,'stroke-width':'2'})
        ET.SubElement(group,'rect', {'x':n(x+2),'y':n(y+14),'width':'5','height':n(h-28),'rx':'2','fill':color})
        ET.SubElement(group,'text', {'x':n(x+25),'y':n(y+45),'font-family':FONT,'font-size':n(title_size),'font-weight':'700','fill':color}).text=node['title']
        if href:
            ET.SubElement(group,'text', {'x':n(x+w-32),'y':n(y+43),'font-family':FONT,'font-size':'22','fill':border}).text='↗'
        for i,line in enumerate(node.get('lines',[])):
            ET.SubElement(group,'text', {'x':n(x+25),'y':n(y+86+i*34),'font-family':FONT,'font-size':n(body_size),'fill':'#3d5264'}).text=line
        if node.get('tag'):
            ET.SubElement(group,'text', {'x':n(x+w-22),'y':n(y+h-17),'text-anchor':'end','font-family':FONT,'font-size':'19','font-weight':'700','fill':color}).text=node['tag']
        value=f'<div style="font-size:{title_size}px;font-weight:bold;color:{color};line-height:1.2">{html.escape(node["title"])}'+(' ↗' if href else '')+'</div>'
        value+=f'<div style="font-size:{body_size}px;color:#3d5264;line-height:1.42;margin-top:14px">'+'<br>'.join(html.escape(t) for t in node.get('lines',[]))+'</div>'
        if node.get('tag'):
            value+=f'<div style="font-size:19px;color:{color};text-align:right;margin-top:8px">{html.escape(node["tag"])}</div>'
        self.cell(node['id'],value,f'rounded=1;arcSize=8;whiteSpace=wrap;html=1;align=left;verticalAlign=top;'
            f'spacingLeft=24;spacingRight=24;spacingTop=20;fillColor=#ffffff;strokeColor={border};strokeWidth=2;'
            'fontFamily=Noto Sans CJK SC;overflow=hidden;',node['rect'],link=mxlink,tooltip=doc)

    def edge(self, e, index):
        kind=e.get('kind','flow'); color=EDGE[kind]; pts=e['points']; dashed=kind!='flow'
        attrs={'d':'M '+' L '.join(f'{x} {y}' for x,y in pts),'fill':'none','stroke':color,'stroke-width':'3',
            'stroke-linejoin':'round','stroke-linecap':'round','marker-end':f'url(#arrow-{kind})'}
        if dashed: attrs['stroke-dasharray']='10 8'
        ET.SubElement(self.layers['edges'],'path',attrs)
        start,end=pts[0],pts[-1]
        sx,sy,sw,sh=self.nodes[e['source']]['rect']; tx,ty,tw,th=self.nodes[e['target']]['rect']
        style=f'edgeStyle=none;rounded=0;html=0;endArrow=block;endFill=1;strokeColor={color};strokeWidth=3;'
        style+=f'dashed={int(dashed)};fontFamily=Noto Sans CJK SC;fontSize=21;fontColor={color};labelBackgroundColor=#f8fafc;'
        style+=f'exitX={(start[0]-sx)/sw};exitY={(start[1]-sy)/sh};exitDx=0;exitDy=0;'
        style+=f'entryX={(end[0]-tx)/tw};entryY={(end[1]-ty)/th};entryDx=0;entryDy=0;'
        cell=ET.SubElement(self.mx,'mxCell',{'id':f'edge-{index}','edge':'1','parent':'1','source':e['source'],
            'target':e['target'],'value':e.get('label',''),'style':style})
        geo=ET.SubElement(cell,'mxGeometry',{'relative':'1','as':'geometry'})
        if len(pts)>2:
            arr=ET.SubElement(geo,'Array',{'as':'points'})
            for x,y in pts[1:-1]: ET.SubElement(arr,'mxPoint',{'x':n(x),'y':n(y)})
        if e.get('label'):
            label=e['label']; lx,ly=e.get('label_at',midpoint(pts)); lw=text_width(label,21)+22
            ET.SubElement(self.layers['labels'],'rect',{'x':n(lx-lw/2),'y':n(ly-20),'width':n(lw),'height':'32','rx':'6','fill':'#f8fafc'})
            ET.SubElement(self.layers['labels'],'text',{'x':n(lx),'y':n(ly+3),'text-anchor':'middle','font-family':FONT,'font-size':'21','fill':color}).text=label
            mx,my=midpoint(pts)
            ET.SubElement(geo,'mxPoint',{'x':n(lx-mx),'y':n(ly-my),'as':'offset'})

    def build(self):
        self.chrome()
        for panel in self.page.get('panels',[]): self.panel(panel)
        # Edges precede vertices in draw.io so cards retain visual priority.
        for i,e in enumerate(self.page.get('edges',[])): self.edge(e,i)
        for node in self.page['nodes']: self.node(node)
        ET.indent(self.svg)
        ET.ElementTree(self.svg).write(HERE/(self.page['file']+'.svg'),encoding='utf-8',xml_declaration=True)
        return self.diagram


def main():
    pages=[json.loads(p.read_text()) for p in sorted((HERE/'pages').glob('*.json'))]
    ids={p['id'] for p in pages}
    assert ids == {x[0] for x in NAV}, f'missing atlas pages: {ids}'
    for page in pages: validate(page,ids)
    mxfile=ET.Element('mxfile',{'host':'app.diagrams.net','agent':'LLM Inference Theory atlas generator',
        'version':'24.7.17','type':'device','pages':str(len(pages))})
    for page in pages: mxfile.append(Renderer(page,pages).build())
    ET.indent(mxfile)
    ET.ElementTree(mxfile).write(HERE/'llm-inference-research-atlas.drawio',encoding='utf-8',xml_declaration=True)
    print(json.dumps({'pages':len(pages),'nodes':sum(len(p['nodes']) for p in pages),
        'relations':sum(len(p.get('edges',[])) for p in pages)},ensure_ascii=False))


if __name__=='__main__':
    main()
