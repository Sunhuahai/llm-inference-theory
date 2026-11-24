#!/usr/bin/env python3
"""Read-only structural checks for the native draw.io and SVG sources."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "inference-optimization-map.drawio"
SVG_DIR = ROOT / "previews"


def validate() -> None:
    source = ET.parse(SOURCE).getroot()
    pages = source.findall("diagram")
    assert len(pages) == 3, f"Expected 3 pages, saw {len(pages)}"
    assert source.get("pages") == "3"
    assert len({p.get("id") for p in pages}) == len(pages)
    svg_pages = sorted(SVG_DIR.glob("[0-9][0-9]-*.svg"))
    assert [p.name for p in svg_pages] == [
        "01-performance.svg", "02-strategies.svg", "03-prediction.svg"
    ], f"Expected exactly 3 matching SVGs, saw {svg_pages}"
    for i, p in enumerate(pages, 1):
        graph = p.find("mxGraphModel")
        assert graph is not None
        root = graph.find("root")
        assert root is not None
        cells = root.findall("mxCell")
        ids = [c.get("id") for c in cells]
        assert len(ids) == len(set(ids)), f"Duplicate mxCell ids in page {i}"
        assert len(cells) >= 20, f"Page {i} seems flattened or empty"
        assert not root.findall(".//image"), f"Embedded images in page {i}"
        assert graph.get("pageWidth") == "1920" and graph.get("pageHeight") == "1080"
        for c in cells:
            if c.get("vertex") != "1":
                continue
            g = c.find("mxGeometry")
            assert g is not None
            x, y, w, h = (float(g.get(key, "0")) for key in ("x", "y", "width", "height"))
            assert 0 <= x <= 1920 and 0 <= y <= 1080
            assert x + w <= 1921 and y + h <= 1081, f"Out of page: {i}, {c.get('id')}"
    namespace = "{http://www.w3.org/2000/svg}"
    for svg in svg_pages:
        root = ET.parse(svg).getroot()
        assert root.tag == namespace + "svg"
        assert root.get("viewBox") == "0 0 1920 1080"
        assert root.find(".//" + namespace + "image") is None
    print("PASS: 3 native pages, 3 valid SVGs, unique editable cells, no embedded raster")


if __name__ == "__main__":
    validate()
