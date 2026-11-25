#!/usr/bin/env python3
"""Check active Markdown local targets, fences, JSON and vector XML.

Historical archive is read-only and excluded as a validation source. Links into it
still resolve normally. External URL health and Markdown anchors are not checked.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def without_code(text: str, name: str, errors: list[str]) -> str:
    output, fence = [], None
    for line in text.splitlines():
        match = re.match(r'^\s{0,3}(`{3,}|~{3,})(.*)$', line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence) and not match.group(2).strip():
                fence = None
            continue
        if fence is None:
            output.append(line)
    if fence is not None:
        errors.append(f'{name}: unclosed code fence')
    return '\n'.join(output)


def reject_constant(value: str):
    raise ValueError(f'non-finite JSON constant: {value}')


def main() -> int:
    errors: list[str] = []
    counts = {'markdown': 0, 'local_links': 0, 'json': 0, 'xml': 0}
    files = sorted(p for p in ROOT.rglob('*') if p.is_file()
                   and not any(part.startswith('.') or part == '__pycache__'
                               for part in p.relative_to(ROOT).parts)
                   and p.relative_to(ROOT).parts[0] != 'archive')
    for path in files:
        relative = str(path.relative_to(ROOT))
        try:
            if path.suffix == '.md':
                counts['markdown'] += 1
                text = without_code(path.read_text(encoding='utf-8'), relative, errors)
                if len(re.findall(r'^\s*\$\$\s*$', text, re.M)) % 2:
                    errors.append(f'{relative}: unpaired display-math delimiter')
                targets = re.findall(r'\]\(([^\n)]+)\)', text)
                targets += re.findall(r'^\s*\[(?!\^)[^\]]+\]:\s*(\S+)', text, re.M)
                for raw in targets:
                    target = raw.strip().split(' "', 1)[0].strip('<>')
                    uri = urlsplit(target)
                    if uri.scheme or uri.netloc or not uri.path:
                        continue
                    if target.startswith('/'):
                        errors.append(f'{relative}: repository link must be relative: {target}')
                        continue
                    counts['local_links'] += 1
                    resolved = (path.parent / unquote(uri.path)).resolve()
                    if not resolved.is_relative_to(ROOT) or not resolved.exists():
                        errors.append(f'{relative}: missing local target {target}')
            elif path.suffix == '.json':
                counts['json'] += 1
                json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_constant)
            elif path.suffix in {'.svg', '.drawio'}:
                counts['xml'] += 1
                ET.parse(path)
        except (ValueError, OSError, ET.ParseError) as exc:
            errors.append(f'{relative}: {exc}')
    print(json.dumps({'checked': counts, 'errors': errors}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
