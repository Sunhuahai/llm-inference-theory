#!/usr/bin/env python3
"""Offline inference-optimization evidence and theory lifecycle; Python 3.10+."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import tempfile
from datetime import datetime, timezone

SKILL = Path(__file__).resolve().parents[1]
DEFAULT_HOME = Path(os.environ.get('LLM_OPTIMIZER_HOME', '~/.local/share/llm-inference-optimizer')).expanduser()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    def reject(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def dump(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ident(value):
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', value), 'invalid identifier')
    return value


def docpath(value):
    p = Path(value)
    require(not p.is_absolute() and all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', x) for x in p.parts) and p.suffix == '.md', 'invalid document path')
    return p


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), 'expected finite number')
    return float(value)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(data if isinstance(data, str) else dump(data))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextlib.contextmanager
def locked(root):
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = root / '.write-lock'
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError('store locked; verify no writer before manually removing stale lock')
    try:
        yield
    finally:
        lock.rmdir()


@contextlib.contextmanager
def newdir(path):
    require(not path.exists(), 'immutable record exists: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.tmp-', dir=path.parent))
    try:
        yield tmp
        require(not path.exists(), 'destination already exists')
        tmp.rename(path)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def hashes(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob('*.md')) if p.is_file()}


def write_index(path, manifest):
    lines = ['# Active/candidate theory documents', '', 'Version: ' + manifest['version'], 'Parent: ' + str(manifest['parent']), 'Seed source: ' + manifest['source_commit'], '', 'Read applicability and evidence before using any local revision.', '']
    lines.extend('- [' + k + '](documents/' + k + ')' for k in manifest['documents'])
    write(path / 'INDEX.md', '\n'.join(lines) + '\n')


def version(home, name):
    p = home / 'versions' / ident(name)
    m = read(p / 'manifest.json')
    require(m['version'] == name and hashes(p / 'documents') == m['documents'], 'theory version integrity mismatch')
    return p, m


def active(home):
    name = read(home / 'active.json')['version']
    version(home, name)
    return name


def init(home):
    with locked(home):
        if (home / 'active.json').exists():
            return active(home)
        seed = read(SKILL / 'knowledge/manifest.json')
        source = SKILL / 'knowledge/initial'
        require(set(hashes(source)) == set(seed['documents']), 'incomplete seed bundle')
        name = 'seed-' + seed['source_commit'][:12]
        dest = home / 'versions' / name
        if not dest.exists():
            with newdir(dest) as tmp:
                shutil.copytree(source, tmp / 'documents')
                write(tmp / 'manifest.json', {'version': name, 'parent': None, 'source_commit': seed['source_commit'], 'documents': hashes(tmp / 'documents'), 'created_at': now(), 'training': []})
                write_index(tmp, read(tmp / 'manifest.json'))
        version(home, name)
        write(home / 'active.json', {'version': name})
        return name


def validate_prediction(p):
    require(p['schema_version'] == 1, 'unsupported schema')
    for k in ('project', 'round', 'theory_version'):
        ident(p[k])
    for k in ('parent_ref', 'candidate_ref', 'context_id', 'mechanism', 'assumptions', 'validation_plan'):
        require(nonempty(p[k]), 'missing ' + k)
    require(p['validation_role'] in ('search', 'holdout'), 'validation role')
    require(isinstance(p['theory_refs'], list) and p['theory_refs'] and all(nonempty(x) for x in p['theory_refs']), 'theory references required')
    require(isinstance(p['metrics'], dict) and p['metrics'], 'no metrics')
    for key, m in p['metrics'].items():
        ident(key)
        require(m['level'] in ('resource', 'execution', 'service'), 'metric level')
        require(m['direction'] in ('lower', 'higher') and nonempty(m['unit']) and nonempty(m['statistic']), 'metric definition')
        require(number(m['baseline']) >= 0 and number(m['baseline_tolerance_rel']) >= 0, 'negative baseline/tolerance')
        kind = m['kind']
        require(kind in ('point', 'interval', 'lower_bound', 'upper_bound', 'unknown'), 'prediction kind')
        if kind in ('point', 'lower_bound', 'upper_bound'):
            require(number(m['value']) >= 0, 'negative prediction')
        elif kind == 'interval':
            lo, hi = number(m['low']), number(m['high'])
            require(0 <= lo <= hi, 'invalid interval')
            if 'value' in m:
                require(lo <= number(m['value']) <= hi, 'point outside interval')
        if kind == 'unknown':
            require(nonempty(m.get('reason')), 'unknown needs reason')
    if p['validation_role'] == 'holdout':
        require('shadow' in p, 'holdout requires shadow prediction')
    if 'shadow' in p:
        s = p['shadow']
        ident(s['version'])
        require(s['version'] != p['theory_version'] and s['points'], 'invalid shadow')
        for k, v in s['points'].items():
            require(k in p['metrics'] and p['metrics'][k]['kind'] in ('point', 'interval') and 'value' in p['metrics'][k], 'shadow requires a paired active point')
            require(number(v) >= 0, 'negative shadow prediction')
    return p


def freeze(home, folder, source):
    p = validate_prediction(read(source))
    require(p['theory_version'] == active(home), 'prediction must use active theory')
    _, manifest = version(home, p['theory_version'])
    if 'shadow' in p:
        _, sm = version(home, p['shadow']['version'])
        require(sm['parent'] == p['theory_version'] and sm.get('proposal'), 'shadow must be a draft of active theory')
        spec_path = home / 'versions' / p['shadow']['version'] / 'proposal.json'
        require(digest(spec_path) == sm['proposal_sha256'], 'shadow proposal integrity mismatch')
        spec = read(spec_path)
        if p['validation_role'] == 'holdout':
            require(p['project'] + '/' + p['round'] in spec['validation_rounds'] and p['context_id'] in spec['validation_contexts'], 'holdout must be preregistered in draft')
    with locked(folder.parent):
        with newdir(folder) as tmp:
            write(tmp / 'prediction.json', p)
            write(tmp / 'freeze.json', {'prediction_sha256': digest(tmp / 'prediction.json'), 'theory_manifest_sha256': digest(home / 'versions' / p['theory_version'] / 'manifest.json'), 'frozen_at': now(), 'shadow_manifest_sha256': digest(home / 'versions' / p['shadow']['version'] / 'manifest.json') if 'shadow' in p else None})
            write(tmp / 'prediction.md', '# 事前预测\n\n' + p['mechanism'] + '\n\n假设：' + p['assumptions'] + '\n\n验证：' + p['validation_plan'] + '\n\n完整数值与身份见 prediction.json。\n')


def frozen(folder):
    f = read(folder / 'freeze.json')
    require(digest(folder / 'prediction.json') == f['prediction_sha256'], 'prediction was modified after freeze')
    return validate_prediction(read(folder / 'prediction.json')), f


def gain(b, x, direction):
    if b == 0:
        return None
    return (b - x if direction == 'lower' else x - b) / b


def compare(p, m):
    for k in ('project', 'round', 'parent_ref', 'candidate_ref', 'context_id'):
        require(m[k] == p[k], 'identity mismatch: ' + k)
    require(m['schema_version'] == 1, 'measurement schema')
    require(m['status'] in ('completed', 'failed', 'blocked', 'inconclusive'), 'invalid status')
    require(m['correctness'] in ('passed', 'failed', 'unknown'), 'invalid correctness')
    require(m['comparability'] in ('comparable', 'conditional', 'noncomparable'), 'invalid comparability')
    require(m['evidence'] in ('measured', 'simulated', 'user_report', 'inconclusive'), 'invalid evidence')
    require(isinstance(m['evidence_refs'], list) and all(nonempty(x) for x in m['evidence_refs']), 'invalid evidence references')
    if m['status'] == 'completed':
        require(nonempty(m.get('implementation_ref')) and nonempty(m.get('evaluator_ref')), 'completed measurement needs implementation and evaluator identities')
    require(set(m['metrics']) <= set(p['metrics']), 'unregistered metric')
    eligible = m['status'] == 'completed' and m['correctness'] == 'passed' and m['comparability'] == 'comparable' and bool(m['evidence_refs'])
    rows = {}
    for k, a in p['metrics'].items():
        row = {q: a[q] for q in ('level', 'unit', 'direction', 'statistic', 'kind')}
        row.update({'frozen_baseline': a['baseline'], 'eligible': False, 'reason': 'missing measurement', 'prediction': {q: a[q] for q in ('value', 'low', 'high', 'reason') if q in a}})
        b = m['metrics'].get(k)
        if b is None:
            rows[k] = row
            continue
        for field in ('unit', 'direction', 'level', 'statistic'):
            require(b[field] == a[field], 'metric definition mismatch: ' + k + '/' + field)
        base, actual = number(b['baseline']), number(b['actual'])
        require(base >= 0 and actual >= 0, 'negative measurement')
        ref = a['baseline']
        drift = abs(base - ref) / ref if ref else (0 if base == 0 else None)
        ok = eligible and drift is not None and drift <= a['baseline_tolerance_rel']
        row.update({'baseline': base, 'actual': actual, 'baseline_drift_rel': drift, 'eligible': ok, 'reason': 'comparable' if ok else 'status/correctness/comparability/evidence/baseline drift gate', 'actual_gain': gain(base, actual, a['direction']) if ok else None})
        if ok and a['kind'] in ('point', 'interval') and 'value' in a:
            row['predicted'] = a['value']
            row['value_error'] = actual - a['value']
            row['predicted_gain'] = gain(ref, a['value'], a['direction'])
            if row['actual_gain'] is not None and row['predicted_gain'] is not None:
                row['gain_error'] = row['actual_gain'] - row['predicted_gain']
                if k in p.get('shadow', {}).get('points', {}):
                    sg = gain(ref, p['shadow']['points'][k], a['direction'])
                    row['shadow_gain_error'] = row['actual_gain'] - sg
        if ok and a['kind'] == 'interval':
            row['interval_hit'] = a['low'] <= actual <= a['high']
        if ok and a['kind'] in ('lower_bound', 'upper_bound'):
            row['bound_satisfied'] = actual >= a['value'] if a['kind'] == 'lower_bound' else actual <= a['value']
        rows[k] = row
    return {'project': p['project'], 'round': p['round'], 'theory_version': p['theory_version'], 'shadow_version': p.get('shadow', {}).get('version'), 'validation_role': p['validation_role'], 'context_id': p['context_id'], 'status': m['status'], 'correctness': m['correctness'], 'comparability': m['comparability'], 'evidence': m['evidence'], 'evidence_refs': m['evidence_refs'], 'metrics': rows}


def finish(folder, source):
    p, f = frozen(folder)
    m = read(source)
    result = compare(p, m)
    with locked(folder):
        with newdir(folder / 'result') as tmp:
            write(tmp / 'measurement.json', m)
            write(tmp / 'comparison.json', result)
            lines = ['# 预测与实际对比（自动计算，解释另见 theory-update.md）', '', '证据：' + result['evidence'] + '；状态：' + result['status'], '', '|指标 / 单位|冻结基线|实测基线|预测值/区间/界|实测值|预测收益(%)|实际收益(%)|偏差(pp)|判定|', '|---|---:|---:|---|---:|---:|---:|---:|---|']
            fmt = lambda v: 'unknown' if v is None else f'{100*v:.3f}'
            for k, r in result['metrics'].items():
                pred = r['prediction']
                predicted = ('[' + str(pred['low']) + ', ' + str(pred['high']) + ']') if r['kind'] == 'interval' else (r['kind'] + ': ' + str(pred['value'])) if 'value' in pred else 'unknown'
                judgment = ('interval ' + str(r['interval_hit'])) if 'interval_hit' in r else ('bound ' + str(r['bound_satisfied'])) if 'bound_satisfied' in r else ('comparable' if r['eligible'] else 'ineligible')
                lines.append(f'|{k} / {r["unit"]}|{r["frozen_baseline"]}|{r.get("baseline", "unknown")}|{predicted}|{r.get("actual", "unknown")}|{fmt(r.get("predicted_gain"))}|{fmt(r.get("actual_gain"))}|{fmt(r.get("gain_error"))}|{judgment}|')
            lines.extend(['', '区间覆盖、边界检查、原始量值和漂移见 comparison.json。未通过比较门槛不等于零收益。', ''])
            write(tmp / 'comparison.md', '\n'.join(lines))
            write(tmp / 'receipt.json', {'prediction_sha256': f['prediction_sha256'], 'measurement_sha256': digest(tmp / 'measurement.json'), 'comparison_sha256': digest(tmp / 'comparison.json'), 'finished_at': now()})
    return result


def record(home, folder, note):
    p, f = frozen(folder)
    require(digest(home / 'versions' / p['theory_version'] / 'manifest.json') == f['theory_manifest_sha256'], 'theory manifest changed')
    version(home, p['theory_version'])
    if 'shadow' in p:
        version(home, p['shadow']['version'])
        require(digest(home / 'versions' / p['shadow']['version'] / 'manifest.json') == f['shadow_manifest_sha256'], 'shadow manifest changed')
    r = folder / 'result'
    receipt = read(r / 'receipt.json')
    require(receipt['prediction_sha256'] == f['prediction_sha256'] and digest(r / 'measurement.json') == receipt['measurement_sha256'] and digest(r / 'comparison.json') == receipt['comparison_sha256'], 'result integrity mismatch')
    m = read(r / 'measurement.json')
    result = compare(p, m)
    require(result == read(r / 'comparison.json'), 'comparison mismatch')
    require(result['evidence'] != 'simulated', 'synthetic evidence may not enter user knowledge')
    require(nonempty(Path(note).read_text(encoding='utf-8')), 'theory update note required, including no-change rationale')
    dest = home / 'evidence' / ident(p['project']) / ident(p['round'])
    with locked(home):
        with newdir(dest) as tmp:
            for src in (folder / 'prediction.json', folder / 'freeze.json', r / 'measurement.json', r / 'comparison.json', r / 'receipt.json'):
                shutil.copy2(src, tmp / src.name)
            shutil.copy2(note, tmp / 'theory-update.md')
    return str(dest)


def evidence(home, key):
    parts = key.split('/')
    require(len(parts) == 2, 'round ID must be project/round')
    path = home / 'evidence' / ident(parts[0]) / ident(parts[1])
    p, f = frozen(path)
    r = read(path / 'receipt.json')
    require(r['prediction_sha256'] == f['prediction_sha256'] and digest(path / 'measurement.json') == r['measurement_sha256'] and digest(path / 'comparison.json') == r['comparison_sha256'], 'evidence integrity mismatch')
    result = compare(p, read(path / 'measurement.json'))
    require(result == read(path / 'comparison.json') and (result['project'], result['round']) == tuple(parts), 'evidence mismatch')
    return result


def propose(home, spec):
    s = read(spec)
    name = ident(s['version'])
    parent = active(home)
    require(s['parent'] == parent and nonempty(s['scope']) and nonempty(s['rationale']), 'proposal requires active parent, scope, rationale')
    require(s['category'] in ('calibration', 'correction', 'extension', 'empirical', 'revision'), 'proposal category')
    training = s['training']
    require(isinstance(training, list) and training and len(set(training)) == len(training), 'unique training rounds required')
    for key in training:
        e = evidence(home, key)
        require(e['evidence'] == 'measured', 'promotion proposals need measured training evidence')
    validation = s['validation_rounds']
    require(isinstance(validation, list) and validation and len(set(validation)) == len(validation) and not set(validation) & set(training), 'preregister unique validation rounds disjoint from training')
    for key in validation:
        parts = key.split('/')
        require(len(parts) == 2, 'invalid validation round ID')
        require(not (home / 'evidence' / ident(parts[0]) / ident(parts[1])).exists(), 'validation round already recorded')
    contexts = s['validation_contexts']
    require(isinstance(contexts, list) and contexts and all(nonempty(x) for x in contexts) and len(set(contexts)) == len(contexts), 'validation contexts required')
    require(isinstance(s['changes'], dict) and s['changes'], 'no document changes')
    require(isinstance(s['targets'], dict) and s['targets'], 'validation targets required')
    for k, n in s['targets'].items():
        ident(k)
        require(type(n) is int and 2 <= n <= len(validation), 'at least two independent holdout rounds per metric; cannot exceed preregistered rounds')
    with locked(home):
        require(active(home) == parent, 'active version changed')
        with newdir(home / 'versions' / name) as tmp:
            shutil.copytree(version(home, parent)[0] / 'documents', tmp / 'documents')
            for path, content in s['changes'].items():
                require(nonempty(content), 'empty document change')
                write(tmp / 'documents' / docpath(path), content)
            write(tmp / 'proposal.json', s)
            write(tmp / 'manifest.json', {'version': name, 'parent': parent, 'source_commit': version(home, parent)[1]['source_commit'], 'documents': hashes(tmp / 'documents'), 'created_at': now(), 'training': training, 'proposal': True, 'proposal_sha256': digest(tmp / 'proposal.json')})
            write_index(tmp, read(tmp / 'manifest.json'))
    return name


def promote(home, name, holdout, approval):
    with locked(home):
        path, m = version(home, name)
        require(m.get('proposal') and m['parent'] == active(home) and digest(path / 'proposal.json') == m['proposal_sha256'], 'draft is not an intact child of active theory')
        s = read(path / 'proposal.json')
        require(nonempty(approval), 'explicit independent review/approval reference required')
        require(len(holdout) == len(set(holdout)) and not set(holdout) & set(m['training']), 'holdout overlaps or duplicates training')
        require(set(holdout) == set(s['validation_rounds']), 'must use all preregistered validation rounds')
        rows = [evidence(home, k) for k in holdout]
        require({e['context_id'] for e in rows} == set(s['validation_contexts']), 'validation context coverage mismatch')
        report = {}
        for e in rows:
            require(e['evidence'] == 'measured' and e['validation_role'] == 'holdout' and e['theory_version'] == m['parent'] and e['shadow_version'] == name, 'holdout must be measured with pre-frozen active/shadow pair')
        for key, minimum in s['targets'].items():
            paired = [e['metrics'][key] for e in rows if key in e['metrics'] and 'shadow_gain_error' in e['metrics'][key]]
            require(len(paired) >= minimum, 'insufficient comparable holdout: ' + key)
            require(len({(r['level'], r['unit'], r['direction'], r['statistic']) for r in paired}) == 1, 'incompatible holdout definitions')
            old = statistics.mean(abs(r['gain_error']) for r in paired)
            new = statistics.mean(abs(r['shadow_gain_error']) for r in paired)
            require(new <= old, 'holdout regression: ' + key)
            report[key] = {'count': len(paired), 'old_mae': old, 'new_mae': new}
        require(any(v['new_mae'] < v['old_mae'] for v in report.values()), 'no held-out improvement')
        require(not (home / 'promotions' / (name + '.json')).exists(), 'promotion receipt already exists; inspect before recovery')
        write(home / 'promotions' / (name + '.json'), {'version': name, 'parent': m['parent'], 'holdout': holdout, 'approval_ref': approval, 'metrics': report, 'promoted_at': now()})
        write(home / 'active.json', {'version': name})
    return report


def rollback(home, target, approval):
    require(nonempty(approval), 'rollback requires approval reference')
    with locked(home):
        current = active(home)
        name = version(home, current)[1]['parent']
        visited = {current}
        while name is not None and name not in visited:
            if name == target:
                version(home, name)
                stamp = now().replace(':', '').replace('+', '_')
                write(home / 'rollbacks' / (stamp + '.json'), {'from': current, 'to': target, 'approval_ref': approval, 'at': now()})
                write(home / 'active.json', {'version': target})
                return target
            visited.add(name)
            name = version(home, name)[1]['parent']
        raise ValueError('rollback target must be an ancestor of the active version')


def calibrate(home, metric, keys):
    """Simple robust multiplicative correction; only a proposed fit, not validation."""
    require(len(keys) >= 2 and len(keys) == len(set(keys)), 'need at least two unique fitting rounds')
    rows = []
    for key in keys:
        e = evidence(home, key)
        r = e['metrics'].get(metric, {})
        require(e['evidence'] == 'measured' and e['validation_role'] == 'search' and r.get('eligible') and r.get('predicted', 0) > 0 and r.get('actual', 0) > 0, 'calibration needs comparable measured search points')
        rows.append((e, r))
    identities = {(e['theory_version'], e['context_id'], r['level'], r['unit'], r['direction'], r['statistic']) for e, r in rows}
    require(len(identities) == 1, 'calibration requires identical context and metric definitions')
    factor = statistics.median(r['actual'] / r['predicted'] for _, r in rows)
    require(math.isfinite(factor) and factor > 0, 'nonfinite correction')
    return {'status': 'proposed_fit_not_validated', 'metric': metric, 'identity': list(identities.pop()), 'training': keys, 'multiplicative_factor': factor, 'note': 'Check physical bounds, then validate on unseen rounds. Do not extrapolate automatically.'}


def summary(home):
    groups, counts = {}, {}
    for p in sorted((home / 'evidence').glob('*/*/comparison.json')):
        key = p.parent.parent.name + '/' + p.parent.name
        e = evidence(home, key)
        state = e['evidence'] + '/' + e['status'] + '/' + e['correctness']
        counts[state] = counts.get(state, 0) + 1
        if e['evidence'] != 'measured':
            continue
        for k, r in e['metrics'].items():
            if not r['eligible']:
                continue
            group = (e['theory_version'], e['context_id'], k, r['level'], r['unit'], r['direction'], r['statistic'])
            g = groups.setdefault(group, {'errors': [], 'direction': [], 'coverage': [], 'bounds': []})
            if 'gain_error' in r:
                g['errors'].append(r['gain_error'])
                g['direction'].append((r['predicted_gain'] > 0) == (r['actual_gain'] > 0))
            if 'interval_hit' in r:
                g['coverage'].append(r['interval_hit'])
            if 'bound_satisfied' in r:
                g['bounds'].append(r['bound_satisfied'])
    return {'counts': counts, 'groups': [{'identity': list(key), 'point_n': len(g['errors']), 'mae': statistics.mean(map(abs, g['errors'])) if g['errors'] else None, 'bias': statistics.mean(g['errors']) if g['errors'] else None, 'improvement_class_accuracy': statistics.mean(g['direction']) if g['direction'] else None, 'interval_n': len(g['coverage']), 'interval_coverage': statistics.mean(g['coverage']) if g['coverage'] else None, 'bound_n': len(g['bounds']), 'bound_satisfaction': statistics.mean(g['bounds']) if g['bounds'] else None} for key, g in groups.items()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, default=DEFAULT_HOME)
    subs = parser.add_subparsers(dest='cmd', required=True)
    for cmd, args in {'init': [], 'resolve': [], 'freeze': ['folder', 'source'], 'compare': ['folder', 'source'], 'record': ['folder', 'note'], 'propose': ['spec'], 'promote': ['version', 'approval', 'holdout'], 'summary': [], 'calibrate': ['metric', 'training'], 'rollback': ['version', 'approval']}.items():
        sub = subs.add_parser(cmd)
        for arg in args:
            sub.add_argument(arg, nargs='+' if arg in ('holdout', 'training') else None, type=Path if arg in ('folder', 'source', 'note', 'spec') else str)
    a = parser.parse_args()
    h = a.home.expanduser().resolve()
    try:
        if a.cmd == 'init':
            result = init(h)
        elif a.cmd == 'resolve':
            name = active(h)
            result = {'version': name, 'documents': str(version(h, name)[0] / 'documents'), 'index': str(version(h, name)[0] / 'INDEX.md'), 'home': str(h)}
        elif a.cmd == 'freeze':
            result = freeze(h, a.folder.resolve(), a.source)
        elif a.cmd == 'compare':
            result = finish(a.folder.resolve(), a.source)
        elif a.cmd == 'record':
            result = record(h, a.folder.resolve(), a.note)
        elif a.cmd == 'propose':
            result = propose(h, a.spec)
        elif a.cmd == 'promote':
            result = promote(h, a.version, a.holdout, a.approval)
        elif a.cmd == 'rollback':
            result = rollback(h, a.version, a.approval)
        elif a.cmd == 'calibrate':
            result = calibrate(h, a.metric, a.training)
        else:
            result = summary(h)
        print(dump(result), end='')
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        parser.exit(2, 'error: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
