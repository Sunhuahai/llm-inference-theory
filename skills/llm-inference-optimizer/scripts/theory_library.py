"""Structured theory catalog, immutable revision contracts, and narrow evaluators."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

LAYERS = ('00-definitions', '01-fundamentals', '02-performance-models',
          '03-optimization-theory', '04-autonomous-optimization', '05-theory-evolution')
KINDS = {'definition', 'derived', 'physical_bound', 'performance_model', 'method', 'learning', 'hypothesis'}
UNITS = {'record', 'scalar', 'ratio', 'count', 's', 'ms', 'byte', 'FLOP', 'FLOP/s',
         'byte/s', 'FLOP/byte', '1/s', 's/token', 'ms/token', 'tokens/s', 'requests/s'}
EVALUATOR_IDS = {
    'FND-COMPUTE': 'gemm_flops', 'FND-IO': 'io_bound', 'FND-ROOFLINE': 'roofline',
    'FND-CRITICAL-PATH': 'critical_path', 'FND-AMDAHL': 'amdahl', 'FND-QUEUEING': 'mm1',
    'PERF-OPERATOR': 'operator_time', 'PERF-COMMUNICATION': 'communication_time',
    'PERF-STATE': 'kv_capacity', 'PERF-EXECUTION': 'execution_time',
    'PERF-GENERATION': 'generation_time', 'OPT-SENSITIVITY': 'elasticity',
    'OPT-BENEFIT': 'benefit', 'OPT-COUPLING': 'coupling',
    'LRN-CALIBRATION': 'median_factor', 'LRN-UNCERTAINTY': 'error_summary',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def identifier(value):
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', value) is not None


def read(path):
    def reject(x):
        raise ValueError('nonfinite JSON: ' + x)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def num(value):
    require(type(value) in (int, float) and math.isfinite(value), 'expected finite numeric value')
    return float(value)


def canonical_unit(value):
    return 'byte' if value == 'bytes' else value


def safe_relative(value):
    p = Path(value)
    require(not p.is_absolute() and p.parts and all(identifier(x) for x in p.parts), 'unsafe path: ' + str(value))
    return p


def snapshot_hashes(root):
    require(root.is_dir(), 'structured theory directory missing')
    result = {}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(), 'symlink prohibited in structured snapshot')
        if p.is_file():
            result[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def fields(items, *, parameter=False):
    require(isinstance(items, list), 'fields must be a list')
    result = {}
    for f in items:
        require(isinstance(f, dict) and identifier(f.get('id')) and f['id'] not in result, 'invalid/duplicate field ID')
        require(f.get('unit') in UNITS and text(f.get('description')), 'invalid unit or description')
        if parameter:
            require(f.get('learnable') is True and f['unit'] not in ('record', 'scalar'), 'invalid learnable parameter')
            b = f.get('bounds')
            require(isinstance(b, dict) and 'min' in b, 'parameter must declare numeric bounds')
            require(set(b) <= {'min', 'max', 'exclusive_min', 'exclusive_max'}, 'unknown bound field')
            lo = num(b['min'])
            if 'max' in b:
                hi = num(b['max'])
                require(lo <= hi and (lo < hi or not (b.get('exclusive_min') or b.get('exclusive_max'))), 'invalid parameter bounds')
            for k in ('exclusive_min', 'exclusive_max'):
                require(k not in b or type(b[k]) is bool, 'invalid bound exclusivity')
        result[f['id']] = f
    return result


def bounded(v, f):
    v = num(v)
    b = f['bounds']
    require(v > b['min'] if b.get('exclusive_min') else v >= b['min'], 'parameter below range: ' + f['id'])
    if 'max' in b:
        require(v < b['max'] if b.get('exclusive_max') else v <= b['max'], 'parameter above range: ' + f['id'])
    return v


def closure(modules, ids):
    seen = set()
    def visit(k):
        require(k in modules, 'unknown module: ' + str(k))
        if k not in seen:
            seen.add(k)
            for dep in modules[k]['depends_on']:
                visit(dep)
    for k in ids:
        visit(k)
    return seen


def impact(modules, ids):
    require(set(ids) <= set(modules), 'unknown impacted module')
    return sorted(k for k in modules if set(ids) & closure(modules, [k]))


def load(root, source_root, *, verify_indexes=True):
    root, source_root = Path(root), Path(source_root)
    snapshot_hashes(root)
    modules, paths = {}, {}
    for p in sorted(root.glob('*/*/model.json')):
        d = read(p)
        rel = p.relative_to(root)
        k = d['id']
        require(d.get('schema_version') == 2 and identifier(k) and k not in modules, 'invalid/duplicate model ID')
        require(len(rel.parts) == 3 and rel.parts[0] == d['layer'] and d['layer'] in LAYERS, 'invalid model layer')
        require(identifier(rel.parts[1]) and (p.parent / 'theory.md').is_file(), 'missing theory document')
        require(type(d['revision']) is int and d['revision'] > 0 and d['kind'] in KINDS, 'invalid model revision/kind')
        for key in ('title', 'formula', 'applicability', 'limitations', 'validation', 'status', 'evidence_type'):
            require(text(d.get(key)), 'missing model field: ' + key)
        fi, fo, fp = fields(d['inputs']), fields(d['outputs']), fields(d['parameters'], parameter=True)
        require(fo and not set(fi) & set(fp), 'missing outputs or overlapping inputs/parameters')
        require(d['kind'] not in ('definition', 'derived', 'physical_bound') or not fp, 'base laws cannot have fitted parameters')
        for key in ('tags', 'depends_on', 'sources'):
            require(isinstance(d[key], list) and all(text(x) for x in d[key]) and len(d[key]) == len(set(d[key])), 'invalid ' + key)
        require(d['sources'], 'model must cite original/local theory source')
        for src in d['sources']:
            require(safe_relative(src).suffix == '.md' and (source_root / src).is_file(), 'missing model source: ' + src)
        require(d.get('evaluator') is None or EVALUATOR_IDS.get(k) == d['evaluator'], 'unregistered evaluator')
        require(d['status'] == ('analytic_reference' if d.get('evaluator') else 'documented_model'), 'evaluator status inconsistent')
        modules[k], paths[k] = d, rel.as_posix()
    require(modules and set(x['layer'] for x in modules.values()) == set(LAYERS), 'missing theory layer')
    require(set(root.rglob('model.json')) == {root / p for p in paths.values()}, 'unregistered model path')
    visiting, visited = set(), set()
    def visit(k):
        require(k in modules and k not in visiting, 'missing or cyclic prerequisite: ' + k)
        if k in visited:
            return
        visiting.add(k)
        for dep in modules[k]['depends_on']:
            require(dep in modules, 'missing prerequisite: ' + dep)
            require(LAYERS.index(modules[dep]['layer']) <= LAYERS.index(modules[k]['layer']), 'dependency points to higher layer')
            visit(dep)
        visiting.remove(k)
        visited.add(k)
    for k in modules:
        visit(k)
    r = read(root / 'relations.json')
    require(r['schema_version'] == 2 and isinstance(r['relations'], list), 'invalid relations')
    edges = [{'from': k, 'to': dep, 'type': 'depends_on'} for k in sorted(modules) for dep in sorted(modules[k]['depends_on'])]
    identities = {(e['from'], e['to'], e['type']) for e in edges}
    for e in r['relations']:
        require(set(e) == {'from', 'to', 'type', 'note'} and e['from'] in modules and e['to'] in modules and e['from'] != e['to'], 'invalid relation endpoints')
        key = (e['from'], e['to'], e['type'])
        require(e['type'] in ('feeds', 'constrains', 'extends') and key not in identities and text(e['note']), 'invalid/duplicate relation')
        identities.add(key)
        edges.append(e)
    registry = {'schema_version': 2, 'modules': paths}
    graph = {'schema_version': 2, 'edges': edges}
    if verify_indexes:
        require(read(root / 'registry.json') == registry, 'registry stale')
        stored = read(root / 'dependencies.json')
        require(stored['schema_version'] == 2 and sorted(stored['edges'], key=lambda e: (e['from'], e['to'], e['type'])) == sorted(edges, key=lambda e: (e['from'], e['to'], e['type'])), 'dependency graph stale')
    data = read(root / 'parameters.json')
    require(data['schema_version'] == 2 and isinstance(data['sets'], dict), 'invalid parameter library')
    for name, ps in data['sets'].items():
        require(identifier(name) and ps['model_id'] in modules, 'unknown parameter model')
        m = modules[ps['model_id']]
        require(ps['revision'] == m['revision'], 'stale parameter revision')
        for key in ('context_id', 'scope', 'method'):
            require(text(ps.get(key)), 'parameter set lacks ' + key)
        require(isinstance(ps['evidence'], list) and ps['evidence'] and len(set(ps['evidence'])) == len(ps['evidence']) and all(isinstance(x, str) and len(x.split('/')) == 2 and all(identifier(y) for y in x.split('/')) for x in ps['evidence']), 'invalid parameter evidence')
        definitions = fields(m['parameters'], parameter=True)
        require(isinstance(ps['values'], dict) and ps['values'] and set(ps['values']) <= set(definitions), 'unknown/empty parameter set')
        for k, v in ps['values'].items():
            require(v['unit'] == definitions[k]['unit'], 'parameter unit mismatch')
            bounded(v['value'], definitions[k])
    return modules, registry, graph, data['sets']


def rebuild(root, source_root, write):
    modules, registry, graph, _ = load(root, source_root, verify_indexes=False)
    write(root / 'registry.json', registry)
    write(root / 'dependencies.json', graph)
    lines = ['# 结构化理论模型总索引', '', '[六层图谱](ATLAS.md) · [注册表](registry.json) · [关系图](dependencies.json) · [参数集](parameters.json) · [契约](schemas/README.md)', '', '模型数：' + str(len(modules)) + '；详细原始文档见 [版本索引](../INDEX.md)。', '']
    for layer in LAYERS:
        lines += ['## ' + layer, '', '| ID | 理论 | revision | JSON |', '| --- | --- | ---: | --- |']
        for k, d in sorted(modules.items()):
            if d['layer'] == layer:
                p = registry['modules'][k]
                lines.append(f'| {k} | [{d["title"]}]({p[:-10]}theory.md) | {d["revision"]} | [定义]({p}) |')
        lines.append('')
    write(root / 'INDEX.md', '\n'.join(lines) + '\n')
    return modules


def resolve_ref(modules, sets, ref, unit, context, prediction_kind):
    require(isinstance(ref, dict) and set(ref) <= {'id', 'revision', 'output', 'parameter_sets'}, 'invalid model_ref')
    require(ref['id'] in modules, 'unknown model_ref')
    m = modules[ref['id']]
    require(type(ref['revision']) is int and ref['revision'] == m['revision'], 'model revision mismatch')
    out = fields(m['outputs'])
    require(ref['output'] in out and canonical_unit(unit) == out[ref['output']]['unit'], 'model output/unit mismatch')
    require(out[ref['output']]['unit'] not in ('record', 'scalar'), 'non-numeric model output')
    if prediction_kind in ('lower_bound', 'upper_bound'):
        require(m['kind'] == 'physical_bound' and prediction_kind == 'lower_bound' and ref['output'] == 'time', 'strict bound needs a matching bound output')
    elif prediction_kind != 'unknown':
        require(m['kind'] != 'physical_bound', 'bound is not a point/interval prediction')
    keys = ref.get('parameter_sets', [])
    require(isinstance(keys, list) and len(keys) == len(set(keys)), 'duplicate parameter sets')
    allowed = closure(modules, [m['id']])
    seen = set()
    for name in keys:
        require(name in sets, 'unknown parameter set: ' + name)
        ps = sets[name]
        require(ps['model_id'] in allowed and ps['context_id'] == context and ps['revision'] == modules[ps['model_id']]['revision'], 'parameter set outside model/context/revision')
        for param in ps['values']:
            pair = (ps['model_id'], param)
            require(pair not in seen, 'conflicting parameter sets')
            seen.add(pair)
    return ref


def check_prediction(p, modules, sets, *, shadow=False):
    metrics = p['metrics']
    for k, m in metrics.items():
        if shadow and k not in p['shadow']['points']:
            continue
        ref = p['shadow'].get('model_refs', {}).get(k) if shadow else m.get('model_ref')
        gap = None if shadow else m.get('model_gap')
        require((isinstance(ref, dict) and bool(ref) and gap is None) or (ref is None and text(gap)), 'provide one model_ref or explicit model_gap: ' + k)
        if ref:
            resolve_ref(modules, sets, ref, m['unit'], p['context_id'], 'point' if shadow else m['kind'])


def longest_path(record):
    require(isinstance(record, dict) and isinstance(record['nodes'], list) and record['nodes'], 'DAG nodes required')
    d = {}
    for n in record['nodes']:
        k, v = n['id'], num(n['time_s'])
        require(identifier(k) and k not in d and v >= 0, 'invalid DAG node')
        d[k] = v
    incoming = {k: [] for k in d}
    outgoing = {k: [] for k in d}
    seen = set()
    for edge in record['edges']:
        require(isinstance(edge, list) and len(edge) == 2, 'invalid DAG edge')
        a, b = edge
        require(a in d and b in d and (a, b) not in seen, 'invalid/duplicate DAG edge')
        seen.add((a, b))
        incoming[b].append(a)
        outgoing[a].append(b)
    degree = {k: len(v) for k, v in incoming.items()}
    ready = [k for k in d if degree[k] == 0]
    times = {}
    while ready:
        k = ready.pop()
        times[k] = d[k] + max((times[a] for a in incoming[k]), default=0)
        for v in outgoing[k]:
            degree[v] -= 1
            if degree[v] == 0:
                ready.append(v)
    require(len(times) == len(d), 'cyclic DAG')
    return max(times.values())


def evaluate(modules, sets, model_id, request):
    m = modules[model_id]
    ev = m.get('evaluator')
    require(ev is not None and EVALUATOR_IDS.get(model_id) == ev, 'no executable evaluator for this module')
    fi, fp, fo = fields(m['inputs']), fields(m['parameters'], parameter=True), fields(m['outputs'])
    inputs = request['inputs']
    require(set(inputs) == set(fi), 'input names do not match contract')
    x = {}
    for k, v in inputs.items():
        require(v['unit'] == fi[k]['unit'], 'input unit mismatch: ' + k)
        x[k] = v['value'] if fi[k]['unit'] == 'record' else num(v['value'])
    name = request.get('parameter_set')
    params = request.get('parameters', {})
    if name is not None:
        require(name in sets and not params, 'use explicit parameters or one parameter set')
        ps = sets[name]
        require(ps['model_id'] == model_id and ps['revision'] == m['revision'] and ps['context_id'] == request.get('context_id'), 'parameter set mismatch')
        params = ps['values']
    require(set(params) == set(fp), 'required parameters missing or unexpected')
    for k, v in params.items():
        require(v['unit'] == fp[k]['unit'], 'parameter unit mismatch')
        x[k] = bounded(v['value'], fp[k])
    def pos(*keys):
        require(all(x[k] > 0 for k in keys), 'expected positive: ' + ','.join(keys))
    def nonneg(*keys):
        require(all(x[k] >= 0 for k in keys), 'expected nonnegative: ' + ','.join(keys))
    def integral(*keys):
        require(all(x[k] >= 0 and x[k].is_integer() for k in keys), 'expected integer count')
    if ev == 'gemm_flops':
        integral('m', 'n', 'k')
        out = {'flops': 2*x['m']*x['n']*x['k']}
    elif ev == 'io_bound':
        nonneg('bytes'); pos('bandwidth')
        out = {'time': x['bytes']/x['bandwidth']}
    elif ev == 'roofline':
        nonneg('flops', 'bytes'); pos('peak_flops', 'peak_bandwidth')
        out = {'time': max(x['flops']/x['peak_flops'], x['bytes']/x['peak_bandwidth']), 'intensity': x['flops']/x['bytes'] if x['bytes'] else None}
    elif ev in ('critical_path', 'execution_time'):
        t = longest_path(x['dag'])
        out = {'time': t}
        if ev == 'execution_time':
            out['time_ms'] = 1000*t
    elif ev == 'amdahl':
        pos('speedup'); require(0 <= x['fraction'] <= 1, 'fraction outside [0,1]')
        out = {'speedup': 1/(1-x['fraction']+x['fraction']/x['speedup'])}
    elif ev == 'mm1':
        nonneg('arrival_rate'); require(x['service_rate'] > x['arrival_rate'], 'unstable queue')
        out = {'mean_time': 1/(x['service_rate']-x['arrival_rate'])}
    elif ev == 'operator_time':
        nonneg('flops', 'bytes'); pos('effective_compute', 'effective_bandwidth')
        t = max(x['flops']/x['effective_compute'], x['bytes']/x['effective_bandwidth'])+x['exposed_overhead']
        out = {'time': t, 'time_ms': 1000*t}
    elif ev == 'communication_time':
        integral('messages'); nonneg('bytes'); pos('bandwidth')
        t = x['messages']*x['startup']+x['bytes']/x['bandwidth']
        out = {'time': t, 'time_ms': 1000*t}
    elif ev == 'kv_capacity':
        integral('layers', 'batch', 'length', 'kv_heads', 'head_dim'); pos('element_bytes')
        out = {'state': 2*math.prod(x.values())}
    elif ev == 'generation_time':
        nonneg('round_time'); pos('committed')
        t = x['round_time']/x['committed']
        out = {'tpot': t, 'tpot_ms': 1000*t}
    elif ev == 'elasticity':
        pos('baseline_time', 'perturbed_time', 'resource_ratio')
        require(x['resource_ratio'] != 1, 'resource ratio must differ from 1')
        out = {'elasticity': (math.log(x['perturbed_time'])-math.log(x['baseline_time']))/math.log(x['resource_ratio'])}
    elif ev == 'benefit':
        pos('baseline_time', 'candidate_time')
        out = {'gain': 1-x['candidate_time']/x['baseline_time'], 'speedup': x['baseline_time']/x['candidate_time']}
    elif ev == 'coupling':
        require(text(request.get('objective_unit')), 'scalar objective_unit required')
        out = {'interaction': x['jab']-x['ja']-x['jb']+x['j0'], 'conditional_b': x['jab']-x['ja']}
    elif ev == 'median_factor':
        data = x['paired_training']
        require(text(data['unit']) and len(data['pairs']) >= 2, 'need at least two same-unit pairs')
        vals = []
        for pair in data['pairs']:
            a, b = num(pair['actual']), num(pair['predicted'])
            require(a > 0 and b > 0, 'positive pairs required')
            vals.append(a/b)
        out = {'factor': statistics.median(vals)}
    elif ev == 'error_summary':
        values = [num(v) for v in x['gain_errors']]
        require(values, 'empty errors')
        out = {'mae': statistics.mean(map(abs, values)), 'bias': statistics.mean(values)}
    else:
        raise ValueError('unimplemented evaluator')
    require(set(out) <= set(fo), 'evaluator output contract mismatch')
    for v in out.values():
        if v is not None:
            num(v)
    return {'model_id': model_id, 'revision': m['revision'], 'kind': m['kind'], 'parameter_set': name, 'context_id': request.get('context_id'), 'output': {k: {'value': v, 'unit': fo[k]['unit']} for k, v in out.items()}, 'objective_unit': request.get('objective_unit'), 'status': 'conditional_calculation_not_measurement', 'applicability': m['applicability'], 'limitations': m['limitations']}
