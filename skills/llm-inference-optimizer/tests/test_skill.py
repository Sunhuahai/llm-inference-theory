import importlib.util
from pathlib import Path
import tempfile
import shutil
import copy
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('skill', Path(__file__).resolve().parents[1] / 'scripts/skill.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home = self.root / 'home'
        self.old = s.init(self.home)
        self.metric = {'level': 'service', 'unit': 'ms/token', 'direction': 'lower', 'statistic': 'mean', 'baseline': 10, 'baseline_tolerance_rel': 0.01, 'kind': 'point', 'value': 8, 'model_ref': {'id': 'PERF-SERVING', 'revision': 1, 'output': 'tpot'}}

    def round(self, name, *, shadow=None, shadow_ref=None, actual=9, baseline=10, status='completed', evidence='measured', record=True):
        folder = self.root / 'rounds' / name
        p = {'schema_version': 1, 'project': 'fixture', 'round': name, 'theory_version': self.old, 'parent_ref': 'base', 'candidate_ref': name, 'context_id': 'ctx', 'mechanism': 'fixture', 'assumptions': 'fixture', 'validation_plan': 'fixture', 'theory_refs': ['theory/INDEX.md'], 'validation_role': 'holdout' if shadow else 'search', 'metrics': {'tpot': dict(self.metric)}}
        if shadow:
            p['shadow'] = {'version': shadow, 'points': {'tpot': 9}, 'model_refs': {'tpot': shadow_ref or {'id': 'PERF-SERVING', 'revision': 2, 'output': 'tpot'}}}
        source = self.root / 'input.json'
        s.write(source, p)
        s.freeze(self.home, folder, source)
        m = {k: p[k] for k in ('schema_version', 'project', 'round', 'parent_ref', 'candidate_ref', 'context_id')}
        m.update({'status': status, 'correctness': 'passed' if status == 'completed' else 'unknown', 'comparability': 'comparable', 'evidence': evidence, 'evidence_refs': ['fixture-log'], 'implementation_ref': 'commit-fixture', 'evaluator_ref': 'fixture-independent', 'metrics': {'tpot': {**{k: self.metric[k] for k in ('level', 'unit', 'direction', 'statistic')}, 'baseline': baseline, 'actual': actual}}})
        s.write(source, m)
        result = s.finish(folder, source)
        if record:
            note = self.root / 'note.md'
            s.write(note, '# fixture no change\n')
            review = s.read(folder / 'result/module-review-template.json')
            for d in review['decisions']:
                d.update(decision='keep', rationale='fixture only')
            s.write(folder / 'theory-update.json', review)
            s.record(self.home, folder, note)
        return folder, result

    def proposal(self):
        spec = self.root / 'proposal.json'
        model = copy.deepcopy(s.version_library(self.home, self.old)[0]['PERF-SERVING'])
        model['revision'] = 2
        model['limitations'] += ' Fixture correction only.'
        s.write(spec, {'version': 'draft', 'parent': self.old, 'scope': 'fixture ctx only', 'category': 'correction', 'rationale': 'fixture', 'training': ['fixture/r1', 'fixture/r2'], 'targets': {'tpot': 2}, 'validation_rounds': ['fixture/r3', 'fixture/r4'], 'validation_contexts': ['ctx'], 'changes': {'theory/calibrated.md': '# fixture'}, 'structured_changes': {'02-performance-models/serving/model.json': model, '02-performance-models/serving/theory.md': '# fixture r2'}})
        return s.propose(self.home, spec)

    def test_lifecycle(self):
        _, r = self.round('r1')
        self.assertAlmostEqual(r['metrics']['tpot']['gain_error'], -0.1)
        self.round('r2')
        self.assertAlmostEqual(s.calibrate(self.home, 'tpot', ['fixture/r1', 'fixture/r2'])['multiplicative_factor'], 1.125)
        self.assertEqual(s.init(self.home), self.old)
        name = self.proposal()
        self.round('r3', shadow=name)
        self.round('r4', shadow=name)
        with self.assertRaises(ValueError):
            s.promote(self.home, name, ['fixture/r1', 'fixture/r3'], 'fixture-review')
        with self.assertRaises(ValueError):
            s.promote(self.home, name, ['fixture/r3'], 'fixture-review')
        r = s.promote(self.home, name, ['fixture/r3', 'fixture/r4'], 'fixture-review')
        self.assertEqual(r['tpot']['new_mae'], 0)
        self.assertEqual(s.active(self.home), name)
        self.assertEqual(s.init(self.home), name)
        self.assertEqual(s.summary(self.home)['groups'][0]['point_n'], 4)
        self.assertEqual(s.module_summary(self.home)['groups'][0]['point_n'], 4)
        self.assertIn('PERF-SERVING', s.version(self.home, name)[1]['changed_modules'])
        self.assertTrue((self.home / 'versions' / name / 'INDEX.md').exists())
        self.assertEqual(s.rollback(self.home, self.old, 'fixture-rollback'), self.old)
        self.assertEqual(s.active(self.home), self.old)

    def test_failures_and_drift(self):
        _, r = self.round('fail', status='failed')
        self.assertFalse(r['metrics']['tpot']['eligible'])
        _, r = self.round('drift', baseline=12)
        self.assertFalse(r['metrics']['tpot']['eligible'])
        self.assertEqual(s.summary(self.home)['groups'], [])

    def test_synthetic_and_frozen(self):
        f, _ = self.round('synthetic', evidence='simulated', record=False)
        note = self.root / 'note.md'
        s.write(note, 'fixture')
        with self.assertRaises(ValueError):
            s.record(self.home, f, note)
        p = s.read(f / 'prediction.json')
        p['metrics']['tpot']['value'] = 9
        s.write(f / 'prediction.json', p)
        with self.assertRaises(ValueError):
            s.frozen(f)

    def test_bound_and_interval_separate(self):
        old_ref = self.metric.pop('model_ref')
        self.metric.update(kind='lower_bound', value=8, model_gap='fixture: no corresponding strict bound module')
        _, r = self.round('bound')
        self.assertTrue(r['metrics']['tpot']['bound_satisfied'])
        self.assertNotIn('gain_error', r['metrics']['tpot'])
        self.metric.pop('model_gap')
        self.metric.update(kind='interval', low=7, high=8, model_ref=old_ref)
        _, r = self.round('interval')
        self.assertFalse(r['metrics']['tpot']['interval_hit'])
        group = s.summary(self.home)['groups'][0]
        self.assertEqual(group['bound_n'], 1)
        self.assertEqual(group['interval_n'], 1)
        self.assertEqual(group['point_n'], 1)

    def test_duplicate_and_mismatch(self):
        folder, _ = self.round('dup')
        note = self.root / 'note.md'
        with self.assertRaises(ValueError):
            s.record(self.home, folder, note)
        p, _ = s.frozen(folder)
        m = s.read(folder / 'result/measurement.json')
        m['context_id'] = 'other'
        with self.assertRaises(ValueError):
            s.compare(p, m)

    def test_structured_seed_and_revision_integrity(self):
        modules, registry, graph, sets = s.version_library(self.home, self.old)
        self.assertEqual(len(modules), 33)
        self.assertEqual(len(graph['edges']), 91)
        self.assertEqual(sets, {})
        self.assertIn('PERF-SERVING', s.tl.impact(modules, ['PERF-OPERATOR']))
        self.assertEqual(s.theory_command(self.home, 'catalog', SimpleNamespace(query='roofline'))['models'][0]['id'], 'FND-ROOFLINE')
        p = self.home / 'versions' / self.old / 'structured' / registry['modules']['FND-ROOFLINE']
        d = s.read(p)
        d['revision'] = 99
        s.write(p, d)
        with self.assertRaises(ValueError):
            s.active(self.home)

    def test_invalid_reference_and_cycle(self):
        modules, registry, _, sets = s.version_library(self.home, self.old)
        with self.assertRaises(ValueError):
            s.tl.resolve_ref(modules, sets, {'id': 'PERF-SERVING', 'revision': 2, 'output': 'tpot'}, 'ms/token', 'ctx', 'point')
        with self.assertRaises(ValueError):
            s.tl.resolve_ref(modules, sets, {'id': 'PERF-SERVING', 'revision': 1, 'output': 'tpot'}, 'ms', 'ctx', 'point')
        with self.assertRaises(ValueError):
            s.tl.resolve_ref(modules, sets, {'id': 'FND-ROOFLINE', 'revision': 1, 'output': 'time'}, 's', 'ctx', 'point')
        target = self.root / 'bad-graph'
        shutil.copytree(self.home / 'versions' / self.old / 'structured', target)
        model_path = target / registry['modules']['FND-ROOFLINE']
        d = s.read(model_path)
        d['depends_on'].append('FND-ROOFLINE')
        s.write(model_path, d)
        with self.assertRaises(ValueError):
            s.tl.load(target, self.home / 'versions' / self.old / 'documents', verify_indexes=False)

    def test_parameter_candidate_and_holdout(self):
        self.metric.update(level='execution', unit='ms', model_ref={'id': 'PERF-OPERATOR', 'revision': 1, 'output': 'time_ms'})
        self.round('r1')
        self.round('r2')
        ps = {'model_id': 'PERF-OPERATOR', 'revision': 1, 'context_id': 'ctx', 'scope': 'fixture only', 'method': 'fixture independent overhead measurement', 'values': {'exposed_overhead': {'value': 0.001, 'unit': 's'}}, 'evidence': ['fixture/r1', 'fixture/r2']}
        proposal = {'version': 'parameter-draft', 'parent': self.old, 'scope': 'fixture', 'category': 'calibration', 'rationale': 'fixture only', 'training': ps['evidence'], 'targets': {'tpot': 2}, 'validation_rounds': ['fixture/r3', 'fixture/r4'], 'validation_contexts': ['ctx'], 'parameter_sets': {'fixture-overhead': ps}}
        file = self.root / 'proposal.json'
        s.write(file, proposal)
        name = s.propose(self.home, file)
        mods, _, _, sets = s.version_library(self.home, name)
        ref = {'id': 'PERF-OPERATOR', 'revision': 1, 'output': 'time_ms', 'parameter_sets': ['fixture-overhead']}
        with self.assertRaises(ValueError):
            s.tl.resolve_ref(mods, sets, ref, 'ms', 'other', 'point')
        self.round('r3', shadow=name, shadow_ref=ref)
        self.round('r4', shadow=name, shadow_ref=ref)
        s.promote(self.home, name, ['fixture/r3', 'fixture/r4'], 'fixture-review')
        self.assertEqual(s.active(self.home), name)
        self.assertEqual(s.version_library(self.home, name)[3]['fixture-overhead']['values']['exposed_overhead']['value'], 0.001)

    def test_legacy_migration(self):
        home = self.root / 'legacy'
        old_name = 'seed-1b4c5a377634'
        dest = home / 'versions' / old_name
        shutil.copytree(s.SKILL / 'knowledge/initial', dest / 'documents')
        s.write(dest / 'documents/theory/private-example.md', '# local legacy note')
        s.write(dest / 'manifest.json', {'version': old_name, 'parent': None, 'source_commit': '1b4c5a37763475eb39b31dd083fcad256fc2578e', 'documents': s.hashes(dest / 'documents'), 'created_at': s.now(), 'training': []})
        s.write(home / 'active.json', {'version': old_name})
        old_digest = s.digest(dest / 'manifest.json')
        self.assertEqual(s.init(home), old_name)
        info = s.migrate(home, 'migrated-v2', 'fixture-review')
        self.assertIn('theory/private-example.md', info['legacy_document_differences'])
        self.assertEqual(s.digest(dest / 'manifest.json'), old_digest)
        self.assertTrue((home / 'versions/migrated-v2/documents/theory/private-example.md').is_file())
        self.assertEqual(len(s.version_library(home, 'migrated-v2')[0]), 33)
        self.assertEqual(s.rollback(home, old_name, 'fixture-rollback'), old_name)

    def test_all_analytic_references(self):
        modules, _, _, sets = s.version_library(self.home, self.old)
        dag = {'nodes': [{'id': 'a', 'time_s': 2}, {'id': 'b', 'time_s': 3}, {'id': 'c', 'time_s': 4}], 'edges': [['a', 'b'], ['a', 'c']]}
        fixtures = [
            ('FND-COMPUTE', {'m': 2, 'n': 3, 'k': 4}, {}, 'flops', 48),
            ('FND-IO', {'bytes': 100, 'bandwidth': 50}, {}, 'time', 2),
            ('FND-ROOFLINE', {'flops': 100, 'bytes': 200, 'peak_flops': 100, 'peak_bandwidth': 100}, {}, 'time', 2),
            ('FND-CRITICAL-PATH', {'dag': dag}, {}, 'time', 6),
            ('FND-AMDAHL', {'fraction': .5, 'speedup': 2}, {}, 'speedup', 4/3),
            ('FND-QUEUEING', {'arrival_rate': 2, 'service_rate': 3}, {}, 'mean_time', 1),
            ('PERF-OPERATOR', {'flops': 100, 'bytes': 200, 'effective_compute': 100, 'effective_bandwidth': 100}, {'exposed_overhead': .1}, 'time', 2.1),
            ('PERF-COMMUNICATION', {'messages': 2, 'bytes': 100, 'bandwidth': 100}, {'startup': .1}, 'time', 1.2),
            ('PERF-STATE', {'layers': 2, 'batch': 1, 'length': 10, 'kv_heads': 2, 'head_dim': 4, 'element_bytes': 2}, {}, 'state', 640),
            ('PERF-EXECUTION', {'dag': dag}, {}, 'time_ms', 6000),
            ('PERF-GENERATION', {'round_time': .1, 'committed': 2}, {}, 'tpot_ms', 50),
            ('OPT-SENSITIVITY', {'baseline_time': 1, 'perturbed_time': 2, 'resource_ratio': 2}, {}, 'elasticity', 1),
            ('OPT-BENEFIT', {'baseline_time': 10, 'candidate_time': 8}, {}, 'gain', .2),
            ('OPT-COUPLING', {'j0': 0, 'ja': 1, 'jb': 2, 'jab': 4}, {}, 'interaction', 1),
            ('LRN-CALIBRATION', {'paired_training': {'unit': 'ms', 'pairs': [{'actual': 2, 'predicted': 1}, {'actual': 6, 'predicted': 3}]}}, {}, 'factor', 2),
            ('LRN-UNCERTAINTY', {'gain_errors': [-.1, .3]}, {}, 'mae', .2),
        ]
        self.assertEqual({f[0] for f in fixtures}, set(s.tl.EVALUATOR_IDS))
        for k, values, parameters, output, expected in fixtures:
            with self.subTest(model=k):
                m = modules[k]
                request = {'inputs': {f['id']: {'value': values[f['id']], 'unit': f['unit']} for f in m['inputs']}, 'parameters': {f['id']: {'value': parameters[f['id']], 'unit': f['unit']} for f in m['parameters']}, 'objective_unit': 'objective'}
                self.assertAlmostEqual(s.tl.evaluate(modules, sets, k, request)['output'][output]['value'], expected)
        with self.assertRaises(ValueError):
            s.tl.longest_path({'nodes': dag['nodes'], 'edges': [['a', 'b'], ['b', 'a']]})
        with self.assertRaises(ValueError):
            s.tl.evaluate(modules, sets, 'FND-IO', {'inputs': {'bytes': {'value': 1, 'unit': 'FLOP'}, 'bandwidth': {'value': 1, 'unit': 'byte/s'}}})


if __name__ == '__main__':
    unittest.main()
