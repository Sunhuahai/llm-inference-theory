import importlib.util
from pathlib import Path
import tempfile
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
        self.metric = {'level': 'service', 'unit': 'ms', 'direction': 'lower', 'statistic': 'mean', 'baseline': 10, 'baseline_tolerance_rel': 0.01, 'kind': 'point', 'value': 8}

    def round(self, name, *, shadow=None, actual=9, baseline=10, status='completed', evidence='measured', record=True):
        folder = self.root / 'rounds' / name
        p = {'schema_version': 1, 'project': 'fixture', 'round': name, 'theory_version': self.old, 'parent_ref': 'base', 'candidate_ref': name, 'context_id': 'ctx', 'mechanism': 'fixture', 'assumptions': 'fixture', 'validation_plan': 'fixture', 'theory_refs': ['theory/INDEX.md'], 'validation_role': 'holdout' if shadow else 'search', 'metrics': {'tpot': dict(self.metric)}}
        if shadow:
            p['shadow'] = {'version': shadow, 'points': {'tpot': 9}}
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
            s.record(self.home, folder, note)
        return folder, result

    def proposal(self):
        spec = self.root / 'proposal.json'
        s.write(spec, {'version': 'draft', 'parent': self.old, 'scope': 'fixture ctx only', 'category': 'calibration', 'rationale': 'fixture', 'training': ['fixture/r1', 'fixture/r2'], 'targets': {'tpot': 2}, 'validation_rounds': ['fixture/r3', 'fixture/r4'], 'validation_contexts': ['ctx'], 'changes': {'theory/calibrated.md': '# fixture'}})
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
        self.metric.update(kind='lower_bound', value=8)
        _, r = self.round('bound')
        self.assertTrue(r['metrics']['tpot']['bound_satisfied'])
        self.assertNotIn('gain_error', r['metrics']['tpot'])
        self.metric.update(kind='interval', low=7, high=8)
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


if __name__ == '__main__':
    unittest.main()
