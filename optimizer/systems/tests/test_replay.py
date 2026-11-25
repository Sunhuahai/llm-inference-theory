"""Control-logic tests; these do not establish optimization effectiveness."""

import copy
from decimal import Decimal, localcontext
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("optimizer_replay", ROOT / "replay.py")
replay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "examples" / "synthetic-search.json").read_text())

    def subset(self, ids):
        data = copy.deepcopy(self.data)
        data["candidates"] = [row for row in data["candidates"] if row["id"] in ids]
        data["outcomes"] = {key: row for key, row in data["outcomes"].items() if key in ids}
        return data

    def test_budget_and_caps_hold_for_all_policies(self):
        for policy in replay.POLICIES:
            for budget in (0, 0.5, 1, 2, 5, 12, 50):
                result = replay.run_replay(self.data, policy=policy, budget=budget, seed=3)
                self.assertLessEqual(result["spent_cost_units"], budget)
                seen = set()
                for event in result["history"]:
                    self.assertNotIn(event["candidate_id"], seen)
                    seen.add(event["candidate_id"])
                    self.assertLessEqual(event["cost_units"], event["cost_cap_units"])
                    self.assertLessEqual(event["cumulative_cost_units"], budget)

    def decimal_fixture(self):
        data = self.subset({"A-fusion"})
        data["baseline"]["setup_cost_units"] = 0.1
        data["candidates"][0]["estimated_cost_units"] = 0.2
        data["candidates"][0]["cost_cap_units"] = 0.2
        data["outcomes"]["A-fusion"]["cost_units"] = 0.2
        return data

    def test_exact_decimal_boundary_is_affordable(self):
        for policy in replay.POLICIES:
            with self.subTest(policy=policy), localcontext() as context:
                context.prec = 3
                result = replay.run_replay(self.decimal_fixture(), policy=policy, budget=0.3)
                self.assertEqual(context.prec, 3)
                self.assertEqual(result["attempted_candidates"], 1)
                self.assertEqual(result["spent_cost_units"], 0.3)
                self.assertEqual(Decimal(result["spent_cost_units_exact"]), Decimal("0.3"))
                self.assertEqual(result["history"][0]["cumulative_cost_units_exact"],
                                 result["spent_cost_units_exact"])

    def test_just_below_budget_does_not_receive_rounding_slack(self):
        budget = math.nextafter(0.3, 0.0)
        result = replay.run_replay(self.decimal_fixture(), budget=budget)
        self.assertEqual(result["attempted_candidates"], 0)
        self.assertLessEqual(Decimal(result["spent_cost_units_exact"]),
                             Decimal(result["budget_cost_units_exact"]))
        data = self.decimal_fixture()
        data["baseline"]["setup_cost_units"] = 0
        data["candidates"][0]["estimated_cost_units"] = 10**16 + 1
        data["candidates"][0]["cost_cap_units"] = 10**16 + 1
        data["outcomes"]["A-fusion"]["cost_units"] = 10**16 + 1
        result = replay.run_replay(data, budget=10**16)
        self.assertEqual(result["attempted_candidates"], 0)

    def test_small_real_overrun_is_censored_without_slack(self):
        data = self.decimal_fixture()
        data["outcomes"]["A-fusion"]["cost_units"] = math.nextafter(0.2, math.inf)
        result = replay.run_replay(data, budget=0.3)
        event = result["history"][0]
        self.assertEqual(event["evaluation_status"], "timeout")
        self.assertEqual(event["cost_units"], 0.2)
        self.assertIsNone(event["observed_latency_ms"])
        self.assertEqual(result["best_candidate_id"], "baseline")
        self.assertEqual(result["spent_cost_units"], 0.3)

    def test_unrepresentable_input_numbers_raise_input_error(self):
        for field in ("predicted_latency_ms", "estimated_cost_units", "cost_cap_units"):
            data = self.decimal_fixture()
            data["candidates"][0][field] = 10**1000
            with self.subTest(field=field), self.assertRaisesRegex(replay.InputError, "finite number"):
                replay.run_replay(data)
        with self.assertRaisesRegex(replay.InputError, "budget.*finite number"):
            replay.run_replay(self.data, budget=10**1000)

    def test_unrepresentable_speedup_is_rejected_when_observed(self):
        data = self.subset({"A-fusion", "E-surprise"})
        data["outcomes"]["E-surprise"]["latency_ms"] = 1e-310
        # E remains unobserved at this budget, so its ratio cannot affect A.
        result = replay.run_replay(data, policy="predicted", budget=4)
        self.assertEqual(result["best_candidate_id"], "A-fusion")
        self.assertEqual(result["attempted_candidates"], 1)
        with self.assertRaisesRegex(replay.InputError, "replay speedup for E-surprise.*finite"):
            replay.run_replay(data, policy="predicted", budget=10)

    def test_unrepresentable_cost_scores_raise_input_error(self):
        data = self.decimal_fixture()
        data["candidates"][0]["estimated_cost_units"] = 1e-310
        with self.assertRaisesRegex(replay.InputError, "cost-aware score.*finite"):
            replay.run_replay(data, exploration=0)
        data = self.decimal_fixture()
        data["baseline"]["latency_ms"] = 1e308
        data["candidates"][0]["uncertainty_ms"] = 1e308
        with self.assertRaisesRegex(replay.InputError, "optimistic gain.*finite"):
            replay.run_replay(data, exploration=0)

    def test_normal_outputs_are_finite_json(self):
        for policy in replay.POLICIES:
            result = replay.run_replay(self.data, policy=policy, budget=12)
            encoded = json.dumps(result, allow_nan=False)
            self.assertEqual(json.loads(encoded), result)

    def test_cli_reports_numeric_errors_without_traceback(self):
        huge_input = self.decimal_fixture()
        huge_input["candidates"][0]["predicted_latency_ms"] = 10**1000
        overflowing_ratio = self.decimal_fixture()
        overflowing_ratio["outcomes"]["A-fusion"]["latency_ms"] = 1e-310
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / "fixture.json"
            for data in (huge_input, overflowing_ratio):
                fixture.write_text(json.dumps(data), encoding="utf-8")
                completed = subprocess.run(
                    [sys.executable, str(ROOT / "replay.py"), str(fixture),
                     "--policy", "predicted", "--budget", "0.3"],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(completed.stdout, "")
                self.assertTrue(completed.stderr.startswith("error: "))
                self.assertNotIn("Traceback", completed.stderr)

    def test_incorrect_and_regressed_candidates_do_not_promote(self):
        data = self.subset({"C-fast-incorrect", "D-regression", "F-build-failure"})
        result = replay.run_replay(data, policy="predicted", budget=20)
        self.assertEqual(result["best_candidate_id"], "baseline")
        self.assertEqual(result["best_replay_latency_ms"], 100)
        self.assertEqual({x["evaluation_status"] for x in result["history"]},
                         {"incorrect", "not_improved", "compile_error"})
        self.assertGreater(result["spent_cost_units"], 1)

    def test_overrun_is_censored_and_charged(self):
        result = replay.run_replay(self.subset({"G-overrun"}), budget=5)
        event = result["history"][0]
        self.assertEqual(event["evaluation_status"], "timeout")
        self.assertIsNone(event["observed_latency_ms"])
        self.assertIsNone(event["correctness_pass"])
        self.assertEqual(event["cost_units"], 4)
        self.assertEqual(result["best_candidate_id"], "baseline")
        self.assertEqual(result["spent_cost_units"], 5)

    def test_zero_budget_and_empty_pool_keep_supplied_baseline(self):
        zero = replay.run_replay(self.data, budget=0)
        self.assertEqual(zero["history"], [])
        self.assertEqual(zero["spent_cost_units"], 0)
        self.assertEqual(zero["stop_reason"], "insufficient_setup_budget")
        empty = replay.run_replay(self.subset(set()), budget=10)
        self.assertEqual(empty["best_candidate_id"], "baseline")
        self.assertEqual(empty["attempted_candidates"], 0)

    def test_selection_api_has_no_future_outcomes(self):
        original = replay.select_candidate
        observed = []

        def audited(candidates, history, incumbent_ms, **kwargs):
            for candidate in candidates:
                self.assertIsInstance(candidate, replay.Candidate)
                self.assertEqual(set(vars(candidate)), replay.PUBLIC_FIELDS)
                self.assertFalse(hasattr(candidate, "outcome"))
            self.assertTrue(all(isinstance(x, replay.Observation) for x in history))
            observed.append((len(history), tuple(x.id for x in candidates)))
            return original(candidates, history, incumbent_ms, **kwargs)

        with patch.object(replay, "select_candidate", side_effect=audited):
            replay.run_replay(self.data, budget=12)
        self.assertGreater(len(observed), 0)

    def test_unseen_outcomes_cannot_change_first_action(self):
        changed = copy.deepcopy(self.data)
        for outcome in changed["outcomes"].values():
            if outcome["status"] == "completed":
                outcome["latency_ms"] *= 7
                outcome["correctness_pass"] = not outcome["correctness_pass"]
            outcome["cost_units"] *= 2
        for policy in replay.POLICIES:
            before = replay.run_replay(self.data, policy=policy, seed=7, budget=12)
            after = replay.run_replay(changed, policy=policy, seed=7, budget=12)
            self.assertEqual(before["history"][0]["candidate_id"], after["history"][0]["candidate_id"])

    def test_invalid_fixtures_are_rejected(self):
        variants = []
        data = copy.deepcopy(self.data)
        data["evidence_origin"] = "measured"
        variants.append(data)
        data = copy.deepcopy(self.data)
        data["candidates"][0]["observed_latency_ms"] = 1
        variants.append(data)
        data = copy.deepcopy(self.data)
        data["outcomes"]["A-fusion"]["latency_ms"] = float("nan")
        variants.append(data)
        data = copy.deepcopy(self.data)
        data["candidates"][0]["parent_id"] = "another-candidate"
        variants.append(data)
        for data in variants:
            with self.assertRaises(replay.InputError):
                replay.run_replay(data)


if __name__ == "__main__":
    unittest.main()
