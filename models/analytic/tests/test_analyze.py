import copy
import importlib.util
import json
from pathlib import Path
import unittest

MODULE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analytic_reference", MODULE / "analyze.py")
analytic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analytic)


class AnalyticTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((MODULE / "examples" / "synthetic-system.json").read_text())

    def test_bottleneck_moves_and_joint_graph_is_recomputed(self):
        result = analytic.analyze(self.data)
        first, second, third = result["scenarios"]
        self.assertEqual([s["time_ms"] for s in result["scenarios"]], [12, 6, 5])
        self.assertEqual(first["critical_path"], ["attention", "transfer"])
        self.assertEqual(third["critical_path"], ["attention", "projection"])
        self.assertEqual(first["nodes"][0]["limiting_resources_in_model"], ["device.hbm"])
        self.assertEqual(second["nodes"][0]["limiting_resources_in_model"], ["device.compute"])
        self.assertEqual(second["bound_ratio_not_speedup"], 2)
        self.assertNotIn("speedup", second)

    def test_same_resource_parallelism_is_not_free(self):
        scenario = self.data["scenarios"][0]
        scenario["serialization_orders"] = []
        scenario["nodes"][1]["after"] = []
        with self.assertRaisesRegex(analytic.InputError, "unordered users"):
            analytic.analyze(self.data)
        scenario["graph_semantics"] = "dependencies_only"
        result = analytic.analyze(self.data)["scenarios"][0]
        self.assertTrue(result["unordered_shared_resources"])
        self.assertEqual(result["value_kind"], "resource_dependency_lower_bound")

    def test_aggregate_shared_resource_bound_can_dominate_path(self):
        scenario = self.data["scenarios"][0]
        scenario["graph_semantics"] = "dependencies_only"
        scenario["serialization_orders"] = []
        scenario["nodes"] = [
            {"id": n, "after": [], "demands": {"device.compute": {"amount": 4e9, "unit": "FLOP"}}}
            for n in ("left", "right")
        ]
        result = analytic.analyze(self.data)["scenarios"][0]
        self.assertEqual(result["critical_path_time_ms"], 4)
        self.assertEqual(result["time_ms"], 8)

    def test_effective_rate_cannot_be_promoted_to_physical_bound(self):
        self.data["resources"]["device.hbm"]["rate_kind"] = "effective_forecast"
        with self.assertRaisesRegex(analytic.InputError, "do not mix rate semantics"):
            analytic.analyze(self.data)
        self.data["mode"] = "forecast"
        for rate in self.data["resources"].values():
            rate["rate_kind"] = "effective_forecast"
        result = analytic.analyze(self.data)
        self.assertEqual(result["scenarios"][0]["value_kind"], "fixed_schedule_model_estimate")
        self.assertNotIn("bound_ratio_not_speedup", result["scenarios"][0])

    def test_cycles_missing_nodes_duplicates_and_units_fail(self):
        changes = [
            lambda d: d["scenarios"][0]["nodes"][0].update(after=["projection"]),
            lambda d: d["scenarios"][0]["nodes"][0].update(after=["missing"]),
            lambda d: d["scenarios"][0]["nodes"][1].update(id="attention"),
            lambda d: d["scenarios"][0]["nodes"][0]["demands"]["device.hbm"].update(unit="GiB"),
            lambda d: d["resources"]["device.hbm"].update(rate_unit="byte/ms"),
            lambda d: d["resources"]["device.hbm"].update(rate=float("nan")),
            lambda d: d["resources"]["device.hbm"].update(rate=True),
            lambda d: d["resources"]["device.hbm"].update(rate=1e-310),
            lambda d: d.update(schema_version=1.0),
        ]
        for change in changes:
            value = copy.deepcopy(self.data)
            change(value)
            with self.subTest(change=change), self.assertRaises(analytic.InputError):
                analytic.analyze(value)

    def test_infeasible_candidate_is_not_ranked_by_ratio(self):
        self.data["scenarios"][1]["memory"]["required_peak_bytes"] = 9e9
        result = analytic.analyze(self.data)["scenarios"][1]
        self.assertFalse(result["capacity_feasible_from_supplied_values"])
        self.assertIsNone(result["bound_ratio_not_speedup"])

    def test_observation_checks_preserve_origin_and_split(self):
        self.data["observations"] = [{"scenario_id": "baseline", "duration_ms": 11,
                                     "split": "holdout", "origin": "synthetic",
                                     "artifact_ref": "SYNTHETIC test fixture"}]
        check = analytic.analyze(self.data)["observation_checks"][0]
        self.assertTrue(check["observation_below_supplied_bound"])
        self.assertEqual(check["origin"], "synthetic")
        self.assertEqual(check["split"], "holdout")
        self.data["mode"] = "forecast"
        for rate in self.data["resources"].values():
            rate["rate_kind"] = "effective_forecast"
        check = analytic.analyze(self.data)["observation_checks"][0]
        self.assertAlmostEqual(check["signed_relative_error"], 1 / 11)

    def test_derived_ratios_reject_overflow_but_allow_negative_errors(self):
        data = copy.deepcopy(self.data)
        data["resources"] = {"device.compute": {
            "rate": 1000, "work_unit": "FLOP", "rate_unit": "FLOP/s",
            "rate_kind": "physical_upper", "source": "SYNTHETIC numeric boundary",
        }}
        data["scenarios"] = data["scenarios"][:2]
        for scenario, amount in zip(data["scenarios"], (1e300, 1e-300)):
            scenario["serialization_orders"] = []
            scenario["nodes"] = [{"id": "single", "after": [], "demands": {
                "device.compute": {"amount": amount, "unit": "FLOP"},
            }}]
        for mode, rate_kind in (("bound", "physical_upper"), ("forecast", "effective_forecast")):
            data["mode"] = mode
            data["resources"]["device.compute"]["rate_kind"] = rate_kind
            with self.subTest(mode=mode), self.assertRaisesRegex(analytic.InputError, "ratio.*finite"):
                analytic.analyze(data)

        # Each node duration is finite; only the comparison can overflow.
        data["scenarios"] = data["scenarios"][:1]
        data["observations"] = [{"scenario_id": "baseline", "duration_ms": 1e-300,
                                 "split": "holdout", "origin": "synthetic",
                                 "artifact_ref": "SYNTHETIC numeric boundary"}]
        with self.assertRaisesRegex(analytic.InputError, "signed_relative_error.*finite"):
            analytic.analyze(data)
        data["observations"][0]["duration_ms"] = 2e300
        result = analytic.analyze(data)
        self.assertAlmostEqual(result["observation_checks"][0]["signed_relative_error"], -0.5)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
