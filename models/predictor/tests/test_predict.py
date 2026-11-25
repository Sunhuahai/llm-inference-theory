import copy
import json
import sys
import unittest
from pathlib import Path

PREDICTOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PREDICTOR))

from model import InputError, expected_speculative_tokens, predict  # noqa: E402
from predict import as_markdown  # noqa: E402


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(
            (PREDICTOR / "examples" / "synthetic-baseline.json").read_text(encoding="utf-8")
        )

    def one(self):
        inp = copy.deepcopy(self.fixture)
        inp["scenarios"] = [{"name": "unchanged", "resource_demand_factors": {
            k: 1.0 for k in inp["baseline"]["resource_busy_fraction"]
        }}]
        return inp

    def test_identity_reproduces_calibrated_baseline(self):
        r = predict(self.one())
        s = r["scenarios"][0]
        self.assertAlmostEqual(s["ttft_mean_ms"], 150)
        self.assertAlmostEqual(s["tpot_mean_ms"], 22)
        self.assertAlmostEqual(s["saturated_output_tps"], 1800)
        self.assertAlmostEqual(s["capacity_speedup"], 1)

    def test_amdahl_only_partial_stage_improvement(self):
        inp = self.one()
        inp["scenarios"][0]["effects"] = [
            {"stage": "decode", "block": "qsa", "compute_speedup": 2}
        ]
        s = predict(inp)["scenarios"][0]
        self.assertAlmostEqual(s["tpot_mean_ms"], 18.5)
        self.assertAlmostEqual(s["ttft_mean_ms"], 150)

    def test_floor_caps_speedup_and_warns(self):
        inp = self.one()
        inp["scenarios"][0]["effects"] = [
            {"stage": "decode", "block": "qsa", "compute_speedup": 100}
        ]
        s = predict(inp)["scenarios"][0]
        self.assertAlmostEqual(s["tpot_mean_ms"], 18)
        self.assertTrue(any("情景时间下限" in hit for hit in s["floor_hits"]))

    def test_new_overhead_is_absolute_ms_not_calibrated(self):
        inp = self.one()
        inp["scenarios"][0]["effects"] = [
            {"stage": "prefill", "block": "host", "added_ms": 3}
        ]
        s = predict(inp)["scenarios"][0]
        self.assertAlmostEqual(s["ttft_mean_ms"], 153)

    def test_overlap_saves_no_more_than_shorter_lane(self):
        inp = self.one()
        inp["scenarios"][0]["effects"] = [
            {"stage": "prefill", "block": "qsa", "overlap_fraction": 1}
        ]
        s = predict(inp)["scenarios"][0]
        # Old qsa = 37 + 6 - 0.5*6 = 40; new qsa = 37.
        self.assertAlmostEqual(150 - s["ttft_mean_ms"], 3 * 100 / 99)

    def test_speculative_expected_tokens_edge_cases(self):
        self.assertAlmostEqual(expected_speculative_tokens(3, 0), 1)
        self.assertAlmostEqual(expected_speculative_tokens(3, 1), 4)
        self.assertAlmostEqual(expected_speculative_tokens(2, 0.5), 1.75)

    def test_mtp_tpot_not_capacity(self):
        s = next(x for x in predict(self.fixture)["scenarios"] if x["name"].startswith("T009"))
        self.assertLess(s["tpot_mean_ms"], 22)
        self.assertLess(s["saturated_output_tps"], 1800)
        self.assertAlmostEqual(s["expected_tokens_per_verification"], 2.533)

    def test_arrival_limited_throughput_does_not_increase(self):
        r = predict(self.fixture)
        for s in r["scenarios"]:
            self.assertAlmostEqual(s["offered_output_tps"], 1024)

    def test_missing_resource_factors_withholds_capacity(self):
        inp = self.one()
        del inp["scenarios"][0]["resource_demand_factors"]
        s = predict(inp)["scenarios"][0]
        self.assertIsNone(s["saturated_output_tps"])
        self.assertTrue(s["warnings"])

    def test_new_capacity_bottleneck(self):
        inp = self.one()
        inp["scenarios"][0]["resource_demand_factors"]["gpu_compute"] = 0.5
        s = predict(inp)["scenarios"][0]
        self.assertEqual(s["capacity_bottleneck"], "hbm")
        self.assertAlmostEqual(s["saturated_output_tps"], 1800 * 0.92 / 0.68)

    def test_mtp_rejects_already_batched_baseline(self):
        inp = copy.deepcopy(self.fixture)
        inp["baseline"]["decode_tokens_per_step"] = 2
        with self.assertRaises(InputError):
            predict(inp)

    def test_duplicate_effects_rejected(self):
        inp = self.one()
        e = {"stage": "decode", "block": "qsa", "compute_speedup": 2}
        inp["scenarios"][0]["effects"] = [e, e]
        with self.assertRaises(InputError):
            predict(inp)

    def test_invalid_inputs_rejected(self):
        for field, value in [("ttft_mean_ms", -1), ("tpot_mean_ms", float("nan"))]:
            inp = self.one()
            inp["baseline"][field] = value
            with self.assertRaises(InputError):
                predict(inp)

    def test_observation_error_is_signed(self):
        inp = self.one()
        inp["scenarios"][0]["observed"] = {
            "ttft_mean_ms": 100, "tpot_mean_ms": 20, "saturated_output_tps": 2000
        }
        e = predict(inp)["scenarios"][0]["observed_error_pct"]
        self.assertAlmostEqual(e["ttft_mean_ms_error_pct"], 50)
        self.assertAlmostEqual(e["tpot_mean_ms_error_pct"], 10)
        self.assertAlmostEqual(e["saturated_output_tps_error_pct"], -10)

    def test_synthetic_markdown_never_calls_demo_a_measurement(self):
        report = as_markdown(predict(self.one()))
        self.assertIn("非 H800 实测", report)
        self.assertIn("合成基线", report)
        self.assertNotIn("基线实测", report)


if __name__ == "__main__":
    unittest.main()
