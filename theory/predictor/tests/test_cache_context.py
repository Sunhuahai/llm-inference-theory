"""All fixtures are synthetic. Do not treat expected values as H800 evidence."""

import copy
import json
import sys
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from cache_context import InputError, markdown, predict  # noqa: E402


class CacheContextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = MODULE_DIR / "examples/synthetic-cache-context.json"
        cls.fixture = json.loads(source.read_text(encoding="utf-8"))

    def run_fixture(self, **config):
        case = copy.deepcopy(self.fixture)
        case["config"].update(config)
        return predict(case)

    def test_baseline_and_upper_bound(self):
        r = self.run_fixture()
        self.assertEqual(r["data_kind"], "synthetic")
        self.assertAlmostEqual(r["baseline_ttft_mean_ms"], 121.5)
        self.assertAlmostEqual(r["absolute_all_prefill_removed_speedup"], 121.5 / 30)
        self.assertEqual(r["cache"]["no_cache"]["ttft_speedup"], 1.0)
        self.assertAlmostEqual(r["cache"]["lru_budget"]["request_hit_rate"], 1 / 8)
        self.assertAlmostEqual(r["cache"]["unbounded_exact_key"]["request_hit_rate"], 5 / 8)
        self.assertEqual(r["cache"]["lru_budget"]["evictions"], 6)
        self.assertGreater(
            r["cache"]["ideal_free_hit_exact_key"]["ttft_speedup"],
            r["cache"]["unbounded_exact_key"]["ttft_speedup"],
        )

    def test_three_hit_metrics_are_different(self):
        r = self.run_fixture()["cache"]["unbounded_exact_key"]
        self.assertAlmostEqual(r["request_hit_rate"], 0.625)
        self.assertNotAlmostEqual(r["cached_token_fraction"], r["request_hit_rate"])
        self.assertNotAlmostEqual(r["reusable_compute_hit_fraction"], r["request_hit_rate"])

    def test_capacity_extremes(self):
        tiny = self.run_fixture(cache_capacity_mib=0)["cache"]["lru_budget"]
        big = self.run_fixture(cache_capacity_mib=1_000)["cache"]
        self.assertEqual(tiny["hit_requests"], 0)
        self.assertEqual(tiny["ending_state_mib"], 0)
        self.assertEqual(big["lru_budget"]["hit_requests"], big["unbounded_exact_key"]["hit_requests"])
        self.assertEqual(big["lru_budget"]["evictions"], 0)

    def test_compression_rehash_and_gate_unknown(self):
        r = self.run_fixture()
        c = r["compression"]
        self.assertIsNotNone(c)
        self.assertTrue(c["compressor_break_even"])
        self.assertIsNone(c["quality_gate_ok"])
        self.assertGreater(c["results"]["compressed_lru_budget"]["request_hit_rate"],
                           r["cache"]["lru_budget"]["request_hit_rate"])
        self.assertIsNone(c["results"]["compressed_lru_budget"]["qualified_ttft_speedup"])
        self.assertIn("非 H800 实测", markdown(r))

    def test_quality_pass_and_fail(self):
        case = copy.deepcopy(self.fixture)
        for row in case["requests"]:
            row["compressed"]["quality_pass"] = True
        yes = predict(case)
        self.assertTrue(yes["compression"]["quality_gate_ok"])
        self.assertIsNotNone(yes["compression"]["results"]["compressed_lru_budget"]["qualified_ttft_speedup"])
        case["requests"][0]["compressed"]["quality_pass"] = False
        no = predict(case)
        self.assertFalse(no["compression"]["quality_gate_ok"])
        self.assertIsNone(no["compression"]["results"]["compressed_lru_budget"]["qualified_ttft_speedup"])

    def test_prefix_key_collision_cannot_change_state(self):
        case = copy.deepcopy(self.fixture)
        case["requests"][2]["baseline"]["state_mib"] += 1
        with self.assertRaisesRegex(InputError, "different prefix size/state"):
            predict(case)

    def test_partial_compression_rows_are_rejected(self):
        case = copy.deepcopy(self.fixture)
        del case["requests"][0]["compressed"]
        with self.assertRaisesRegex(InputError, "every request or none"):
            predict(case)

    def test_reject_nan_and_invalid_quality(self):
        case = copy.deepcopy(self.fixture)
        case["config"]["cache_capacity_mib"] = float("nan")
        with self.assertRaises(InputError):
            predict(case)
        case = copy.deepcopy(self.fixture)
        case["requests"][2]["compressed"]["quality_pass"] = "maybe"
        with self.assertRaises(InputError):
            predict(case)


if __name__ == "__main__":
    unittest.main()
