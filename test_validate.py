import json
import tempfile
import unittest
from pathlib import Path

from validate import length_sorted_batches, load_oracle_scores, oracle_cache_path


class LengthSortedBatchesTest(unittest.TestCase):
    def test_every_index_appears_once(self):
        lengths = [5, 90, 3, 40, 40, 7, 600, 12, 1]
        batches = length_sorted_batches(lengths, max_pairs=3, max_tokens=200)
        seen = sorted(i for b in batches for i in b)
        self.assertEqual(seen, list(range(len(lengths))))

    def test_limits_hold_and_longest_come_first(self):
        lengths = [5, 90, 3, 40, 40, 7, 600, 12, 1, 88, 91, 2, 2, 2]
        batches = length_sorted_batches(lengths, max_pairs=4, max_tokens=300)
        for b in batches:
            self.assertLessEqual(len(b), 4)
            # padded to the longest member, unless it is alone and over the budget
            self.assertTrue(len(b) == 1 or len(b) * max(lengths[i] for i in b) <= 300)
        firsts = [max(lengths[i] for i in b) for b in batches]
        self.assertEqual(firsts, sorted(firsts, reverse=True))

    def test_oversized_item_gets_its_own_batch(self):
        batches = length_sorted_batches([10, 1000, 10], max_pairs=8, max_tokens=100)
        self.assertEqual(batches[0], [1])

    def test_ties_are_stable_and_empty_input_is_fine(self):
        self.assertEqual(length_sorted_batches([4, 4, 4], 2, 100), [[0, 1], [2]])
        self.assertEqual(length_sorted_batches([], 2, 100), [])


class OracleCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "test").mkdir()
        (self.tmp / "test" / "manifest.json").write_text('{"shards": []}')
        (self.tmp / "ac").mkdir()
        (self.tmp / "ac" / "best_model.pt").write_bytes(b"weights")
        (self.tmp / "ref.txt").write_text("a\nb\n")
        self.pc = {
            "test": str(self.tmp / "test"),
            "tgt_autocomplete_model": str(self.tmp / "ac"),
            "tgt_cap": 16,
            "tgt_max_length": 600,
            "reference_test_file": str(self.tmp / "ref.txt"),
        }

    def test_byte_targets_are_not_cached(self):
        del self.pc["tgt_autocomplete_model"]
        self.assertIsNone(oracle_cache_path(self.pc))

    def test_key_tracks_its_inputs(self):
        base = oracle_cache_path(self.pc)
        self.assertEqual(base, oracle_cache_path(self.pc))
        self.assertEqual(base.parent, self.tmp / "test")
        self.assertNotEqual(base, oracle_cache_path({**self.pc, "tgt_cap": 8}))
        (self.tmp / "ref.txt").write_text("a\nb\nc\n")
        self.assertNotEqual(base, oracle_cache_path(self.pc))

    def test_load_returns_none_when_missing_or_damaged(self):
        path = self.tmp / "oracle.json"
        self.assertIsNone(load_oracle_scores(path, "k"))
        path.write_text("not json")
        self.assertIsNone(load_oracle_scores(path, "k"))
        path.write_text(json.dumps({"k": {"bleu": 99.0, "chrf": 98.0}}))
        self.assertEqual(load_oracle_scores(path, "k"), {"bleu": 99.0, "chrf": 98.0})
        self.assertIsNone(load_oracle_scores(path, "other"))


if __name__ == "__main__":
    unittest.main()
