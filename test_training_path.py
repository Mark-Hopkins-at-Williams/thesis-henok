import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import M2M100Config, M2M100ForConditionalGeneration

from configure import create_bitexts, read_finetuning_params
from finetune import finetune
from precomputed import PrecomputedBatches, PrecomputedPairs, finalize
from test_precomputed import make_corpus, process
from test_target_side import LANG, EOS, make_tokenizer, random_text
from validate import diagnose_hypotheses


def build_data(directory):
    src, tgt, kept = make_corpus(directory)
    out = directory / "compressed"
    process(directory, out, worker=0, num_workers=1)
    finalize(out)
    return out, kept


def make_config(data_dir, **overrides):
    params = {
        "base_model": "facebook/nllb-200-distilled-600M",
        "finetune": False,
        "batch_size": 4,
        "num_steps": 30,
        "report_every": 1000,
        "validate_every": 1000,
        "checkpoint_every": 10,
        "config_overrides": {"pad_token_id": 0, "vocab_size": 2307},
    }
    params.update(overrides)
    return {
        "finetuning_parameters": params,
        "precomputed": {
            "train": str(data_dir),
            "dev": str(data_dir),
            "test": str(data_dir),
            "src_cap": 8,
            "tgt_cap": 8,
        },
    }


def tiny_model(seed):
    torch.manual_seed(seed)
    cfg = M2M100Config(
        vocab_size=2307,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=2,
    )
    return M2M100ForConditionalGeneration(cfg)


class TestBatches(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.data, self.kept = build_data(self.dir)
        self.pairs = PrecomputedPairs(self.data, 8, 8)

    def tearDown(self):
        self._tmp.cleanup()

    def test_padding_and_masks(self):
        batches = PrecomputedBatches(self.pairs, 4, only_once_thru=True)
        src, tgt, _ = next(iter(batches))
        self.assertEqual(src["input_ids"].shape[0], 4)
        for row in range(4):
            expected_src, expected_tgt = self.pairs[row]
            n, m = len(expected_src), len(expected_tgt)
            self.assertEqual(src["input_ids"][row, :n].tolist(), expected_src)
            self.assertTrue((src["input_ids"][row, n:] == 0).all())
            self.assertEqual(int(src["attention_mask"][row].sum()), n)
            self.assertEqual(tgt["input_ids"][row, :m].tolist(), expected_tgt)
            self.assertTrue((tgt["input_ids"][row, m:] == -100).all())

    def test_evaluation_pass_stops_and_keeps_the_partial_batch(self):
        batches = PrecomputedBatches(self.pairs, 4, only_once_thru=True)
        sizes = [src["input_ids"].shape[0] for src, _, _ in batches]
        self.assertEqual(sum(sizes), len(self.pairs))
        self.assertEqual(sizes[-1], len(self.pairs) % 4 or 4)
        self.assertEqual(list(iter(batches)), [])  # exhausted until restart()
        batches.restart()
        self.assertEqual(sum(1 for _ in batches), len(sizes))

    def test_training_pass_wraps_around(self):
        batches = PrecomputedBatches(self.pairs, 4)
        it = iter(batches)
        n_batches = -(-len(self.pairs) // 4)
        first = next(it)[0]["input_ids"]
        for _ in range(n_batches - 1):
            next(it)
        again = next(it)[0]["input_ids"]  # start of the second epoch
        self.assertTrue(torch.equal(first, again))

    def test_seek_reproduces_the_stream(self):
        a = PrecomputedBatches(self.pairs, 4)
        it = iter(a)
        for _ in range(3):
            next(it)
        rest_a = [next(it)[1]["input_ids"] for _ in range(3)]
        b = PrecomputedBatches(self.pairs, 4)
        b.seek(12)
        it_b = iter(b)
        rest_b = [next(it_b)[1]["input_ids"] for _ in range(3)]
        for x, y in zip(rest_a, rest_b):
            self.assertTrue(torch.equal(x, y))
        self.assertEqual(a.position, b.position)

    def test_create_bitexts_for_a_precomputed_config(self):
        bitexts = create_bitexts(make_config(self.data))
        self.assertEqual(bitexts["required_vocab"], 259 + 256 * 8)
        self.assertEqual(bitexts["cipher_map"], {})
        self.assertEqual(len(bitexts["pairs"]["test"]), len(self.kept))
        self.assertFalse(bitexts["train"].only_once_thru)
        self.assertTrue(bitexts["dev"].only_once_thru)


class TestResume(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.data, _ = build_data(self.dir)

    def tearDown(self):
        self._tmp.cleanup()

    def train(self, name, num_steps, model, resume=False):
        config = make_config(self.data, num_steps=num_steps)
        bitexts = create_bitexts(config)
        model_dir = self.dir / name
        model_dir.mkdir(exist_ok=True)
        finetune(
            model,
            bitexts["train"],
            bitexts["dev"],
            str(model_dir),
            read_finetuning_params(config),
            resume=resume,
        )
        return bitexts["train"].position

    def test_interrupted_run_matches_uninterrupted_run(self):
        straight = tiny_model(0)
        position_straight = self.train("straight", 30, straight)

        self.train("resumed", 20, tiny_model(0))  # "crashes" after step 20
        self.assertTrue((self.dir / "resumed" / "checkpoint.pt").exists())
        restored = tiny_model(1)  # a different initialisation: must be overwritten
        position_resumed = self.train("resumed", 30, restored, resume=True)

        self.assertEqual(position_resumed, position_straight)
        initial = tiny_model(0).state_dict()  # guard against a vacuous comparison
        moved = max(
            (straight.state_dict()[k] - v).abs().max().item() for k, v in initial.items()
        )
        self.assertGreater(moved, 1e-3, "training did not change the weights")
        for (name, a), (_, b) in zip(
            straight.state_dict().items(), restored.state_dict().items()
        ):
            self.assertTrue(torch.allclose(a, b, atol=1e-6), name)

    def test_resume_without_a_checkpoint_fails(self):
        with self.assertRaises(FileNotFoundError):
            self.train("empty", 10, tiny_model(0), resume=True)


class TestParameters(unittest.TestCase):
    def test_old_configs_keep_their_behaviour(self):
        with open("configs/ru-en.ac.json") as reader:
            params = read_finetuning_params(json.load(reader))
        self.assertEqual(params.checkpoint_every, 0)
        self.assertEqual(params.config_overrides, {})
        self.assertIsNone(params.forced_bos_token_id)
        self.assertIsNone(params.seed)

    def test_new_settings_are_read(self):
        with open("configs/en-ru-ac16.json") as reader:
            params = read_finetuning_params(json.load(reader))
        self.assertEqual(params.config_overrides, {"pad_token_id": 0, "vocab_size": 4355})
        self.assertEqual(params.forced_bos_token_id, 1)
        self.assertEqual(params.patience, 20)


class TestDiagnostics(unittest.TestCase):
    def test_classification(self):
        import random

        decoder = make_tokenizer(cap=8, keep_first=True, keep_eos=True)
        rng = random.Random(4)
        texts = [random_text(rng, 30) for _ in range(6)]
        valid = [decoder(t)[1:] for t in texts]
        explicit = [[b + 3 for b in t.encode()] + [EOS] for t in texts]  # nothing removed
        no_eos = [v[:-1] for v in valid]

        def rows(seqs):
            return [[EOS, LANG] + s for s in seqs]

        def run(seqs):
            row_ids = rows(seqs)
            return diagnose_hypotheses(
                decoder, row_ids, decoder.batch_decode(torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(r) for r in row_ids], batch_first=True))
            )

        self.assertEqual(run(valid)["valid"], 6)
        self.assertEqual(run(explicit)["canonical_form_shorter"] + run(explicit)["valid"], 6)
        self.assertGreaterEqual(run(explicit)["canonical_form_shorter"], 1)
        self.assertEqual(run(no_eos)["other"], 6)


if __name__ == "__main__":
    unittest.main()
