import random
import unittest

import torch

from compressor import AutocompletingTokenizer
from tokenization import ByteTokenizer, strip_generated_ids

VOCAB = 259  # pad, lang, eos, 256 bytes
LANG, EOS = 1, 2
MAX_ID = 130  # keep text ASCII (byte 127 + 3)


class MockAutocompleteModel(torch.nn.Module):
    """Deterministic stand-in for the autocomplete model.

    After token t it is confident (p ~ 0.99) that the next token is t + 1, unless t is
    a multiple of 5 (or a special token), in which case it is clueless (uniform). So a
    kept token with residue r = t % 5 in 1..4 is followed by a chain of 5 - r
    predictable tokens, i.e. run lengths are 1..4. eos_after optionally makes the model
    confident of EOS after one particular token.
    """

    def __init__(self, eos_after=None):
        super().__init__()
        self.eos_after = eos_after

    def guess(self, t):
        if t == self.eos_after:
            return EOS
        if t >= 3 and t % 5 != 0 and t + 1 <= MAX_ID:
            return t + 1
        return None

    def forward(self, input_ids):
        logits = torch.zeros(*input_ids.shape, VOCAB)
        for b in range(input_ids.shape[0]):
            for j in range(input_ids.shape[1]):
                g = self.guess(int(input_ids[b, j]))
                if g is not None:
                    logits[b, j, g] = 10.0
        return logits


def make_tokenizer(cap=8, eos_after=None, **kwargs):
    return AutocompletingTokenizer(
        MockAutocompleteModel(eos_after),
        torch.device("cpu"),
        max_length_encoding=cap,
        **kwargs,
    )


def random_text(rng, n, surprises=False, finish_chain=True):
    """Random ASCII text. After a confident context the next byte follows the model's
    guess (so the compressor removes it); with surprises=True it sometimes deviates.
    finish_chain=True runs the text on until the model is clueless, so the text never
    ends in the middle of a predictable run."""
    model = MockAutocompleteModel()
    ids = [LANG]
    while len(ids) <= n:
        g = model.guess(ids[-1])
        if g is not None and not (surprises and rng.random() < 0.25):
            ids.append(g)
        else:
            ids.append(rng.randrange(3, MAX_ID + 1))
    while finish_chain and model.guess(ids[-1]) is not None:
        ids.append(model.guess(ids[-1]))
    return bytes(t - 3 for t in ids[1:]).decode("ascii")


class TestTargetSideCompression(unittest.TestCase):
    def test_marker_encoding(self):
        # ids 6,7,8,9,10 then 20: only 6 and 20 are kept; 6 is followed by 4 removed.
        text = bytes([3, 4, 5, 6, 7, 17]).decode()
        tok = make_tokenizer(keep_first=True, keep_eos=True)
        self.assertEqual(tok(text), [LANG, 6 + 256 * 4, 20, EOS])

    def test_marker_is_capped(self):
        text = bytes([3, 4, 5, 6, 7, 17]).decode()
        tok = make_tokenizer(cap=2, keep_first=True, keep_eos=True)
        self.assertEqual(tok(text), [LANG, 6 + 256 * 2, 20, EOS])

    def test_default_removes_first_byte_and_eos(self):
        # LANG is unconfident in the mock, so make the first byte predictable by using
        # a model that is confident of id 9 after LANG and of EOS after 9.
        model = MockAutocompleteModel(eos_after=9)
        model.guess_orig = model.guess
        model.guess = lambda t: 9 if t == LANG else model.guess_orig(t)
        tok = AutocompletingTokenizer(model, torch.device("cpu"))
        self.assertEqual(tok(bytes([6]).decode()), [LANG])  # both vanish, count lost

    def test_keep_first_and_keep_eos(self):
        model = MockAutocompleteModel(eos_after=9)
        model.guess_orig = model.guess
        model.guess = lambda t: 9 if t == LANG else model.guess_orig(t)
        tok = AutocompletingTokenizer(
            model, torch.device("cpu"), keep_first=True, keep_eos=True
        )
        self.assertEqual(tok(bytes([6]).decode()), [LANG, 9, EOS])

    def test_target_sequences_start_with_lang_and_end_with_eos(self):
        rng = random.Random(0)
        tok = make_tokenizer(keep_first=True, keep_eos=True)
        for _ in range(50):
            ids = tok(random_text(rng, rng.randrange(1, 40), surprises=True))
            self.assertEqual(ids[0], LANG)
            self.assertEqual(ids[-1], EOS)

    def test_roundtrip_exact_at_every_cap(self):
        rng = random.Random(1)
        for cap in (1, 2, 3, 8):
            tok = make_tokenizer(cap=cap, keep_first=True, keep_eos=True)
            for _ in range(60):
                text = random_text(rng, rng.randrange(1, 60), surprises=(cap == 8))
                rec = tok.decompress([tok(text)[1:]])[0]
                self.assertEqual(rec.decode(), text, f"cap={cap}")

    def test_saturated_run_overshoots_if_text_ends_mid_chain(self):
        # Known limitation: a marker at the cap only says "at least k", and the greedy
        # rule keeps going while the model is confident. If the text stops while the
        # model is still confident, the fill-in runs past the end.
        tok = make_tokenizer(cap=2, keep_first=True, keep_eos=True)
        text = bytes([3, 4, 5]).decode()  # ids 6,7,8: 6 kept, 7 and 8 removed; 9 predicted
        ids = tok(text)
        self.assertEqual(ids, [LANG, 6 + 256 * 2, EOS])
        rec = tok.decompress([ids[1:]])[0]
        self.assertNotEqual(rec.decode(), text)
        self.assertTrue(rec.startswith(text.encode()))

    def test_batch_decompress_matches_one_at_a_time(self):
        rng = random.Random(3)
        tok = make_tokenizer(cap=4, keep_first=True, keep_eos=True)
        seqs = [
            tok(random_text(rng, rng.randrange(1, 50), surprises=True))[1:]
            for _ in range(12)
        ]
        self.assertEqual(
            tok.decompress(seqs), [tok.decompress([s])[0] for s in seqs]
        )

    def test_batch_decode_strips_generation_artifacts(self):
        tok = make_tokenizer(cap=8, keep_first=True, keep_eos=True)
        kept = [20, 30, EOS]  # both ids are multiples of 5: nothing to fill in
        rows = [
            [EOS, LANG] + kept + [0, 0],  # what generate() returns
            [LANG] + kept + [-100, -100, -100],  # gold labels
        ]
        expect = bytes([17, 27]).decode()
        self.assertEqual(tok.batch_decode(torch.tensor(rows)), [expect, expect])


class TestBatchedCompression(unittest.TestCase):
    def texts(self):
        rng = random.Random(5)
        texts = [
            random_text(rng, rng.randrange(0, 70), surprises=bool(i % 2))
            for i in range(40)
        ]
        return texts + [""]

    def test_matches_per_sentence_tokenizer(self):
        texts = self.texts()
        for keep_first in (False, True):
            for keep_eos in (False, True):
                for cap in (1, 2, 8):
                    tok = make_tokenizer(
                        cap=cap, keep_first=keep_first, keep_eos=keep_eos
                    )
                    expected = [tok(t) for t in texts]
                    # a tiny budget forces many chunks of different shapes
                    self.assertEqual(
                        tok.compress_batch(texts, max_tokens_per_forward=150),
                        expected,
                        f"keep_first={keep_first} keep_eos={keep_eos} cap={cap}",
                    )

    def test_matches_when_truncated(self):
        texts = self.texts()
        tok = make_tokenizer(cap=8, keep_first=True, keep_eos=True, max_length=10)
        self.assertEqual(tok.compress_batch(texts), [tok(t) for t in texts])

    def test_one_pass_serves_every_cap(self):
        texts = self.texts()
        tok = make_tokenizer(cap=8, keep_first=True, keep_eos=True)
        runs = tok.compress_runs(texts)
        for cap in (1, 3, 16):
            tok_cap = make_tokenizer(cap=cap, keep_first=True, keep_eos=True)
            for text, (base, run) in zip(texts, runs):
                capped = run.astype(int).clip(max=cap)
                rebuilt = [LANG] + (base.astype(int) + 256 * capped).tolist()
                self.assertEqual(rebuilt, tok_cap(text))


class TestByteTarget(unittest.TestCase):
    def test_byte_batch_decode_roundtrip(self):
        tok = ByteTokenizer(max_length=64)
        sents = ["hello", "Привет, мир", ""]
        self.assertEqual(tok.batch_decode([tok(s) for s in sents]), sents)

    def test_strip_generated_ids(self):
        self.assertEqual(strip_generated_ids([2, 1, 5, 6, 2, 0]), [5, 6, 2])
        self.assertEqual(strip_generated_ids([1, 5, 6, -100]), [5, 6])
        self.assertEqual(strip_generated_ids([2, 5, 6, 2, 7]), [5, 6, 2])


if __name__ == "__main__":
    unittest.main()
