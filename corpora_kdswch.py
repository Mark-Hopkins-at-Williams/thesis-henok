import random
import torch
from torch.utils.data import DataLoader, IterableDataset

from typing import Dict, Tuple, List, Optional, Iterator, Callable
from tokenization import Tokenizer
from extract_tok_utils import (
    build_replacement_plans,
    filter_overlapping_plans,
    apply_replacements,
    collate_and_pad,
)
from torch.nn.utils.rnn import pad_sequence

from align import extract_phrase_pairs

CorpusId = Tuple[str, str]  # typedef


class MultifileBitext(IterableDataset):
    def __init__(
        self,
        lang1_files: List[str],
        lang2_files: List[str],
        lines: Optional[List[Tuple[int, int]]] = None,
    ):
        self.lang1_files = lang1_files
        self.lang2_files = lang2_files
        self.lines = lines

    def line_streamer(self, lang_index) -> Iterator[str]:
        lang_files = self.lang1_files if lang_index == 0 else self.lang2_files
        for file_index in range(len(self.lang1_files)):
            file_path = lang_files[file_index]
            current_line = 0
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if (
                        self.lines is None
                        or self.lines[file_index][0]
                        <= current_line
                        < self.lines[file_index][1]
                    ):
                        yield line.rstrip("\n")
                    current_line += 1
                    if (
                        self.lines is not None
                        and current_line >= self.lines[file_index][1]
                    ):
                        break

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        return zip(self.line_streamer(0), self.line_streamer(1))


class Bitext(IterableDataset):
    def __init__(
        self, lang1_file: str, lang2_file: str, lines: Optional[Tuple[int, int]] = None
    ):
        self.lang1_file = lang1_file
        self.lang2_file = lang2_file
        self.lines = lines

    def line_streamer(self, file_path: str) -> Iterator[str]:
        current_line = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if self.lines is None or self.lines[0] <= current_line < self.lines[1]:
                    yield line.rstrip("\n")
                current_line += 1
                if self.lines is not None and current_line >= self.lines[1]:
                    break

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        return zip(
            self.line_streamer(self.lang1_file), self.line_streamer(self.lang2_file)
        )


class TokenizedBitext:
    def __init__(self, bitext, tokenizer, lang1_code, lang2_code):
        self.bitext = bitext
        self.tokenizer = tokenizer
        self.lang1_code = lang1_code
        self.lang2_code = lang2_code

    def __iter__(self):
        bitext_iter = iter(self.bitext)
        for text1, text2 in bitext_iter:
            yield self._tokenize(text1, text2)

    def _tokenize(self, lang1_text, lang2_text):
        lang1_tokens = (
            self.tokenizer(lang1_text, lang_code=self.lang1_code)["input_ids"]
            .squeeze()
            .tolist()
        )
        lang2_tokens = (
            self.tokenizer(lang2_text, lang_code=self.lang2_code)["input_ids"]
            .squeeze()
            .tolist()
        )
        return (lang1_tokens, lang2_tokens)


class CodeswitchedBitext:
    def __init__(
        self,
        tokenized_bitext,
        alignment,
        randomizer=lambda ls: random.randint(0, len(ls) - 1),
    ):
        self.tokenized_bitext = tokenized_bitext
        self.alignment = alignment
        self.randomizer = randomizer

    def __iter__(self):
        bitext_iter = iter(self.tokenized_bitext)
        for i, (toks1, toks2) in enumerate(bitext_iter):
            alignment = self.alignment[i]
            pairs = extract_phrase_pairs(toks1, toks2, alignment)
            indices = list(range(len(pairs)))
            index_chosen = False
            while len(indices) > 0 and not index_chosen:
                j = self.randomizer(indices)
                span1_start, span1_end, span2_start, span2_end = pairs[indices[j]]
                if (
                    span1_start == 0
                    or span2_start == 0
                    or span1_end == len(toks1) - 1
                    or span2_end == len(toks2) - 1
                ):  # don't include lang code or end of sentence marker
                    indices = indices[:j] + indices[j + 1 :]
                else:
                    index_chosen = True
            if index_chosen:
                revised_toks1 = []
                for j in range(len(toks1)):
                    if j == span1_start:
                        revised_toks1.extend(toks2[span2_start : span2_end + 1])
                    elif j < span1_start or j > span1_end:
                        revised_toks1.append(toks1[j])
                toks1 = revised_toks1
            yield (toks1, toks2)


class BitextIterableDataset(IterableDataset):
    def __init__(self, bitext_iterable):
        self.bitext_iterable = bitext_iterable

    def __iter__(self):
        yield from self.bitext_iterable


class BatchedBitext:
    def __init__(self, bitext, batch_size, src_pad_token=0, tgt_pad_token=0):
        self.bitext = bitext
        self.batch_size = batch_size
        self.src_pad_token = src_pad_token
        self.tgt_pad_token = tgt_pad_token

    def collate_fn(self, batch):
        src, tgt = zip(*batch)
        src = [torch.tensor(x) for x in src]
        tgt = [torch.tensor(x) for x in tgt]
        src_padded = pad_sequence(
            src, batch_first=True, padding_value=self.src_pad_token
        )
        tgt_padded = pad_sequence(
            tgt, batch_first=True, padding_value=self.tgt_pad_token
        )
        return src_padded, tgt_padded

    def __iter__(self):
        loader = DataLoader(
            BitextIterableDataset(self.bitext),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            drop_last=False,
        )
        for batch in loader:
            yield batch


class MixtureOfBitexts:
    def __init__(
        self,
        bitexts: Dict[Tuple[str, str], Bitext],
        sampling_probs: Optional[List[float]] = None,
        only_once_thru: bool = False,
    ):
        self.bitexts = bitexts
        self.keys = list(bitexts)
        self.batch_iters = {}
        for key in self.keys:
            self.batch_iters[key] = iter(self.bitexts[key])

        total = sum(sampling_probs) if sampling_probs else len(bitexts)
        self.sampling_probs = [
            p / total for p in (sampling_probs or [1.0] * len(bitexts))
        ]
        self.only_once_thru = only_once_thru
        self.completed_bitexts = set()

    def restart(self):
        self.completed_bitexts = set()
        for key in self.keys:
            self.batch_iters[key] = iter(self.bitexts[key])

    def __iter__(self):
        still_looping = True
        while still_looping:
            still_choosing = True
            while still_choosing and len(self.completed_bitexts) < len(self.keys):
                lang_pair = random.choices(self.keys, weights=self.sampling_probs, k=1)[
                    0
                ]
                try:
                    lang1_sents, lang2_sents = next(self.batch_iters[lang_pair])
                    still_choosing = False
                except StopIteration:
                    if self.only_once_thru:
                        self.completed_bitexts.add(lang_pair)
                    else:
                        self.batch_iters[lang_pair] = iter(self.bitexts[lang_pair])
            if not still_choosing:
                yield lang1_sents, lang2_sents, lang_pair[0], lang_pair[1]
            else:
                still_looping = False

    @staticmethod
    def create_from_files(
        text_files: Dict[str, str],
        lps: List[Tuple[str, str, Optional[Tuple[int, int]]]],
        batch_size: int,
        sampling_probs: Optional[List[float]] = None,
        only_once_thru: bool = False,
    ) -> "MixtureOfBitexts":
        bitexts = {
            (l1, l2): Bitext(text_files[l1], text_files[l2], lines)
            for (l1, l2, lines) in lps
        }
        return MixtureOfBitexts(bitexts, batch_size, sampling_probs, only_once_thru)

    @staticmethod
    def create_from_config(
        config: dict, split: str, only_once_thru: bool = False
    ) -> "MixtureOfBitexts":
        all_corpora = dict()
        for corpus in config["corpora"]:
            for key in config["corpora"][corpus]:
                all_corpora[(corpus, key)] = config["corpora"][corpus][key][split]
        bitexts = dict()
        for bitext in config["bitexts"]:
            src = (bitext["corpus"], bitext["src"])
            tgt = (bitext["corpus"], bitext["tgt"])
            lines = (
                bitext["train_lines"]
                if split == "train" and "train_lines" in bitext
                else None
            )
            bitexts[(src, tgt)] = Bitext(all_corpora[src], all_corpora[tgt], lines)
        params = config["finetuning_parameters"]
        return MixtureOfBitexts(
            bitexts,
            params["batch_size"],
            sampling_probs=None,
            only_once_thru=only_once_thru,
        )

    def get_language_codes(self) -> List[str]:
        return sorted({code for pair in self.keys for code in pair})


class TokenizedMixtureOfTextAndGoalEncoding:
    def __init__(self, tmob, encoder, pad_token_id=0, return_raw_tokens=False):
        self.tmob = tmob
        self.encoder = encoder
        self.encoder.eval()
        self.pad_token_id = pad_token_id
        self.return_raw_tokens = return_raw_tokens
        self._iter = iter(self.tmob)

    def next_batch(self):
        try:
            lang1_sents, lang2_sents, lang1, _ = next(self._iter)
        except StopIteration:
            return None
        lang2_sents = lang2_sents.to(self.encoder.device)
        lang2_attn_mask = (lang2_sents != self.pad_token_id).long()
        with torch.no_grad():
            encodings = self.encoder(input_ids=lang2_sents, attention_mask=lang2_attn_mask).last_hidden_state
        lang1_sents = lang1_sents.to(self.encoder.device)
        lang1_attn_mask = (lang1_sents != self.pad_token_id).long()
        lang1_dict = {"input_ids": lang1_sents, "attention_mask": lang1_attn_mask}
        if self.return_raw_tokens:
            return lang1_dict, lang1, encodings, lang2_attn_mask, lang2_sents
        return lang1_dict, lang1, encodings, lang2_attn_mask

    def restart(self):
        self.tmob.restart()
        self._iter = iter(self.tmob)
