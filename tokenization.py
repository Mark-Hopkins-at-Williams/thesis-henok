import sys
from transformers import AutoTokenizer
from typing import Dict, Tuple, List, Optional, Iterator, Callable
import warnings
from abc import ABC
from abc import abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __call__(self, sents: List[str]):
        pass

    @abstractmethod
    def get_special_tokens(self):
        pass

    @abstractmethod
    def batch_decode(self):
        pass


class HuggingfaceTokenizer(Tokenizer):

    def __init__(self, model_name, max_length=None):
        self.max_length = max_length
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`clean_up_tokenization_spaces` was not set.*",
                category=FutureWarning,
                module="transformers.tokenization_utils_base",
            )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except OSError:
                sys.stderr.write("Tokenizer not found. Using NLLB tokenizer instead.\n")
                sys.stderr.flush()
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "facebook/nllb-200-distilled-600M"
                )
        self.special_tokens = dict(
            zip(self.tokenizer.all_special_tokens, self.tokenizer.all_special_ids)
        )

    def __len__(self):
        return len(self.tokenizer)

    def __call__(self, sents: List[str], lang_code=None):
        if lang_code is not None:
            self.tokenizer.src_lang = lang_code
        result = self.tokenizer(
            sents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length if self.max_length is not None else None,
        )
        retval = result["input_ids"].squeeze().tolist()
        return retval

    def get_special_tokens(self):
        return self.special_tokens

    def batch_decode(self, token_ids):
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)


class NllbTokenizer(HuggingfaceTokenizer):
    def __init__(self, size, max_length=None):
        super().__init__(f"facebook/nllb-200-distilled-{size}", max_length=max_length)


def strip_generated_ids(row, lang_id=1, eos_id=2):
    """Cleans one row of generated (or gold target) ids in the byte/autocomplete layout.

    Drops the decoder-start token, the language token, padding (0 and -100), and
    everything after the first EOS. Returns the remaining ids, EOS included if present.
    """
    row = row.tolist() if hasattr(row, "tolist") else list(row)
    if lang_id in row[:2]:
        row = row[row.index(lang_id) + 1 :]
    elif row and row[0] == eos_id:
        row = row[1:]
    kept = []
    for t in row:
        if t == eos_id:
            kept.append(t)
            break
        if t > eos_id:
            kept.append(t)
    return kept


class ByteTokenizer(Tokenizer):
    def __init__(self, max_length=None):
        self.special_tokens = {"<pad>": 0, "</s>": 2, "lang_Lang": 1}
        self.max_length = max_length

    def __call__(self, sent: str, lang_code="lang_Lang"):
        tokens = [byte + len(self.special_tokens) for byte in sent.encode()]
        if self.max_length is not None and len(tokens) > self.max_length - 2:
            tokens = tokens[: self.max_length - 2]
        result = (
            [self.special_tokens[lang_code]] + tokens + [self.special_tokens["</s>"]]
        )
        return result

    def __len__(self):
        return 256 + len(self.special_tokens)

    def get_special_tokens(self):
        return self.special_tokens

    def batch_decode(self, token_ids):
        results = []
        for row in token_ids:
            kept = strip_generated_ids(row)
            data = bytes(t - len(self.special_tokens) for t in kept if t > 2)
            results.append(data.decode("utf-8", errors="replace"))
        return results

    #     results = []  # list of strings, each of which is decoded sentence

    #     non_printables = set(self.special_tokens)

    #     for sent in token_ids:
    #         # Convert all token IDs in one go
    #         decoded_tokens = [self.reverse_mappings[id.item()] for id in sent]

    #         # Filter out special tokens and make string representation
    #         decoded = " ".join(
    #             tok for tok in decoded_tokens if tok not in non_printables
    #         )
    #         results.append(decoded)

    #     return results
