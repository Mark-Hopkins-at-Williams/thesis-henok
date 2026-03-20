# CE Loss Incorporation Process

## Goal
Incorporate standard translation cross-entropy (CE) loss into `est.v10.py` alongside the existing
alignment loss (cosine similarity + norm), inspired by the approach in:
> *Contrastive Learning for Many-to-many Multilingual Neural Machine Translation*

## Background

### Loss functions in play
| Version | Loss |
|---------|------|
| `est.v8.py` | Alignment loss only: cosine similarity + norm (token-level, with attention) |
| `est.v9.py` | InfoNCE contrastive CE loss only (sentence-level mean pooling + cross-entropy on similarity matrix) |
| `est.v10.py` | **Target:** alignment loss (from v8) + translation CE loss (standard seq2seq) |

### What CE loss means here
NOT the InfoNCE contrastive loss from v9. This is the standard seq2seq cross-entropy:
given the source token ids, run the full model (encoder + decoder) and compute
cross-entropy over the predicted target tokens. This is the same loss used in
normal NMT training.

## Problem: `next_batch()` return signature

`TokenizedMixtureOfTextAndGoalEncoding.next_batch()` currently returns a 4-tuple:
```
(lang1_dict, lang1, goal_encodings, lang2_attn_mask)
```
The raw target token ids (`lang2_sents`) are discarded — but we need them for CE loss labels.

### Files that unpack exactly 4 values (would break if signature changed)
- `test_corpora.py:461` — explicit unpack
- `batch_viz_attention.py:172` — explicit unpack inside loop
- `provide_permuted_encoding_to_decoder.py:31` — explicit unpack
- `est.v7.py`, `est.v8.py`, `est.v9.py` — pass batch to `compute_loss` which unpacks 4 values
- Various scripts in `scripts/`, `corrupt_gold_comparisons.py`, etc.

## Solution

Add a `return_raw_tokens=False` flag to `TokenizedMixtureOfTextAndGoalEncoding.__init__()`.
- Default `False`: existing 4-tuple behavior, no breakage to any other file.
- When `True`: `next_batch()` returns a 5-tuple `(lang1_dict, lang1, goal_encodings, lang2_attn_mask, lang2_sents)`.

Only `est.v10.py` uses `return_raw_tokens=True`.

## Changes

### 1. `corpora_kdswch.py` — `TokenizedMixtureOfTextAndGoalEncoding`

Added `return_raw_tokens=False` to `__init__()` and conditional return in `next_batch()`:

```python
def __init__(self, tmob, encoder, pad_token_id=0, return_raw_tokens=False):
    ...
    self.return_raw_tokens = return_raw_tokens

def next_batch(self):
    ...
    if self.return_raw_tokens:
        return lang1_dict, lang1, encodings, lang2_attn_mask, lang2_sents
    return lang1_dict, lang1, encodings, lang2_attn_mask
```

### 2. `est.v10.py`

`finetune()` signature now takes `pad_token_id`:
```python
def finetune(model, train_data1, dev_data, model_dir, ft_params, goal_lang_key="en", pad_token_id=0):
```

`compute_loss` unpacks 5 values and adds CE loss after the alignment loss:
```python
def compute_loss(batch):
    ...
    sents, lang, goal_encodings, goal_attn_mask, lang2_sents = batch
    ...
    # (alignment loss computed as before, result in `loss`)

    # CE loss: full model forward pass with target token ids as labels
    labels = lang2_sents.clone()
    labels[labels == pad_token_id] = -100
    ce_loss = model(
        input_ids=sents["input_ids"],
        attention_mask=sents["attention_mask"],
        labels=labels,
    ).loss
    loss = loss + ce_loss
    return loss
```

Mixtures constructed with `return_raw_tokens=True` and `pad_token_id` passed to `finetune`:
```python
train_mix = TokenizedMixtureOfTextAndGoalEncoding(
    train_data, static_model.model.encoder, pad_token_id=pad_token_id, return_raw_tokens=True
)
dev_mix = TokenizedMixtureOfTextAndGoalEncoding(
    dev_data, static_model.model.encoder, pad_token_id=pad_token_id, return_raw_tokens=True
)

finetune(model, train_mix, dev_mix, experiment_dir, ft_params,
         goal_lang_key=goal_lang_key, pad_token_id=pad_token_id)
```

## Notes
- The decoder is frozen in these experiments (`--> DECODER FROZEN <--`), so CE loss
  gradients only flow through the encoder.
- `encoder.eval()` is called inside `compute_loss` to disable dropout for the alignment
  loss computation; the full model forward pass for CE still benefits from this.
