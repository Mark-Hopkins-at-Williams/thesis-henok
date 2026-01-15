import argparse
import json
import os
from pathlib import Path

import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from attention import SimpleAttention
from configure import (
    create_experiment_dir,
    create_permutations,
    harvest_language_codes,
    initialize_tokenizer,
    read_finetuning_params,
)
from corpora import (
    MixtureOfBitexts,
    TokenizedMixtureOfBitexts,
    TokenizedMixtureOfTextAndGoalEncoding,
)
from myutil import prepare_model_for_finetuning, logger
from permutations import save_permutation_map

matplotlib.use("Agg")


# -------------------------
# Visualization utilities
# -------------------------

def visualize_attention_grid(
    weights: torch.Tensor,
    save_path: Path,
    title: str,
):
    """
    Visualize a single attention matrix.

    weights: [T_q, T_k]
    """
    weights = weights.detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(weights, aspect="auto")
    plt.colorbar(label="attention weight")

    plt.xlabel("Key tokens (other encoding)")
    plt.ylabel("Query tokens (current encoding)")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_attention_batch(
    attn,
    enc_a,
    enc_b,
    out_dir: Path,
    prefix: str,
    batch_idx: int,
):
    """
    Visualize attention grids for every sentence in a batch.

    enc_a: [B, T_a, H]
    enc_b: [B, T_b, H]
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        _, weights = attn(enc_a, enc_b)
        # weights: [B, T_a, T_b]

    B = weights.size(0)

    for i in range(B):
        save_path = out_dir / f"{prefix}_batch{batch_idx}_sent{i}.png"
        visualize_attention_grid(
            weights[i],
            save_path,
            title=f"{prefix} | batch {batch_idx} | sent {i}",
        )


# -------------------------
# Main driver
# -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize SimpleAttention alignments from config"
    )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    ft_params = read_finetuning_params(config)
    experiment_dir = create_experiment_dir(config, args.config)

    logger(f"Experiment dir: {experiment_dir}")

    lang_codes = harvest_language_codes(config)
    tokenizer = initialize_tokenizer(config)

    pmap = create_permutations(config, tokenizer)
    save_permutation_map(pmap, Path(experiment_dir) / "permutations.json")

    dev_data = MixtureOfBitexts.create_from_config(
        config, "dev", only_once_thru=True
    )

    tokenized_dev = TokenizedMixtureOfBitexts(
        dev_data,
        tokenizer,
        lang_codes=lang_codes,
        permutation_map=pmap,
        use_alt_pad_token_for_tgt_lang=False,
    )

    static_model = prepare_model_for_finetuning(ft_params)
    encoder = static_model.model.encoder
    encoder.eval()

    dev_mix = TokenizedMixtureOfTextAndGoalEncoding(
        tokenized_dev,
        encoder,
    )

    attn = SimpleAttention()

    viz_root = Path(experiment_dir) / "attention_visualizations"
    viz_root.mkdir(parents=True, exist_ok=True)

    logger("Starting attention visualization over dev set")

    batch_idx = 0
    dev_mix.restart()

    with torch.no_grad():
        batch = dev_mix.next_batch()
        while batch is not None:
            sents, lang, goal_encodings = batch

            sents = sents.to(encoder.device)
            goal_encodings = goal_encodings.to(encoder.device)

            sent_encodings = encoder(**sents).last_hidden_state

            lang_dir = viz_root / f"{lang[0]}_{lang[1]}"
            lang_dir.mkdir(parents=True, exist_ok=True)

            visualize_attention_batch(
                attn,
                sent_encodings,
                goal_encodings,
                out_dir=lang_dir,
                prefix="sent_to_goal",
                batch_idx=batch_idx,
            )

            visualize_attention_batch(
                attn,
                goal_encodings,
                sent_encodings,
                out_dir=lang_dir,
                prefix="goal_to_sent",
                batch_idx=batch_idx,
            )

            batch_idx += 1
            batch = dev_mix.next_batch()

    logger("Attention visualization complete")


if __name__ == "__main__":
    main()
