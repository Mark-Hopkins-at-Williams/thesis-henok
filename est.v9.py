import argparse
from attention import SimpleAttention
from configure import create_experiment_dir
from configure import harvest_language_codes
from configure import initialize_tokenizer
from configure import read_finetuning_params
from corpora_kdswch import Bitext, TokenizedBitext, CodeswitchedBitext, BatchedBitext
from corpora_kdswch import MixtureOfBitexts
from corpora_kdswch import TokenizedMixtureOfTextAndGoalEncoding
from extract_tok_utils import build_fast_align_dict_from_raw
import json
import matplotlib
import matplotlib.pyplot as plt
from myutil import cleanup
from myutil import logger
from myutil import prepare_model_for_finetuning
import numpy as np
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Adafactor
from transformers import get_constant_schedule_with_warmup
from validate import evaluate_experiment

matplotlib.use("Agg")


def plot_losses(train_x, train_y, dev_x, dev_y, out_path: str):
    plt.clf()
    plt.plot(train_x, train_y, label="train", color="blue", linewidth=2)
    plt.plot(dev_x, dev_y, label="dev", color="red", linewidth=2)
    plt.xlabel("training steps")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)


def finetune(model, train_data1, dev_data, model_dir, ft_params, goal_lang_key="en"):
    def mean_pool(encodings, attn_mask):
        # encodings: (batch, seq_len, hidden); attn_mask: (batch, seq_len)
        mask = attn_mask.unsqueeze(-1).float()
        return (encodings * mask).sum(dim=1) / mask.sum(dim=1)

    def compute_loss(batch, temperature=0.1):
        encoder.eval()
        sents, lang, goal_encodings, goal_attn_mask = batch

        sent_attn_mask = sents["attention_mask"].to(encoder.device)
        sents = {k: v.to(encoder.device) for k, v in sents.items()}
        goal_encodings = goal_encodings.to(encoder.device)
        goal_attn_mask = goal_attn_mask.to(encoder.device)

        sent_encodings = encoder(**sents).last_hidden_state

        # R(xi): mean-pooled trainable encoder output on source (lang1)
        src_vecs = mean_pool(sent_encodings, sent_attn_mask)  # (B, H)

        # R(xj) / R(yj): mean-pooled static encoder output on goal (lang2)
        tgt_vecs = mean_pool(goal_encodings, goal_attn_mask)  # (B, H)

        # cosine similarity
        src_vecs = F.normalize(src_vecs, dim=-1)
        tgt_vecs = F.normalize(tgt_vecs, dim=-1)

        # sim[i, j] = cos(R(xi), R(tgt_j)) / τ
        sim = torch.matmul(src_vecs, tgt_vecs.T) / temperature  # (B, B)

        B = sim.size(0)

        # sim+: positive pair similarity — diagonal (xi matched with its correct translation xj)
        numerator = torch.diagonal(sim)  # (B,)

        # sim-: negatives only — mask out the diagonal so yj ≠ xj
        mask = torch.eye(B, dtype=torch.bool, device=sim.device)
        sim_neg = sim.masked_fill(mask, float('-inf'))

        # log Σ_{yj} exp(sim-(R(xi), R(yj)) / τ)
        log_denominator = torch.logsumexp(sim_neg, dim=1)  # (B,)

        # -Σ_{xi,xj ∈ D} log [ exp(sim+/τ) / Σ_{yj} exp(sim-/τ) ]
        loss = (-numerator + log_denominator).mean()
        return loss

    logger(f"Training {model_dir}")
    model.save_pretrained(model_dir)
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=1e-4,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)
    cleanup()
    train_losses, train_plot_x, train_plot_y = [], [], []
    dev_plot_x, dev_plot_y = [], []
    best_dev_loss, steps_since_best = None, 0
    encoder = model.model.encoder
    #attn not needed for contrastive loss w pan et al
    #attn = SimpleAttention()
    for i in tqdm(range(ft_params.num_training_steps)):
        try:
            loss = compute_loss(train_data1.next_batch())
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger("GPU OOM. Cleaning up.", to_stderr=True)
                optimizer.zero_grad(set_to_none=True)
                cleanup()
                continue
            else:
                raise e
        if i > 0 and i % ft_params.report_every == 0:
            avg_train_loss = np.mean(train_losses[-ft_params.report_every :])
            logger(f"Step {i} (train): {avg_train_loss:.4f}")
            train_plot_x.append(i)
            train_plot_y.append(avg_train_loss)
        if i > 0 and i % ft_params.validate_every == 0:
            logger("Validating...")

            def evaluate(dev_data):
                dev_data.restart()
                dev_losses = dict()
                with torch.no_grad():
                    batch = dev_data.next_batch()
                    while batch is not None:
                        sents, lang, goal_encodings, goal_attn_mask = batch
                        loss = compute_loss(batch)
                        if lang not in dev_losses:
                            dev_losses[lang] = []
                        dev_losses[lang].append(loss.item())
                        batch = dev_data.next_batch()
                return {k: np.mean(dev_losses[k]) for k in dev_losses}

            dev_loss = evaluate(dev_data)
            for lang in dev_loss:
                logger(f"Dev loss ({lang}): {dev_loss[lang]:.2f}")
            dev_plot_x.append(i)
            dev_plot_y.append(dev_loss[("europarl", "es-enciphered")] if ("europarl", "es-enciphered") in dev_loss else 0.0)
            plot_losses(
                train_plot_x,
                train_plot_y,
                dev_plot_x,
                dev_plot_y,
                os.path.join(model_dir, "training.png"),
            )
            current_dev_loss = dev_loss.get(("europarl", "es-enciphered"), float("inf"))
            if (
                best_dev_loss is None
                or current_dev_loss < best_dev_loss
            ):
                logger("Saving new best model.")
                best_dev_loss = current_dev_loss
                steps_since_best = 0
                model.save_pretrained(model_dir)
            else:
                steps_since_best += 1
                logger(
                    f"No improvement. Patience: {ft_params.patience - steps_since_best}"
                )
                if steps_since_best >= ft_params.patience:
                    logger("Early stopping.")
                    break


def main():
    parser = argparse.ArgumentParser(description="Finetune NLLB model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Directory to save finetuned model"
    )
    args = parser.parse_args()
    with open(args.config) as reader:
        config = json.load(reader)

    ft_params = read_finetuning_params(config)
    experiment_dir = create_experiment_dir(config, args.config)
    lang_codes = harvest_language_codes(config)
    tokenizer = initialize_tokenizer(config)
    pad_token_id = tokenizer.get_special_tokens()["<pad>"]

    src_cfg = config["corpora"]["europarl"]["es-enciphered"]
    goal_lang_key = next(k for k in config["corpora"]["europarl"] if k != "es-enciphered")
    tgt_cfg = config["corpora"]["europarl"][goal_lang_key]

    alignment_map = build_fast_align_dict_from_raw(
        src_file=src_cfg["train"],
        tgt_file=tgt_cfg["train"],
        tokenizer=tokenizer,
        src_lang=src_cfg["lang_code"],
        tgt_lang=tgt_cfg["lang_code"],
    )

    def build_mixture(split, use_code_switch):
        bitexts = {}
        for bitext_cfg in config["bitexts"]:
            corpus = bitext_cfg["corpus"]
            src_key = (corpus, bitext_cfg["src"])
            tgt_key = (corpus, bitext_cfg["tgt"])
            lines = bitext_cfg.get("train_lines") if split == "train" else None
            raw_bitext = Bitext(
                config["corpora"][corpus][bitext_cfg["src"]][split],
                config["corpora"][corpus][bitext_cfg["tgt"]][split],
                lines=lines,
            )
            tokenized = TokenizedBitext(
                raw_bitext, tokenizer, lang_codes[src_key], lang_codes[tgt_key]
            )
            if use_code_switch and bitext_cfg["src"] == "es-enciphered":
                iterable = CodeswitchedBitext(tokenized, alignment_map)
            else:
                iterable = tokenized
            bitexts[(src_key, tgt_key)] = BatchedBitext(iterable, batch_size=ft_params.batch_size, src_pad_token=pad_token_id, tgt_pad_token=pad_token_id)
        return MixtureOfBitexts(bitexts, only_once_thru=(split != "train"))

    train_data = build_mixture("train", use_code_switch=True)
    dev_data = build_mixture("dev", use_code_switch=False)

    static_model = prepare_model_for_finetuning(ft_params)
    model = prepare_model_for_finetuning(ft_params)

    train_mix = TokenizedMixtureOfTextAndGoalEncoding(
        train_data, static_model.model.encoder, pad_token_id=pad_token_id
    )
    dev_mix = TokenizedMixtureOfTextAndGoalEncoding(
        dev_data, static_model.model.encoder, pad_token_id=pad_token_id
    )

    finetune(
        model,
        train_mix,
        dev_mix,
        experiment_dir,
        ft_params,
        goal_lang_key=goal_lang_key,
    )
    evaluate_experiment(experiment_dir)


if __name__ == "__main__":
    main()
