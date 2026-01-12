import argparse
from attention import SimpleAttention
from configure import create_experiment_dir
from configure import create_permutations
from configure import harvest_language_codes
from configure import initialize_tokenizer
from configure import read_finetuning_params
from corpora import MixtureOfBitexts, TokenizedMixtureOfBitexts
from corpora import TokenizedMixtureOfTextAndGoalEncoding
import json
import matplotlib
import matplotlib.pyplot as plt
from myutil import cleanup
from myutil import logger
from myutil import prepare_model_for_finetuning
import numpy as np
import os
from pathlib import Path
from permutations import save_permutation_map
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


def finetune(model, train_data, dev_data, model_dir, ft_params):
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
    for i in tqdm(range(ft_params.num_training_steps)):
        try:
            encoder.eval()
            sents, lang, goal_encodings = train_data.next_batch()
            sents = sents.to(encoder.device)
            goal_encodings = goal_encodings.to(encoder.device)
            sent_encodings = encoder(**sents).last_hidden_state
            loss = ((sent_encodings - goal_encodings) ** 2).mean()

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
                encoder.eval()
                dev_data.restart()
                dev_losses = dict()
                with torch.no_grad():
                    batch = dev_data.next_batch()
                    while batch is not None:
                        sents, lang, goal_encodings = batch
                        sents = sents.to(encoder.device)
                        goal_encodings = goal_encodings.to(encoder.device)
                        sent_encodings = encoder(**sents).last_hidden_state
                        loss = ((sent_encodings - goal_encodings) ** 2).mean()
                        if lang not in dev_losses:
                            dev_losses[lang] = []
                        dev_losses[lang].append(loss.item())
                        batch = dev_data.next_batch()
                return {k: np.mean(dev_losses[k]) for k in dev_losses}

            dev_loss = evaluate(dev_data)
            for lang in dev_loss:
                logger(f"Dev loss ({lang}): {dev_loss[lang]:.2f}")
            dev_plot_x.append(i)
            dev_plot_y.append(dev_loss[("europarl", "es-enciphered")])
            plot_losses(
                train_plot_x,
                train_plot_y,
                dev_plot_x,
                dev_plot_y,
                os.path.join(model_dir, "training.png"),
            )
            if (
                best_dev_loss is None
                or dev_loss[("europarl", "es-enciphered")] < best_dev_loss
            ):
                logger("Saving new best model.")
                best_dev_loss = dev_loss[("europarl", "es-enciphered")]
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
    pmap = create_permutations(config, tokenizer)
    save_permutation_map(pmap, Path(experiment_dir) / "permutations.json")
    train_data = MixtureOfBitexts.create_from_config(
        config, "train", only_once_thru=False
    )
    dev_data = MixtureOfBitexts.create_from_config(config, "dev", only_once_thru=True)
    tokenized_train = TokenizedMixtureOfBitexts(
        train_data,
        tokenizer,
        lang_codes=lang_codes,
        permutation_map=pmap,
        use_alt_pad_token_for_tgt_lang=False,
    )
    tokenized_train2 = TokenizedMixtureOfBitexts(
        train_data,
        tokenizer,
        lang_codes=lang_codes,
        permutation_map=pmap,
        use_alt_pad_token_for_tgt_lang=False,
    )
    tokenized_dev = TokenizedMixtureOfBitexts(
        dev_data,
        tokenizer,
        lang_codes=lang_codes,
        permutation_map=pmap,
        use_alt_pad_token_for_tgt_lang=False,
    )

    static_model = prepare_model_for_finetuning(ft_params)
    model = prepare_model_for_finetuning(ft_params)

    train_mix = TokenizedMixtureOfTextAndGoalEncoding(
        tokenized_train, static_model.model.encoder
    )
    dev_mix = TokenizedMixtureOfTextAndGoalEncoding(
        tokenized_dev, static_model.model.encoder
    )

    finetune(
        model,
        train_mix,
        dev_mix,
        experiment_dir,
        ft_params,
    )
    evaluate_experiment(experiment_dir)


if __name__ == "__main__":
    main()
