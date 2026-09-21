import argparse
from configure import create_experiment_dir
from configure import read_finetuning_params
from configure import create_bitexts
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
import random
import torch
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


def finetune(model, train_data, dev_data, model_dir, ft_params, resume=False):
    logger(f"Training {model_dir}")

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if ft_params.should_finetune:
        optimizer = Adafactor(
            [p for p in model.parameters() if p.requires_grad],
            scale_parameter=False,
            relative_step=False,
            lr=1e-4,
            clip_threshold=1.0,
            weight_decay=1e-3,
        )

        num_warmup_steps = int(0.05 * ft_params.num_training_steps)
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps
        )
    else:
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            lr=None,
            clip_threshold=1.0,
            weight_decay=0.01,
        )
        scheduler = None

    accumulation_steps = ft_params.gradient_accumulation_steps
    optimizer.zero_grad(set_to_none=True)

    def evaluate(model, dev_data):
        dev_data.restart()
        model.eval()
        try:
            losses = []
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                for x, y, _ in tqdm(dev_data):
                    x = {k: v.to(model.device) for k, v in x.items()}
                    y = {k: v.to(model.device) for k, v in y.items()}
                    loss = model(**x, labels=y["input_ids"]).loss
                    losses.append(loss.item())
        finally:
            model.train()
        return float(np.mean(losses))

    cleanup()
    train_losses = []
    train_plot_x, train_plot_y = [], []
    dev_plot_x, dev_plot_y = [], []

    best_dev_loss = None
    steps_since_best = 0
    start_step = 0
    checkpoint_path = os.path.join(model_dir, "checkpoint.pt")

    def save_checkpoint(step):
        state = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_dev_loss": best_dev_loss,
            "steps_since_best": steps_since_best,
            "train_plot": (train_plot_x, train_plot_y),
            "dev_plot": (dev_plot_x, dev_plot_y),
            "recent_train_losses": train_losses[-ft_params.report_every :],
            "data_position": getattr(train_data, "position", None),
        }
        torch.save(state, checkpoint_path + ".tmp")
        os.replace(checkpoint_path + ".tmp", checkpoint_path)  # never half-written

    if resume:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"nothing to resume from: {checkpoint_path}")
        if not hasattr(train_data, "seek"):
            raise ValueError("cannot resume: the training data cannot be repositioned")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        if scheduler is not None and state["scheduler"] is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_step = state["step"]
        best_dev_loss = state["best_dev_loss"]
        steps_since_best = state["steps_since_best"]
        train_plot_x, train_plot_y = state["train_plot"]
        dev_plot_x, dev_plot_y = state["dev_plot"]
        train_losses = list(state["recent_train_losses"])
        train_data.seek(state["data_position"])
        del state
        cleanup()
        logger(f"Resumed from step {start_step}")

    model.train()
    train_iter = iter(train_data)

    for step in tqdm(
        range(start_step + 1, ft_params.num_training_steps + 1),
        initial=start_step,
        total=ft_params.num_training_steps,
    ):
        try:
            x, y, _ = next(train_iter)
            x = {k: v.to(model.device) for k, v in x.items()}
            y = {k: v.to(model.device) for k, v in y.items()}
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(**x, labels=y["input_ids"]).loss
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            train_losses.append(loss.item() * accumulation_steps)
            if step % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), ft_params.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger(
                    f"\nGPU OOM during training step. Skipping batch ({x['input_ids'].shape}).",
                    to_stderr=True,
                )
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                continue
            else:
                raise

        if step % ft_params.report_every == 0:  # logging
            avg_train_loss = np.mean(train_losses[-ft_params.report_every :])
            logger(f"Step {step} (train): {avg_train_loss:.4f}")
            train_plot_x.append(step)
            train_plot_y.append(avg_train_loss)

        if step % ft_params.validate_every == 0:  # validation
            logger("Validating...")
            dev_loss = evaluate(model, dev_data)
            logger(f"Dev loss: {dev_loss:.4f}")

            dev_plot_x.append(step)
            dev_plot_y.append(dev_loss)

            plot_losses(
                train_plot_x,
                train_plot_y,
                dev_plot_x,
                dev_plot_y,
                os.path.join(model_dir, "training.png"),
            )

            if best_dev_loss is None or dev_loss < best_dev_loss:
                logger("Saving new best model.")
                best_dev_loss = dev_loss
                steps_since_best = 0
                model.save_pretrained(model_dir)
            else:
                steps_since_best += 1
                logger(
                    f"No improvement. Patience: "
                    f"{ft_params.patience - steps_since_best}"
                )
                if steps_since_best >= ft_params.patience:
                    logger("Early stopping.")
                    break

        if ft_params.checkpoint_every and step % ft_params.checkpoint_every == 0:
            save_checkpoint(step)


def main():
    parser = argparse.ArgumentParser(description="Finetune NLLB model.")
    parser.add_argument("--config", type=str, help="Experiment configuration (json)")
    parser.add_argument(
        "--resume",
        type=str,
        help="Existing experiment directory to continue from its checkpoint.pt",
    )
    args = parser.parse_args()
    if args.resume is None and args.config is None:
        parser.error("give --config, or --resume with an experiment directory")
    config_file = (
        Path(args.resume) / "experiment.json" if args.resume else args.config
    )
    with open(config_file) as reader:
        config = json.load(reader)

    ft_params = read_finetuning_params(config)
    if ft_params.seed is not None:
        random.seed(ft_params.seed)
        np.random.seed(ft_params.seed)
        torch.manual_seed(ft_params.seed)
    if args.resume:
        experiment_dir = args.resume
    else:
        experiment_dir = create_experiment_dir(config, args.config)
    bitexts = create_bitexts(config)
    if not args.resume:
        save_permutation_map(
            bitexts["cipher_map"], Path(experiment_dir) / "ciphers.json"
        )
    model = prepare_model_for_finetuning(ft_params)
    if model.config.vocab_size < bitexts.get("required_vocab", 0):
        raise ValueError(
            f"model vocabulary ({model.config.vocab_size}) is smaller than the data "
            f"needs ({bitexts['required_vocab']}); set config_overrides.vocab_size"
        )
    finetune(
        model,
        bitexts["train"],
        bitexts["dev"],
        experiment_dir,
        ft_params,
        resume=bool(args.resume),
    )
    evaluate_experiment(experiment_dir)


if __name__ == "__main__":
    main()
