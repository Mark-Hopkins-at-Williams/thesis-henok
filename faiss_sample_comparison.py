from configure import (
    harvest_language_codes,
    initialize_tokenizer,
    read_finetuning_params,
)
from validate import evaluate_translations
from corpora import MixtureOfBitexts, TokenizedMixtureOfBitexts
from myutil import prepare_model_for_finetuning

import sys
import torch
import matplotlib.pyplot as plt
import faiss
import numpy as np


def translate_with_noise_and_faiss_similarity(
    model,
    tokenizer,
    dev_data,
    target_lang_code,
    batch_size,
    sigma,
    max_length=128,
):
    """
    Returns:
      - gold translations
      - corrupted translations
      - avg FAISS NN distance per sentence
    """
    model.eval()
    encoder = model.model.encoder
    decoder = model.model.decoder

    gold_translations = []
    corrupted_translations = []
    faiss_sims = []

    counter = 0
    with torch.no_grad():
        batch = dev_data.next_batch()
        while batch is not None:
            print(f"batch {counter}!")
            sys.stdout.flush()
            counter += 1

            x, _, _, _ = batch
            x = x.to(model.device)

            # ----- encoder -----
            out = encoder(**x)
            encoder_states_gold = out.last_hidden_state  # [B, T, H]

            noise = sigma * torch.randn_like(encoder_states_gold)
            encoder_states_corrupted = encoder_states_gold + noise

            attention_mask = x["attention_mask"]  # [B, T]

            # ----- FAISS similarity (sentence-level) -----
            for b in range(encoder_states_gold.size(0)):
                mask_b = attention_mask[b].bool()

                gold_tokens = (
                    encoder_states_gold[b][mask_b]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype("float32")
                )
                corr_tokens = (
                    encoder_states_corrupted[b][mask_b]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype("float32")
                )

                # Build FAISS index on gold tokens
                index = faiss.IndexFlatL2(gold_tokens.shape[1])
                index.add(gold_tokens)

                # Query with corrupted tokens
                D, _ = index.search(corr_tokens, 1)
                avg_nn_dist = float(D.mean())

                faiss_sims.append(avg_nn_dist)

            encoder_attn_mask = attention_mask.float()

            def decode(encoder_states):
                input_ids = torch.tensor([[2, target_lang_code]] * batch_size).to(
                    model.device
                )
                output_complete = torch.tensor([1] * batch_size).to(model.device)

                while 1 in output_complete and input_ids.shape[-1] < max_length:
                    decoder_out = decoder(
                        input_ids=input_ids,
                        encoder_hidden_states=encoder_states,
                        encoder_attention_mask=encoder_attn_mask,
                    ).last_hidden_state

                    logits = model.lm_head(decoder_out)
                    next_ids = torch.argmax(logits, dim=-1)[:, -1]

                    next_ids = next_ids * output_complete + (output_complete == 0)
                    output_complete = output_complete * (next_ids != 2)

                    input_ids = torch.cat([input_ids, next_ids.unsqueeze(1)], dim=1)

                return tokenizer.batch_decode(input_ids.to("cpu"))

            gold_translations.extend(decode(encoder_states_gold))
            corrupted_translations.extend(decode(encoder_states_corrupted))

            batch = dev_data.next_batch()

    return gold_translations, corrupted_translations, faiss_sims


def main():
    # ---- config ----
    config = {
        "model_dir": "experiments/assume",
        "corpora": {
            "l1-l2": {
                "lang1": {
                    "lang_code": "srd_Latn",
                    "train": "/mnt/storage/data/flores/dev.srd_Latn",
                    "dev": "/mnt/storage/data/flores/dev.srd_Latn",
                    "test": "/mnt/storage/data/flores/test.srd_Latn",
                    "permutation": 0,
                },
                "lang2": {
                    "lang_code": "eng_Latn",
                    "train": "/mnt/storage/data/flores/dev.eng_Latn",
                    "dev": "/mnt/storage/data/flores/dev.eng_Latn",
                    "test": "/mnt/storage/data/flores/test.eng_Latn",
                    "permutation": 0,
                },
            }
        },
        "bitexts": [{"corpus": "l1-l2", "src": "lang1", "tgt": "lang2"}],
        "finetuning_parameters": {
            "base_model": "facebook/nllb-200-distilled-600M",
            "batch_size": 32,
            "num_steps": 1002,
            "freeze_encoder": False,
        },
    }

    ft_params = read_finetuning_params(config)
    lang_codes = harvest_language_codes(config)
    tokenizer = initialize_tokenizer(config)
    model = prepare_model_for_finetuning(ft_params)

    # ---- experiment sweep ----
    sigmas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

    bleu_by_sigma = []
    faiss_all = []
    bleu_all = []

    for s in sigmas:
        print(f"\nRunning noise σ={s}")

        dev_data = MixtureOfBitexts.create_from_config(
            config, "dev", only_once_thru=True
        )
        tokenized_dev = TokenizedMixtureOfBitexts(
            dev_data, tokenizer, lang_codes=lang_codes
        )

        gold, corrupted, faiss_sims = translate_with_noise_and_faiss_similarity(
            model,
            tokenizer,
            tokenized_dev,
            target_lang_code=256047,  # eng_Latn
            batch_size=ft_params.batch_size,
            sigma=s,
        )

        metrics = evaluate_translations(corrupted, gold)
        bleu = metrics["bleu"] if isinstance(metrics, dict) else metrics

        bleu_by_sigma.append(bleu)
        faiss_all.extend(faiss_sims)
        bleu_all.extend([bleu] * len(faiss_sims))

    # ---- plot: BLEU vs sigma ----
    plt.figure()
    plt.plot(sigmas, bleu_by_sigma, marker="o")
    plt.xlabel("Sigma (noise std)")
    plt.ylabel("BLEU")
    plt.title("BLEU vs Noise Level (σ)")
    plt.grid(True)
    plt.savefig("bleu_vs_sigma.png", bbox_inches="tight")

    # ---- plot: BLEU vs FAISS similarity ----
    plt.figure()
    plt.scatter(faiss_all, bleu_all, alpha=0.35)
    plt.xlabel("Avg FAISS NN Distance (token-level)")
    plt.ylabel("BLEU")
    plt.title("BLEU vs Encoder FAISS Similarity")
    plt.grid(True)
    plt.savefig("bleu_vs_faiss.png", bbox_inches="tight")

    print("\nSaved:")
    print(" - bleu_vs_sigma.png")
    print(" - bleu_vs_faiss.png")


if __name__ == "__main__":
    main()
