from configure import (
    harvest_language_codes,
    initialize_tokenizer,
    read_finetuning_params,
)
from corpora import MixtureOfBitexts, TokenizedMixtureOfBitexts
from myutil import prepare_model_for_finetuning
import torch
import matplotlib.pyplot as plt


# ---------------------------
# Token stats function
# ---------------------------
def compute_token_stats(encoder_states_gold, encoder_states_corrupted, attention_mask):
    medians_gold = []
    medians_corr = []
    vars_gold = []
    vars_corr = []

    B, T, H = encoder_states_gold.shape

    for b in range(B):
        mask_b = attention_mask[b].bool()
        tokens_gold = encoder_states_gold[b][mask_b]  # [num_tokens, H]
        tokens_corr = encoder_states_corrupted[b][mask_b]

        # flatten tokens across hidden dims
        tokens_gold_flat = tokens_gold.reshape(-1)
        tokens_corr_flat = tokens_corr.reshape(-1)

        medians_gold.append(tokens_gold_flat.median().item())
        medians_corr.append(tokens_corr_flat.median().item())
        vars_gold.append(tokens_gold_flat.var().item())
        vars_corr.append(tokens_corr_flat.var().item())

    return medians_gold, medians_corr, vars_gold, vars_corr


# ---------------------------
# Config
# ---------------------------
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

# ---------------------------
# Setup model and tokenizer
# ---------------------------
ft_params = read_finetuning_params(config)
lang_codes = harvest_language_codes(config)
tokenizer = initialize_tokenizer(config)
model = prepare_model_for_finetuning(ft_params)
model.eval()

# ---------------------------
# Load dev data
# ---------------------------
dev_data = MixtureOfBitexts.create_from_config(config, "dev", only_once_thru=True)
tokenized_dev = TokenizedMixtureOfBitexts(dev_data, tokenizer, lang_codes=lang_codes)

# ---------------------------
# Prepare storage
# ---------------------------
all_medians_gold = []
all_medians_corr = []
all_vars_gold = []
all_vars_corr = []

sigma = 0.1  # example noise level

# ---------------------------
# Loop through batches
# ---------------------------
with torch.no_grad():
    batch = tokenized_dev.next_batch()
    while batch is not None:
        x, _, _, _ = batch
        x = x.to(model.device)

        out = model.model.encoder(**x)
        encoder_states_gold = out.last_hidden_state
        noise = sigma * torch.randn_like(encoder_states_gold)
        encoder_states_corrupted = encoder_states_gold + noise
        attention_mask = x["attention_mask"]

        medians_gold, medians_corr, vars_gold, vars_corr = compute_token_stats(
            encoder_states_gold, encoder_states_corrupted, attention_mask
        )

        all_medians_gold.extend(medians_gold)
        all_medians_corr.extend(medians_corr)
        all_vars_gold.extend(vars_gold)
        all_vars_corr.extend(vars_corr)

        batch = tokenized_dev.next_batch()

# ---------------------------
# Plotting
# ---------------------------
plt.figure(figsize=(8, 5))
plt.plot(all_medians_gold, label="Gold median", alpha=0.7)
plt.plot(all_medians_corr, label="Corrupted median", alpha=0.7)
plt.xlabel("Sentence index")
plt.ylabel("Token median (all dims)")
plt.title("Token medians: Gold vs Corrupted")
plt.legend()
plt.grid(True)
plt.savefig("token_medians.png", bbox_inches="tight")

plt.figure(figsize=(8, 5))
plt.plot(all_vars_gold, label="Gold variance", alpha=0.7)
plt.plot(all_vars_corr, label="Corrupted variance", alpha=0.7)
plt.xlabel("Sentence index")
plt.ylabel("Token variance (all dims)")
plt.title("Token variance: Gold vs Corrupted")
plt.legend()
plt.grid(True)
plt.savefig("token_variances.png", bbox_inches="tight")

print("Saved token_medians.png and token_variances.png")
