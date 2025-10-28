import sys
import os
sys.path.append(os.path.abspath(".."))
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import faiss
from corpora import MixtureOfBitexts, TokenizedMixtureOfBitexts, load_tokenizer
from transformers import AutoModelForSeq2SeqLM

USE_CUDA = torch.cuda.is_available()


def compute_language_embeddings(model, tokenized_data, lang_codes):
    """
    Collect fine-grained token-level embeddings for the first sentence in each batch.
    Returns a dict: lang_code -> numpy array of shape (total_tokens, hidden_dim)
    """
    model.eval()
    encoder = model.model.encoder
    embeddings = {lang: [] for lang in lang_codes.values()}

    with torch.no_grad():
        batch = tokenized_data.next_batch()
        while batch is not None:
            x, y, src_lang, tgt_lang = batch

            # Source embeddings: first sentence in batch, all tokens
            x = x.to(model.device)
            x_enc = encoder(**x).last_hidden_state[0].cpu().numpy()  # [seq_len, hidden_dim]
            embeddings[lang_codes[src_lang]].append(x_enc)

            # Target embeddings: first sentence in batch, all tokens
            y = y.to(model.device)
            y_enc = encoder(**y).last_hidden_state[0].cpu().numpy()
            embeddings[lang_codes[tgt_lang]].append(y_enc)

            batch = tokenized_data.next_batch()

    # Concatenate all batches per language
    for lang in embeddings:
        if embeddings[lang]:
            embeddings[lang] = np.vstack(embeddings[lang])
        else:
            embeddings[lang] = np.zeros((1, model.config.d_model))
    return embeddings


def compute_faiss_heatmap(embeddings, lang_list):
    """
    embeddings: dict lang_code -> (num_tokens, hidden_dim)
    lang_list: ordered list of languages
    Returns a matrix of average nearest-neighbor distances (not symmetric).
    """
    n = len(lang_list)
    heatmap = np.zeros((n, n))

    for i, lang_i in enumerate(lang_list):
        xb = embeddings[lang_i].astype('float32')
        index = faiss.IndexFlatL2(xb.shape[1])
        index.add(xb)

        for j, lang_j in enumerate(lang_list):
            xq = embeddings[lang_j].astype('float32')
            D, I = index.search(xq, 1)  # nearest neighbor distances
            avg_distance = D.mean()
            heatmap[i, j] = avg_distance

    return heatmap


def plot_heatmap(matrix, labels, out_file="language_heatmap.png"):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(matrix, cmap="viridis")
    plt.colorbar(im)
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.yticks(np.arange(len(labels)), labels)
    plt.title("FAISS Avg Nearest-Neighbor L2 Distances (Fine-Grained)")
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Saved heatmap to {out_file}")


def main():
    # Load config JSON
    with open("config.json") as f:
        config = json.load(f)

    # Extract language codes
    lang_codes = {
        (c, k): config['corpora'][c][k]['lang_code']
        for c in config['corpora'] for k in config['corpora'][c]
    }
    LANGS = list(lang_codes.values())

    # Load model
    model_name = config["finetuning_parameters"]["base_model"]
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if USE_CUDA:
        model.cuda()

    # Load dev data and tokenize
    dev_data = MixtureOfBitexts.create_from_config(config, "dev", only_once_thru=True)
    tokenizer = load_tokenizer(model_name)
    tokenized_dev = TokenizedMixtureOfBitexts(dev_data, tokenizer, max_length=128,
                                              lang_codes=lang_codes, permutation_map={})

    # Compute fine-grained embeddings
    embeddings = compute_language_embeddings(model, tokenized_dev, lang_codes)

    # Compute FAISS heatmap
    heatmap = compute_faiss_heatmap(embeddings, LANGS)

    # Plot heatmap
    plot_heatmap(heatmap, LANGS, out_file="faiss_language_heatmap.png")


if __name__ == "__main__":
    main()
