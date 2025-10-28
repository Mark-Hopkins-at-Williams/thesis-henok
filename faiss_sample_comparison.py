import json
import faiss
import torch
import matplotlib.pyplot as plt
import numpy as np
from corpora import MixtureOfBitexts, TokenizedMixtureOfBitexts, load_tokenizer
from transformers import AutoModelForSeq2SeqLM

USE_CUDA = torch.cuda.is_available()

# Selected languages
LANGS = ["eng_Latn", "ary_Arab", "dzo_Tibt", "mag_Deva", "knc_Latn", "shn_Mymr"]

def mean_encoder_output(model, tokenized_data, lang_code):
    """Compute mean pooled encoder output for a language."""
    model.eval()
    encoder = model.model.encoder
    embeddings = []
    with torch.no_grad():
        batch = tokenized_data.next_batch()
        while batch is not None:
            x, y, src_lang, tgt_lang = batch
            # Check source language
            if src_lang == lang_code:
                x = x.to(model.device)
                x_enc = encoder(**x)
                # mean pooling over sequence
                embeddings.append(x_enc.last_hidden_state.mean(dim=1).cpu())
            batch = tokenized_data.next_batch()
    if len(embeddings) == 0:
        return None
    return torch.cat(embeddings, dim=0).mean(dim=0).numpy()  # final mean vector

def compute_distance_matrix(model, tokenized_data):
    vectors = {}
    for lang in LANGS:
        vectors[lang] = mean_encoder_output(model, tokenized_data, lang)
    n = len(LANGS)
    dist_matrix = np.zeros((n, n))
    for i, lang_i in enumerate(LANGS):
        for j, lang_j in enumerate(LANGS):
            # L2 distance between mean vectors
            dist_matrix[i, j] = np.linalg.norm(vectors[lang_i] - vectors[lang_j])
    return dist_matrix

def plot_heatmap(matrix, labels, out_file="heatmap.png"):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(matrix, cmap="viridis")
    plt.colorbar(im)
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.yticks(np.arange(len(labels)), labels)
    plt.title("Mean L2 Distance between Languages")
    plt.tight_layout()
    plt.savefig(out_file)

def main():
    # Load model
    model_name = "facebook/nllb-200-distilled-600M"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if USE_CUDA:
        model.cuda()

    # Load config & dev data
    with open("config.json") as f:
        config = json.load(f)
    lang_codes = {
        (c, k): config['corpora'][c][k]['lang_code'] 
        for c in config['corpora'] for k in config['corpora'][c]
    }
    dev_data = MixtureOfBitexts.create_from_config(config, "dev", only_once_thru=True)
    tokenizer = load_tokenizer(model_name)
    tokenized_dev = TokenizedMixtureOfBitexts(dev_data, tokenizer, max_length=128, lang_codes=lang_codes, permutation_map={})

    # Compute distance matrix
    dist_matrix = compute_distance_matrix(model, tokenized_dev)

    # Plot
    plot_heatmap(dist_matrix, LANGS, out_file="language_heatmap.png")

if __name__ == "__main__":
    main()
