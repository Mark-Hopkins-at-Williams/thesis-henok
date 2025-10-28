import sys
import os
sys.path.append(os.path.abspath(".."))
import json
import torch
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import faiss
from corpora import MixtureOfBitexts, TokenizedMixtureOfBitexts, load_tokenizer
from transformers import AutoModelForSeq2SeqLM
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


USE_CUDA = torch.cuda.is_available()


def compute_language_embeddings(model, tokenized_data, lang_codes):
    """
    Collect fine-grained token-level embeddings for the first sentence in each batch.
    Returns a dict: lang_code -> numpy array of shape (total_tokens, hidden_dim)
    """
    model.eval()
    encoder = model.model.encoder
    embeddings = defaultdict(list)

    with torch.no_grad():
        batch = tokenized_data.next_batch()
        while batch is not None:
            x, _, src_lang, _ = batch
            
            # Source embeddings: first (and only) sentence in batch, all tokens
            x = x.to(model.device)
            x_enc = encoder(**x).last_hidden_state[0].cpu().numpy()  # [seq_len, hidden_dim]
            embeddings[lang_codes[src_lang]].append(x_enc)           
            
            batch = tokenized_data.next_batch()
    return dict(embeddings)


def compute_faiss_heatmap(embeddings):
    """
    embeddings: dict lang_code -> (num_tokens, hidden_dim)
    lang_list: ordered list of languages
    Returns a matrix of average nearest-neighbor distances (not symmetric).
    """
    lang_list = embeddings.keys()
    n = len(lang_list)
    distances = defaultdict(list)
    heatmap = np.zeros((n, n))

    for i, lang_i in tqdm(enumerate(lang_list)):        
        for sent_index in range(len(embeddings[lang_i])):
            xb = embeddings[lang_i][sent_index]        
            index = faiss.IndexFlatL2(xb.shape[1])
            index.add(xb)
            for j, lang_j in enumerate(lang_list):
                xq = embeddings[lang_j][sent_index].astype('float32')
                D, I = index.search(xq, 1)  # nearest neighbor distances
                avg_distance = D.mean()
                distances[(i, j)].append(avg_distance)
    for (i, j) in distances:
        heatmap[i, j] = sum(distances[(i,j)]) / len(distances[(i,j)])        

    return heatmap, lang_list



def plot_clustermap(matrix, labels, out_file="language_clustermap.png"):
    # Convert the matrix to a DataFrame for labeled axes
    import pandas as pd
    df = pd.DataFrame(matrix, index=labels, columns=labels)

    # Create the clustermap
    g = sns.clustermap(
        df,
        cmap="viridis",
        xticklabels=True,
        yticklabels=True,
        linewidths=0.5,
    )

    # Customize titles and layout
    plt.suptitle("FAISS Avg Nearest-Neighbor L2 Distances (Fine-Grained)", y=1.02)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

    # Save and show
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Saved clustermap to {out_file}")



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
    print('computing embeddings...')
    embeddings = compute_language_embeddings(model, tokenized_dev, lang_codes)

    # Compute FAISS heatmap
    print('computing heatmap...')
    heatmap, lang_list = compute_faiss_heatmap(embeddings)

    # Plot heatmap
    plot_clustermap(heatmap, lang_list, out_file="faiss_language_heatmap.png")


if __name__ == "__main__":
    main()
