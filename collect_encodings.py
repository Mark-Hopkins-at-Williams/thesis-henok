import argparse
import gc
import json
import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import shutil
import sys
import torch
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    get_constant_schedule_with_warmup,
)
from configure import USE_CUDA
from corpora import MixtureOfBitexts, TokenizedMixtureOfBitexts, load_tokenizer
from permutations import (
    create_random_permutation_with_fixed_points,
    save_permutation_map,
)
from validate import translate_tokenized_mixture_of_bitexts, evaluate_translations


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def prepare_model(base_model: str, freeze_decoder: bool, freeze_encoder: bool, should_finetune: bool):
    if should_finetune:
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model) 
        print('loaded pretrained model')
    else: 
        model_config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForSeq2SeqLM.from_config(model_config)
        print('loaded architecture only')
    if hasattr(model.config, "max_length"):  # this should be in a GenerationConfig
        delattr(model.config, "max_length")
    if freeze_decoder:
        print("--> DECODER FROZEN <--")
        for param in model.get_decoder().parameters():
            param.requires_grad = False
    else:
        print("--> decoder NOT frozen <--")
    if freeze_encoder:
        print("--> ENCODER FROZEN <--")
        for param in model.get_encoder().parameters():
            param.requires_grad = False
    else:
        print("--> encoder NOT frozen <--")
    if USE_CUDA:
        torch.cuda.set_device(0)
        model.cuda()
    return model


def evaluate(model, dev_data, batches: int = 100):
    
    model.eval()
    encoder = model.model.encoder
    print('hi')
    with torch.no_grad():
        for _ in range(batches):
            x, y, _, _ = dev_data.next_batch()
            x = x.to(model.device)
            y = y.to(model.device)
            x_encoding = encoder(**x)
            y_encoding = encoder(**y)
            xq = x_encoding.last_hidden_state[0].cpu().numpy()
            xb = y_encoding.last_hidden_state[0].cpu().numpy()
            
            index = faiss.IndexFlatL2(1024)
            index.add(xb)
            k = 1
            D, I = index.search(xq, k)  # D = distances, I = indices of nearest neighbors
            print(I) 
            print(D)
            exit()



def main():
    parser = argparse.ArgumentParser(description="Finetune NLLB model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Directory to save finetuned model"
    )
    args = parser.parse_args()

    with open(args.config) as reader:
        config = json.load(reader)
          
    lang_codes = dict()        
    for corpus in config['corpora']:
        for key in config['corpora'][corpus]:
            lang_codes[(corpus, key)] = config['corpora'][corpus][key]['lang_code']
    model_name = "facebook/nllb-200-distilled-600M"
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if USE_CUDA:
        model.cuda()

    
    dev_data = MixtureOfBitexts.create_from_config(config, "dev", only_once_thru=True)
    tokenizer = load_tokenizer(model_name)

    tokenized_dev = TokenizedMixtureOfBitexts(
        dev_data, tokenizer, max_length=128, lang_codes=lang_codes, permutation_map=dict()
    )
    
    
    
    evaluate(model, tokenized_dev, batches=2)
    
    
if __name__ == "__main__":
    main()
