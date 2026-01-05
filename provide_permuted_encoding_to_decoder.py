from configure import harvest_language_codes
from configure import initialize_tokenizer
from configure import read_finetuning_params
from corpora import MixtureOfBitexts, TokenizedMixtureOfBitexts
from myutil import prepare_model_for_finetuning
import torch


def directly_provide_encoding_state(model, dev_data, permute=False):
    model.eval()
    encoder = model.model.encoder
    decoder = model.model.decoder
    with torch.no_grad():
        x, _, _, _ = dev_data.next_batch()
        x = x.to(model.device)
        x_encoding = encoder(**x)
        encoder_states = x_encoding.last_hidden_state
        if permute:
            encoder_states = encoder_states.flip(1)
        encoder_attn_mask = x["attention_mask"].float()
        input_ids = torch.tensor([[2, 256057], [2, 256057]]).to(model.device)
        decoder_output = decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_states,
            encoder_attention_mask=encoder_attn_mask,
        ).last_hidden_state
        out = model.lm_head(decoder_output)
        probs = torch.softmax(out, dim=-1)
        probs0 = probs[0, -1, :].tolist()
        indexed_probs = sorted(enumerate(probs0), key=lambda x: -x[1])
        print(indexed_probs[:5])


def main():
    config = {
        "model_dir": "experiments/assume",
        "corpora": {
            "l1-l2": {
                "lang1": {
                    "lang_code": "eng_Latn",
                    "train": "test_files/lang1.txt",
                    "dev": "test_files/dev.lang1",
                    "test": "test_files/test.lang1",
                    "permutation": 0,
                },
                "lang2": {
                    "lang_code": "fra_Latn",
                    "train": "test_files/lang2.txt",
                    "dev": "test_files/dev.lang2",
                    "test": "test_files/test.lang2",
                    "permutation": 1,
                },
            }
        },
        "bitexts": [
            {"corpus": "l1-l2", "src": "lang1", "tgt": "lang2", "train_lines": [0, 8]}
        ],
        "finetuning_parameters": {
            "base_model": "facebook/nllb-200-distilled-600M",
            "batch_size": 2,
            "num_steps": 1002,
            "freeze_encoder": False,
        },
    }
    ft_params = read_finetuning_params(config)
    lang_codes = harvest_language_codes(config)
    tokenizer = initialize_tokenizer(config)
    dev_data = MixtureOfBitexts.create_from_config(config, "dev", only_once_thru=False)
    tokenized_dev = TokenizedMixtureOfBitexts(
        dev_data, tokenizer, lang_codes=lang_codes
    )
    model = prepare_model_for_finetuning(ft_params)
    directly_provide_encoding_state(model, tokenized_dev)


if __name__ == "__main__":
    main()
