from configure import harvest_language_codes
from configure import initialize_tokenizer
from configure import read_finetuning_params
from validate import evaluate_translations
from corpora import MixtureOfBitexts, TokenizedMixtureOfBitexts
from myutil import prepare_model_for_finetuning
import sys
import torch


def directly_provide_encoding_state(
    model,
    tokenizer,
    dev_data,
    target_lang_code,
    batch_size,
    noise_variance=0.0,
    max_length=128,
):
    model.eval()
    encoder = model.model.encoder
    decoder = model.model.decoder
    translations = []
    counter = 0
    with torch.no_grad():
        batch = dev_data.next_batch()
        while batch is not None:
            print(f"batch {counter}!")
            sys.stdout.flush()
            counter += 1
            x, _, _, _ = batch
            x = x.to(model.device)
            x_encoding = encoder(**x)
            encoder_states = x_encoding.last_hidden_state
            sigma = noise_variance**0.5  # standard deviation
            noise = sigma * torch.randn_like(encoder_states)
            encoder_states = encoder_states + noise
            encoder_attn_mask = x["attention_mask"].float()
            input_ids = torch.tensor([[2, target_lang_code]] * batch_size).to(
                model.device
            )
            output_complete = torch.tensor([1] * batch_size).to(model.device)
            while 1 in output_complete and input_ids.shape[-1] < max_length:
                decoder_output = decoder(
                    input_ids=input_ids,
                    encoder_hidden_states=encoder_states,
                    encoder_attention_mask=encoder_attn_mask,
                ).last_hidden_state
                out = model.lm_head(decoder_output)
                next_ids = torch.argmax(out, dim=-1)[:, -1]
                next_ids = next_ids * output_complete + (output_complete == 0)
                output_complete = output_complete * (next_ids != 2)
                input_ids = torch.cat([input_ids, next_ids.unsqueeze(1)], dim=1)

            result = input_ids.to("cpu")
            translations.extend(tokenizer.batch_decode(result))
            batch = dev_data.next_batch()
    return translations


def main():
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
    dev_data = MixtureOfBitexts.create_from_config(config, "dev", only_once_thru=True)
    tokenized_dev = TokenizedMixtureOfBitexts(
        dev_data, tokenizer, lang_codes=lang_codes
    )
    model = prepare_model_for_finetuning(ft_params)
    gold_translations = directly_provide_encoding_state(
        model,
        tokenizer,
        tokenized_dev,
        batch_size=ft_params.batch_size,
        target_lang_code=256047,
        noise_variance=0,
    )
    dev_data = MixtureOfBitexts.create_from_config(config, "dev", only_once_thru=True)
    tokenized_dev = TokenizedMixtureOfBitexts(
        dev_data, tokenizer, lang_codes=lang_codes
    )
    corrupted_translations = directly_provide_encoding_state(
        model,
        tokenizer,
        tokenized_dev,
        batch_size=ft_params.batch_size,
        target_lang_code=256047,
        noise_variance=0.02,
    )

    for gold, corrupted in zip(gold_translations, corrupted_translations):
        print(gold)
        print(corrupted)
        print("\n")

    print(evaluate_translations(corrupted_translations, gold_translations))


if __name__ == "__main__":
    main()
