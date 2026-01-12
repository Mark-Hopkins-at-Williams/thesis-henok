import argparse
from configure import harvest_language_codes
from configure import initialize_tokenizer
from configure import USE_CUDA
from corpora import MixtureOfBitexts, TokenizedMixtureOfBitexts
import evaluate
import json
from myutil import logger
from pathlib import Path
from permutations import load_permutation_map
from transformers import AutoModelForSeq2SeqLM


def translate(
    src_tokenized,
    tokenizer,
    model,
    tgt_lang,
    permutation=None,
    a=32,
    b=3,
    num_beams=4,
    **kwargs,
):
    model.eval()
    result = model.generate(
        **src_tokenized.to(model.device),
        forced_bos_token_id=tokenizer.get_special_tokens()[tgt_lang],
        max_new_tokens=int(a + b * src_tokenized.input_ids.shape[1]),
        num_beams=num_beams,
        **kwargs,
    )
    result = result.to("cpu")
    if permutation is not None:
        result.apply_(permutation.get_inverse())
    return tokenizer.batch_decode(result)


def translate_tokenized_mixture_of_bitexts(mix, model, tokenizer, lang_codes, pmap):
    if USE_CUDA:
        model.cuda()
    batch = mix.next_batch()
    translations = dict()
    while batch is not None:
        src, tgt, src_lang, tgt_lang = batch
        permutation = pmap[tgt_lang] if tgt_lang in pmap else None
        src_code = lang_codes[src_lang]
        tgt_code = lang_codes[tgt_lang]
        key = "->".join([src_code, tgt_code])
        if key not in translations:
            translations[key] = []
        translated = translate(src, tokenizer, model, tgt_code, permutation)
        translations[key].extend(translated)
        logger(f"translation: {translated[0]}")
        batch = mix.next_batch()
    return translations


def evaluate_translations(candidate_translations, reference_translations):
    bleu_calc = evaluate.load("sacrebleu")
    chrf_calc = evaluate.load("chrf")
    reference_translations = [[ref] for ref in reference_translations]
    bleu_result = bleu_calc.compute(
        predictions=candidate_translations, references=reference_translations
    )
    chrf_result = chrf_calc.compute(
        predictions=candidate_translations, references=reference_translations
    )
    return {
        "bleu": round(bleu_result["score"], 3),
        "chrf": round(chrf_result["score"], 3),
    }


def evaluate_experiment(experiment_dir):
    logger(f"Initializing model from: {experiment_dir}")
    config_file = Path(experiment_dir) / "experiment.json"
    with open(config_file) as reader:
        config = json.load(reader)
    model = AutoModelForSeq2SeqLM.from_pretrained(experiment_dir)
    if USE_CUDA:
        model.cuda()
    lang_codes = harvest_language_codes(config)
    tokenizer = initialize_tokenizer(config)
    pmap = load_permutation_map(Path(experiment_dir) / "permutations.json")
    test_data = MixtureOfBitexts.create_from_config(config, "test", only_once_thru=True)
    tokenized_test = TokenizedMixtureOfBitexts(
        test_data, tokenizer, lang_codes=lang_codes, permutation_map=pmap
    )
    logger(f"Translating test data")
    translations = translate_tokenized_mixture_of_bitexts(
        tokenized_test, model, tokenizer, lang_codes, pmap
    )
    with open(Path(experiment_dir) / "translations.json", "w") as writer:
        json.dump(translations, writer)
    logger("...translation complete.")
    logger(f"Collating reference translations")
    test_data = MixtureOfBitexts.create_from_config(config, "test", only_once_thru=True)
    references = dict()
    batch = test_data.next_batch()
    while batch is not None:
        _, tgt, src_lang, tgt_lang = batch
        src_code = lang_codes[src_lang]
        tgt_code = lang_codes[tgt_lang]
        key = "->".join([src_code, tgt_code])
        if key not in references:
            references[key] = []
        references[key].extend(tgt)
        batch = test_data.next_batch()
    with open(Path(experiment_dir) / "references.json", "w") as writer:
        json.dump(references, writer)
    logger("...references complete.")
    logger(f"Scoring translations")
    scores = dict()
    for key in translations:
        scores[key] = evaluate_translations(translations[key], references[key])
    with open(Path(experiment_dir) / "scores.json", "w") as writer:
        json.dump(scores, writer)
    logger("...scoring complete.")


def evaluate_model(model_name, config_file):
    with open(config_file) as reader:
        config = json.load(reader)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if USE_CUDA:
        model.cuda()
    lang_codes = harvest_language_codes(config)
    tokenizer = initialize_tokenizer(config)
    pmap = dict()
    test_data = MixtureOfBitexts.create_from_config(config, "test", only_once_thru=True)
    tokenized_test = TokenizedMixtureOfBitexts(
        test_data, tokenizer, lang_codes=lang_codes, permutation_map=pmap
    )
    logger(f"Translating test data")
    translations = translate_tokenized_mixture_of_bitexts(
        tokenized_test, model, tokenizer, lang_codes, pmap
    )
    with open("translations.json", "w") as writer:
        json.dump(translations, writer)
    logger("...translation complete.")
    logger(f"Collating reference translations")
    test_data = MixtureOfBitexts.create_from_config(config, "test", only_once_thru=True)
    references = dict()
    batch = test_data.next_batch()
    while batch is not None:
        _, tgt, src_lang, tgt_lang = batch
        src_code = lang_codes[src_lang]
        tgt_code = lang_codes[tgt_lang]
        key = "->".join([src_code, tgt_code])
        if key not in references:
            references[key] = []
        references[key].extend(tgt)
        batch = test_data.next_batch()
    with open("references.json", "w") as writer:
        json.dump(references, writer)
    logger("...references complete.")
    logger(f"Scoring translations")
    scores = dict()
    for key in translations:
        scores[key] = evaluate_translations(translations[key], references[key])
    with open("scores.json", "w") as writer:
        json.dump(scores, writer)
    logger("...scoring complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate finetuning experiment.")
    parser.add_argument("--dir", type=str, required=True, help="Experiment directory.")
    args = parser.parse_args()
    # evaluate_experiment(args.dir)
    evaluate_model(
        "facebook/nllb-200-distilled-600M", "examples/nllb_seed_config_small.json"
    )
