import argparse
from configure import create_bitexts_from_experiment_dir
from configure import harvest_language_codes
from configure import initialize_tokenizer
from configure import USE_CUDA
from configure import create_bitexts
from corpora import MixtureOfBitexts
import evaluate
import hashlib
import json
from myutil import logger
from pathlib import Path
from precomputed import collate_pairs, MANIFEST
import time
import torch
from tokenization import ByteTokenizer, strip_generated_ids
from transformers import AutoModelForSeq2SeqLM


def translate(
    src_tokenized,
    tgt_tokenizer,
    model,
    tgt_lang,
    permutation=None,
    a=32,
    b=3,
    num_beams=4,
    **kwargs,
):
    model.eval()
    src_tokenized = {k: v.to(model.device) for k, v in src_tokenized.items()}
    result = model.generate(
        **src_tokenized,
        forced_bos_token_id=3,  # tgt_tokenizer.get_special_tokens()[tgt_lang],
        max_new_tokens=int(a + b * src_tokenized["input_ids"].shape[1]),
        num_beams=num_beams,
        **kwargs,
    )
    result = result.to("cpu")
    if permutation is not None:
        result.apply_(permutation.get_inverse())
    return tgt_tokenizer.batch_decode(result)


def translate_tokenized_mixture_of_bitexts(mix, model, tokenizer_map, cipher_map):
    if USE_CUDA:
        model.cuda()
    translations = dict()
    for batch in mix:
        src, _, metadata = batch
        cipher = (
            cipher_map[metadata["lang2_tokenizer"], metadata["lang2_encipherment"]]
            if metadata["lang2_encipherment"] != "0"
            else None
        )
        src_code = metadata["lang1_code"]
        tgt_code = metadata["lang2_code"]
        key = "->".join([src_code, tgt_code])
        if key not in translations:
            translations[key] = []
        translated = translate(
            src,
            tokenizer_map[metadata["lang2_tokenizer"]],
            model,
            tgt_code,
            cipher,
        )
        translations[key].extend(translated)
        logger(f"translation: {translated[0]}")
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


def load_target_decoder(pc):
    """The tokenizer that turns generated target ids back into text."""
    if "tgt_autocomplete_model" in pc:
        from compressor import load_autocompleting_tokenizer

        return load_autocompleting_tokenizer(
            pc["tgt_autocomplete_model"],
            None,
            pc["tgt_cap"],
            keep_first=True,
            keep_eos=True,
        )
    return ByteTokenizer()  # uncompressed byte target


def diagnose_hypotheses(decoder, rows, texts):
    """Are the generated sequences ones the compressor could have produced?

    A sequence is valid if compressing its decompression gives it back. If the
    canonical form is shorter, the model wrote out bytes the autocomplete model would
    have removed (harmless: the text is still what it wrote). Anything else - a marker
    promising bytes the model would not predict, a missing EOS, a wrong count - means
    the filled-in text may not be what the model intended. It is a heuristic split.
    """
    kept = [strip_generated_ids(row) for row in rows]
    again = [c[1:] for c in decoder.compress_batch(texts)]
    counts = {"valid": 0, "canonical_form_shorter": 0, "other": 0}
    for original, canonical in zip(kept, again):
        if canonical == original:
            counts["valid"] += 1
        elif len(canonical) < len(original):
            counts["canonical_form_shorter"] += 1
        else:
            counts["other"] += 1
    return {**counts, "total": len(kept)}


def length_sorted_batches(lengths, max_pairs, max_tokens):
    """Groups indices into batches of similar length, longest first.

    A batch is padded to its longest member, so it holds at most `max_pairs` items and
    at most `max_tokens` padded tokens (a lone item longer than that gets its own
    batch). Sorting keeps padding small and stops one long sentence from holding up
    a batch of short ones; longest first means an out-of-memory error shows up at once.
    """
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches, current = [], []
    for i in order:
        full = len(current) == max_pairs
        too_wide = current and (len(current) + 1) * lengths[current[0]] > max_tokens
        if current and (full or too_wide):
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


def oracle_cache_path(pc):
    """Where the oracle score for this test set and target tokenizer is kept, or None.

    The oracle (gold targets pushed through the decoder) does not depend on the model
    being evaluated, so it is computed once. The key covers everything it does depend
    on, including the source of the decoding code, so any change to that invalidates it.
    """
    if "tgt_autocomplete_model" not in pc:
        return None  # byte targets decode instantly; nothing worth keeping
    model_file = Path(pc["tgt_autocomplete_model"]) / "best_model.pt"
    reference = Path(pc["reference_test_file"])
    here = Path(__file__).parent

    def stamp(path):
        return [path.stat().st_size, path.stat().st_mtime_ns]

    parts = {
        "manifest": hashlib.sha256(
            (Path(pc["test"]) / MANIFEST).read_bytes()
        ).hexdigest(),
        "code": [
            hashlib.sha256((here / name).read_bytes()).hexdigest()
            for name in ("compressor.py", "tokenization.py")
        ],
        "tgt_cap": pc["tgt_cap"],
        "tgt_max_length": pc.get("tgt_max_length"),
        "model": [str(model_file.resolve()), *stamp(model_file)],
        "reference": [str(reference.resolve()), *stamp(reference)],
    }
    key = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]
    return Path(pc["test"]) / f"oracle-{key}.json"


def load_oracle_scores(path, key):
    try:
        return json.loads(path.read_text())[key]
    except (OSError, ValueError, KeyError, AttributeError):
        return None


def evaluate_precomputed(
    experiment_dir, config, num_beams=4, batch_size=64, max_src_tokens=16000
):
    """Evaluation for configs with a "precomputed" section.

    Scores translations against the raw reference text (not a round trip through the
    target tokenizer) and also scores the gold target decoded through the same path,
    which is the ceiling that compression + decompression alone puts on the score.
    The ceiling is cached (see oracle_cache_path). Sentences are translated in batches
    of similar source length, then put back in file order.
    """
    experiment_dir = Path(experiment_dir)
    pc, ft = config["precomputed"], config["finetuning_parameters"]
    bitexts = create_bitexts(config)
    pairs = bitexts["pairs"]["test"]
    model = AutoModelForSeq2SeqLM.from_pretrained(experiment_dir)
    if USE_CUDA:
        model.cuda()
    model.eval()
    decoder = load_target_decoder(pc)
    lang_id = ft.get("forced_bos_token_id", 1)
    tgt_limit = pc.get("tgt_max_length") or 1024
    key = pc.get("lang_pair", "src->tgt")

    with open(pc["reference_test_file"], encoding="utf-8") as reader:
        ref_lines = [line.rstrip("\n") for line in reader]
    references = [ref_lines[pairs.line_id(i)] for i in range(len(pairs))]
    with open(experiment_dir / "references.json", "w") as writer:
        json.dump({key: references}, writer)

    logger("Translating test data")
    started = time.perf_counter()
    data = [pairs[i] for i in range(len(pairs))]
    hyp_rows = [None] * len(data)
    batches = length_sorted_batches(
        [len(src) for src, _ in data], batch_size, max_src_tokens
    )
    with torch.no_grad():
        for batch in batches:
            src, _ = collate_pairs([data[i] for i in batch])
            src = {k: v.to(model.device) for k, v in src.items()}
            out = model.generate(
                **src,
                forced_bos_token_id=lang_id,
                max_new_tokens=min(tgt_limit, int(32 + 4 * src["input_ids"].shape[1])),
                num_beams=num_beams,
            )
            for i, row in zip(batch, out.cpu()):
                hyp_rows[i] = row
    took = time.perf_counter() - started
    logger(f"...generation took {took:.0f}s ({len(batches)} batches)")

    def decode(rows):
        texts = []
        for i in range(0, len(rows), 64):
            texts.extend(decoder.batch_decode(rows[i : i + 64]))
        return texts

    started = time.perf_counter()
    hyps = decode(hyp_rows)
    logger(f"...decoding took {time.perf_counter() - started:.0f}s")
    with open(experiment_dir / "translations.json", "w") as writer:
        json.dump({key: hyps}, writer)
    logger("...translation complete. Scoring.")
    with open(experiment_dir / "scores.json", "w") as writer:
        json.dump({key: evaluate_translations(hyps, references)}, writer)

    cache = oracle_cache_path(pc)
    oracle_scores = load_oracle_scores(cache, key) if cache else None
    if oracle_scores is None:
        started = time.perf_counter()
        oracle = decode([tgt for _, tgt in data])
        oracle_scores = evaluate_translations(oracle, references)
        logger(f"...oracle took {time.perf_counter() - started:.0f}s")
        if cache:
            try:
                cache.write_text(json.dumps({key: oracle_scores}))
            except OSError:
                logger(f"could not write oracle cache {cache}")
    else:
        logger(f"...oracle from cache {cache.name}")
    with open(experiment_dir / "scores_oracle.json", "w") as writer:
        json.dump({key: oracle_scores}, writer)
    if hasattr(decoder, "compress_batch"):
        with open(experiment_dir / "diagnostics.json", "w") as writer:
            json.dump(diagnose_hypotheses(decoder, hyp_rows, hyps), writer, indent=2)
    logger("...scoring complete.")


def evaluate_experiment(experiment_dir, **kwargs):
    with open(Path(experiment_dir) / "experiment.json") as reader:
        if "precomputed" in (config := json.load(reader)):
            return evaluate_precomputed(experiment_dir, config, **kwargs)
    logger(f"Initializing model from: {experiment_dir}")
    bitexts = create_bitexts_from_experiment_dir(experiment_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(experiment_dir)
    if USE_CUDA:
        model.cuda()

    logger(f"Collating reference translations")
    references = dict()
    test_data = bitexts["test"]
    test_data.restart()
    tokenizer_map = bitexts["tokenizer_map"]
    cipher_map = bitexts["cipher_map"]
    for _, tgt, metadata in test_data:
        src_code = metadata["lang1_code"]
        tgt_code = metadata["lang2_code"]
        tgt_tokenizer = tokenizer_map[metadata["lang2_tokenizer"]]
        key = "->".join([src_code, tgt_code])
        if key not in references:
            references[key] = []
        tgt_ids = tgt["input_ids"]
        tgt_ids[tgt_ids == -100] = 2  # TODO: make more general
        cipher = (
            cipher_map[metadata["lang2_tokenizer"], metadata["lang2_encipherment"]]
            if metadata["lang2_encipherment"] != "0"
            else None
        )
        if cipher is not None:
            tgt_ids.apply_(cipher.get_inverse())
        tgt = tgt_tokenizer.batch_decode(tgt_ids)
        references[key].extend(tgt)
    with open(Path(experiment_dir) / "references.json", "w") as writer:
        json.dump(references, writer)
    logger("...references complete.")

    logger(f"Translating test data")
    test_data.restart()
    translations = translate_tokenized_mixture_of_bitexts(
        test_data, model, tokenizer_map, bitexts["cipher_map"]
    )
    with open(Path(experiment_dir) / "translations.json", "w") as writer:
        json.dump(translations, writer)
    logger("...translation complete.")

    logger(f"Scoring translations")
    scores = dict()
    for key in translations:
        scores[key] = evaluate_translations(translations[key], references[key])
    with open(Path(experiment_dir) / "scores.json", "w") as writer:
        json.dump(scores, writer)
    logger("...scoring complete.")


# TODO: update
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Sentences per batch (precomputed configs).",
    )
    parser.add_argument(
        "--max_src_tokens",
        type=int,
        default=16000,
        help="Padded source tokens per batch (precomputed configs).",
    )
    args = parser.parse_args()
    evaluate_experiment(
        args.dir, batch_size=args.batch_size, max_src_tokens=args.max_src_tokens
    )
    # evaluate_model(
    #    "facebook/nllb-200-distilled-600M", "examples/nllb_seed_config_small.json"
    # )
