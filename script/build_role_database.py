#!/usr/bin/env python
"""Build the role -> top-skills database used by app/services/role_database.py.

Standalone script version of notebooks/02_experiments.ipynb (Stage 3):
reruns the trained skill-NER model over job postings, aggregates the skills
seen per role title, and keeps the top-K most frequent per role.

By default this only aggregates over the train+val postings (the same ones
the model actually learned from / was validated on), matching the notebook.
Pass --include-test-split to also fold in the held-out test set once you're
done using it for evaluation.

Usage:
    python scripts/build_role_database.py
    python scripts/build_role_database.py --model-dir model/skill_ner_distilbert_best --top-k 20
"""
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from common import (
    BEST_MODEL_DIR,
    DEFAULT_DATA_PATH,
    MAX_LEN,
    ROLE_DB_PATH,
    load_postings,
    split_dataset,
)
from app.utils.helpers import tags_to_spans, tokenize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_role_database")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    p.add_argument("--model-dir", type=Path, default=BEST_MODEL_DIR)
    p.add_argument("--out", type=Path, default=ROLE_DB_PATH)
    p.add_argument("--title-col", default="cleaned_title")
    p.add_argument("--top-k", type=int, default=15, help="Top-K most frequent skills kept per role")
    p.add_argument("--min-postings", type=int, default=1, help="Minimum postings required to keep a role")
    p.add_argument("--include-test-split", action="store_true",
                    help="Also use the held-out test split when aggregating skills (off by "
                         "default, to keep the role DB built only from data the model was "
                         "trained/validated on — matches notebooks/02_experiments.ipynb)")
    p.add_argument("--seed", type=int, default=42,
                    help="Must match the --seed used in scripts/train.py for the "
                         "train/val/test split to line up")
    return p.parse_args()


def make_extractor(tokenizer, model, tag_list, device):
    """Returns an extract_skills(text) closure over this checkpoint's
    tokenizer/model/tag_list — same logic as app/models/extractor.py, just
    inlined so this script doesn't depend on predictor.py's singleton
    (which always loads from the app's fixed model dir, not --model-dir).
    """
    o_idx = tag_list.index("O")

    @torch.no_grad()
    def extract_skills(text):
        words = tokenize(text)
        if not words:
            return []
        enc = tokenizer(
            words, is_split_into_words=True, truncation=True,
            max_length=MAX_LEN, padding="max_length", return_tensors="pt",
        )
        word_ids = enc.word_ids(batch_index=0)
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits
        preds = logits.argmax(-1)[0].cpu().tolist()

        word_tags = [o_idx] * len(words)
        seen = set()
        for pos, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            word_tags[wid] = preds[pos]

        spans = tags_to_spans(word_tags, tag_list)
        return [{"skill": " ".join(words[s:e]), "type": cat} for s, e, cat in spans]

    return extract_skills


def build_role_database(df, extract_skills, title_col, top_k, min_postings):
    role_db = {}
    for role, group in df.groupby(title_col):
        if len(group) < min_postings:
            continue
        counter = Counter()
        for text in group["cleaned_text"]:
            for item in extract_skills(text):
                counter[(item["skill"], item["type"])] += 1
        top = counter.most_common(top_k)
        role_db[role] = [{"skill": s, "type": t, "freq": f} for (s, t), f in top]
    return role_db


def main():
    args = parse_args()

    if not args.model_dir.exists():
        raise SystemExit(
            f"Model checkpoint not found: {args.model_dir}\n"
            "Run scripts/train.py first (or point --model-dir at an existing checkpoint)."
        )
    if not args.data.exists():
        raise SystemExit(f"Data file not found: {args.data}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info("Loading model from %s", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    id2label = model.config.id2label
    tag_list = [id2label[i] for i in range(len(id2label))]
    extract_skills = make_extractor(tokenizer, model, tag_list, device)

    logger.info("Loading postings from %s", args.data)
    df = load_postings(args.data)

    idx_train, idx_val, idx_test = split_dataset(df, random_state=args.seed)
    keep_idx = idx_train.union(idx_val)
    if args.include_test_split:
        keep_idx = keep_idx.union(idx_test)
    subset = df.loc[keep_idx]
    logger.info("Aggregating skills over %d postings (%s)", len(subset),
                "train+val+test" if args.include_test_split else "train+val")

    role_db = build_role_database(subset, extract_skills, args.title_col, args.top_k, args.min_postings)

    logger.info("Roles in database: %d", len(role_db))
    if role_db:
        sample_role = max(role_db, key=lambda r: sum(s["freq"] for s in role_db[r]))
        logger.info("Example — %r:", sample_role)
        for s in role_db[sample_role]:
            logger.info("  %-25s (%s)  seen in %d posting(s)", s["skill"], s["type"], s["freq"])
    else:
        logger.warning("No roles made it into the database — check --title-col and --min-postings.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(role_db, indent=2))
    logger.info("Saved role database to %s", args.out)


if __name__ == "__main__":
    main()
