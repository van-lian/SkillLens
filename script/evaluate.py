#!/usr/bin/env python
"""Evaluate a trained skill-NER checkpoint.

Two things, either of which can be skipped:

1. Entity-level precision / recall / F1 on the held-out test split (the same
   split scripts/train.py used — reproduced here via --seed, same as
   01_training.ipynb Stage 8).
2. A pipeline smoke test that imports the *actual* app code
   (app.models.extractor.extract_skills, app.models.comparator.compare_to_role,
   app.models.comparator.recommend) and exercises the known-role /
   unknown-role / empty-text edge cases from 03_testing.ipynb. A passing run
   here is a real signal that `uvicorn app.main:app` will work, since it
   loads through predictor.py and role_database.py exactly as the API does.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model-dir model/skill_ner_distilbert_best --split all
    python scripts/evaluate.py --skip-pipeline-check
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from common import (
    BEST_MODEL_DIR,
    DEFAULT_DATA_PATH,
    PAD_TAG,
    ROLE_DB_PATH,
    SkillNERDataset,
    add_bio_tags,
    entity_prf,
    load_postings,
    split_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate")

SAMPLE_CV = (
    "Experienced backend developer with 4 years building REST APIs in Python "
    "and Node.js. Strong SQL and PostgreSQL skills, some exposure to Docker "
    "and Git-based workflows."
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    p.add_argument("--model-dir", type=Path, default=BEST_MODEL_DIR)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42, help="Must match scripts/train.py's --seed")
    p.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    p.add_argument("--skip-metrics", action="store_true", help="Skip the precision/recall/F1 pass")
    p.add_argument("--skip-pipeline-check", action="store_true",
                    help="Skip importing app.* and exercising extract_skills / compare_to_role / recommend")
    return p.parse_args()


@torch.no_grad()
def evaluate_split(model, loader, device):
    model.eval()
    pred_seqs, true_seqs = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = logits.argmax(-1).cpu().tolist()

        for p_row, y_row in zip(preds, labels.tolist()):
            idxs = [j for j, t in enumerate(y_row) if t != PAD_TAG]
            pred_seqs.append([p_row[j] for j in idxs])
            true_seqs.append([y_row[j] for j in idxs])

    return entity_prf(pred_seqs, true_seqs)


def run_pipeline_check():
    """Exercises the actual app code, not a reimplementation of it. Note this
    always loads through predictor.py's fixed model dir (model/skill_ner_distilbert_best),
    same as the API — independent of any --model-dir override above.
    """
    logger.info("Running pipeline smoke test against app/ modules (app's fixed model path)...")
    from app.models.predictor import predictor
    from app.services.role_database import role_database

    predictor.load()
    role_database.load()

    from app.models.comparator import compare_to_role, recommend
    from app.models.extractor import extract_skills

    assert extract_skills("") == [], "extract_skills('') should return []"
    logger.info("Empty-text case OK: extract_skills('') -> []")

    skills = extract_skills(SAMPLE_CV)
    logger.info("extract_skills(sample CV) -> %s", skills)
    assert isinstance(skills, list)

    gap = compare_to_role(SAMPLE_CV, "a role that almost certainly does not exist", top_n=10)
    assert gap["found"] is False and gap["match_score"] == 0, "unknown role should be found=False, score=0"
    logger.info("Unknown-role case OK: %s", gap)

    roles = role_database.list_roles()
    if roles:
        real_role = roles[0]
        gap = compare_to_role(SAMPLE_CV, real_role, top_n=10)
        recs = recommend(gap)
        logger.info("compare_to_role(%r) -> match_score=%s, have=%s, missing=%s",
                    real_role, gap["match_score"], gap["have"], gap["missing"])
        logger.info("recommend() -> %s", recs)
        assert gap["found"] is True
    else:
        logger.warning("Role database is empty — skipping the known-role check.")

    logger.info("Pipeline smoke test passed.")


def main():
    args = parse_args()

    if args.skip_metrics and args.skip_pipeline_check:
        sys.exit("Nothing to do — both --skip-metrics and --skip-pipeline-check were passed.")

    if not args.skip_metrics:
        if not args.model_dir.exists():
            sys.exit(f"Model checkpoint not found: {args.model_dir}\nRun scripts/train.py first.")
        if not args.data.exists():
            sys.exit(f"Data file not found: {args.data}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: %s", device)

        logger.info("Loading model from %s", args.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(device)

        logger.info("Loading postings from %s", args.data)
        df = load_postings(args.data)
        df = add_bio_tags(df)
        idx_train, idx_val, idx_test = split_dataset(df, random_state=args.seed)

        splits = {"train": idx_train, "val": idx_val, "test": idx_test}
        targets = splits if args.split == "all" else {args.split: splits[args.split]}

        for name, idx in targets.items():
            loader = DataLoader(SkillNERDataset(df, idx, tokenizer), batch_size=args.batch_size)
            precision, recall, f1 = evaluate_split(model, loader, device)
            logger.info("%-5s (n=%4d) — precision %.3f | recall %.3f | f1 %.3f",
                        name, len(idx), precision, recall, f1)

    if not args.skip_pipeline_check:
        if not ROLE_DB_PATH.exists():
            logger.warning(
                "%s not found — run scripts/build_role_database.py first for the pipeline "
                "check to exercise a real role. Skipping pipeline check.", ROLE_DB_PATH,
            )
        else:
            run_pipeline_check()


if __name__ == "__main__":
    main()
