#!/usr/bin/env python
"""Fine-tune DistilBERT for skill-span NER and save the best checkpoint.

Standalone script version of notebooks/01_training.ipynb, wired to the paths
app/models/predictor.py and app/services/role_database.py actually expect
(model/skill_ner_distilbert_best/, model/id2tag.json) instead of the
notebook's own ../Model/ scratch space.

Usage:
    python scripts/train.py
    python scripts/train.py --data data/processed/data_tech_only.csv --epochs 6 --batch-size 32
    python scripts/train.py --out-dir /tmp/model_run_1

Requires internet access the first time, to download distilbert-base-uncased
from huggingface.co (or pre-fetch with `huggingface-cli download distilbert-base-uncased`).
A GPU is strongly recommended but not required.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from common import (
    DEFAULT_DATA_PATH,
    ID2LABEL,
    LABEL2ID,
    MODEL_OUT_DIR,
    NUM_TAGS,
    O_IDX,
    PAD_TAG,
    SkillNERDataset,
    TAG_LIST,
    add_bio_tags,
    entity_prf,
    load_postings,
    split_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")

MODEL_NAME = "distilbert-base-uncased"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to the processed postings CSV")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--patience", type=int, default=2,
                    help="Early-stopping patience, in epochs without validation-F1 improvement")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42,
                    help="Also controls the train/val/test split — keep this consistent with "
                         "scripts/build_role_database.py and scripts/evaluate.py")
    p.add_argument("--out-dir", type=Path, default=MODEL_OUT_DIR,
                    help="Where to save skill_ner_distilbert_best/ and id2tag.json")
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_epoch(model, loader, device, optimizer=None):
    """One pass over `loader`. Trains (and steps `optimizer`) if given,
    otherwise just evaluates. Returns (avg_loss, precision, recall, f1).
    """
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, n_examples = 0.0, 0
    pred_seqs, true_seqs = [], []

    with torch.set_grad_enabled(train):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            n_examples += input_ids.size(0)

            preds = outputs.logits.argmax(-1)
            for p_row, y_row in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                idxs = [j for j, t in enumerate(y_row) if t != PAD_TAG]
                pred_seqs.append([p_row[j] for j in idxs])
                true_seqs.append([y_row[j] for j in idxs])

    precision, recall, f1 = entity_prf(pred_seqs, true_seqs)
    return total_loss / max(n_examples, 1), precision, recall, f1


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    if not args.data.exists():
        sys.exit(
            f"Data file not found: {args.data}\n"
            "Point --data at the processed postings CSV (data_tech_only.csv)."
        )

    logger.info("Loading postings from %s", args.data)
    df = load_postings(args.data)
    df = add_bio_tags(df)
    covered = sum(any(t != O_IDX for t in tags) for tags in df["bio_tags"])
    logger.info("Postings with >=1 tagged skill span: %d/%d", covered, len(df))

    idx_train, idx_val, idx_test = split_dataset(df, random_state=args.seed)
    logger.info("train: %d  val: %d  test: %d", len(idx_train), len(idx_val), len(idx_test))

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception:
        logger.error(
            "Could not download the pretrained tokenizer/model from huggingface.co. "
            "Run this somewhere with internet access, or pre-download once with "
            "`huggingface-cli download %s`.", MODEL_NAME,
        )
        raise

    train_loader = DataLoader(SkillNERDataset(df, idx_train, tokenizer), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SkillNERDataset(df, idx_val, tokenizer), batch_size=args.batch_size)
    test_loader = DataLoader(SkillNERDataset(df, idx_test, tokenizer), batch_size=args.batch_size)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_TAGS, id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(device)
    logger.info("Trainable params: %s", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_dir = args.out_dir / "skill_ner_distilbert_best"
    best_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    patience_counter = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, *_ = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_p, val_r, val_f1 = run_epoch(model, val_loader, device)

        logger.info(
            "Epoch %02d | train_loss %.4f | val_loss %.4f val_precision %.3f val_recall %.3f val_f1 %.3f",
            epoch, train_loss, val_loss, val_p, val_r, val_f1,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            logger.info("  -> new best (val_f1=%.3f), saved to %s", val_f1, best_dir)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping — validation F1 stopped improving.")
                break

    if best_val_f1 < 0:
        sys.exit("Training produced no checkpoint (0 epochs completed) — nothing to save.")

    logger.info("Training finished in %.1fs. Reloading best checkpoint for the held-out test eval.",
                time.time() - start)
    model = AutoModelForTokenClassification.from_pretrained(best_dir).to(device)
    test_loss, test_p, test_r, test_f1 = run_epoch(model, test_loader, device)
    logger.info("TEST — precision %.3f | recall %.3f | f1 %.3f", test_p, test_r, test_f1)

    id2tag = {str(i): t for i, t in enumerate(TAG_LIST)}
    id2tag_path = args.out_dir / "id2tag.json"
    id2tag_path.write_text(json.dumps(id2tag, indent=2))
    logger.info("Saved tag list to %s", id2tag_path)

    logger.info(
        "Done. Model ready at %s — run scripts/build_role_database.py next to generate "
        "model/role_database.json, then scripts/evaluate.py to sanity-check the full pipeline.",
        best_dir,
    )


if __name__ == "__main__":
    main()
