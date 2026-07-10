"""Shared data-loading, BIO-tagging, and dataset utilities for the training
and evaluation scripts.

Mirrors notebooks/01_training.ipynb (Stages 1-2) and 02_experiments.ipynb
(Stage 2 reload) exactly, so scripts/train.py, scripts/evaluate.py, and
scripts/build_role_database.py stay in lockstep with the notebooks' logic
and with the model app/models/predictor.py actually loads.

Deliberate difference from the notebooks: paths here point at the project
root's model/ and data/ directories (matching what app/models/predictor.py
and app/services/role_database.py expect), not the notebooks' own ../Model/
scratch space. See the "Known open item" in the project notes — these
scripts are the fix for that mismatch; the notebooks were intentionally
left as-is.
"""
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.helpers import tags_to_spans, tokenize  # noqa: E402

# --- Paths -------------------------------------------------------------
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data_tech_only.csv"
MODEL_OUT_DIR = PROJECT_ROOT / "model"
BEST_MODEL_DIR = MODEL_OUT_DIR / "skill_ner_distilbert_best"
ID2TAG_PATH = MODEL_OUT_DIR / "id2tag.json"
ROLE_DB_PATH = MODEL_OUT_DIR / "role_database.json"

# --- Tag set (must match notebooks/01_training.ipynb Stage 2 exactly) --
CATEGORY_COLS = {
    "TECH": "skills_technical",
    "SOFT": "skills_soft",
    "TOOL": "skills_tool",
    "DOMAIN": "skills_domain",
    "CERT": "skills_certification",
}

TAG_LIST = ["O"]
for _cat in CATEGORY_COLS:
    TAG_LIST += [f"B-{_cat}", f"I-{_cat}"]
TAG_TO_IDX = {t: i for i, t in enumerate(TAG_LIST)}
ID2LABEL = {i: t for i, t in enumerate(TAG_LIST)}
LABEL2ID = {t: i for i, t in enumerate(TAG_LIST)}
O_IDX = TAG_TO_IDX["O"]
NUM_TAGS = len(TAG_LIST)

MAX_LEN = 100
PAD_TAG = -100  # ignored by the loss


# --- Data loading --------------------------------------------------------
def parse_skills(cell):
    if pd.isna(cell):
        return []
    return [s.strip() for s in str(cell).split(";") if s.strip()]


def load_postings(csv_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load & filter the processed postings CSV, same as Stage 1 of
    01_training.ipynb, and parse the per-category skill columns into lists.
    """
    df = pd.read_csv(csv_path)
    df = df[df["skill_count"] > 0].dropna(subset=["cleaned_text"]).reset_index(drop=True)
    for col in CATEGORY_COLS.values():
        df[col + "_list"] = df[col].apply(parse_skills)
    return df


# --- BIO tagging (distant supervision) — Stage 2 of 01_training.ipynb ---
def find_span(tokens, phrase_tokens, taken):
    n, m = len(tokens), len(phrase_tokens)
    if m == 0:
        return None
    for i in range(n - m + 1):
        if any(taken[i:i + m]):
            continue
        if tokens[i:i + m] == phrase_tokens:
            return i
    return None


def build_bio_tags(text, skills_by_cat):
    tokens = tokenize(text)
    tags = [O_IDX] * len(tokens)
    taken = [False] * len(tokens)

    ordered = [(cat, s) for cat, skills in skills_by_cat.items() for s in skills]
    ordered.sort(key=lambda x: -len(tokenize(x[1])))  # longest phrase first

    for cat, skill in ordered:
        phrase_tokens = tokenize(skill)
        start = find_span(tokens, phrase_tokens, taken)
        if start is None:
            continue
        end = start + len(phrase_tokens)
        tags[start] = TAG_TO_IDX[f"B-{cat}"]
        for i in range(start + 1, end):
            tags[i] = TAG_TO_IDX[f"I-{cat}"]
        for i in range(start, end):
            taken[i] = True
    return tokens, tags


def add_bio_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 'tokens' and 'bio_tags' columns to df in place, and returns it."""
    all_tokens, all_tags = [], []
    for _, row in df.iterrows():
        skills_by_cat = {cat: row[col + "_list"] for cat, col in CATEGORY_COLS.items()}
        toks, tags = build_bio_tags(row["cleaned_text"], skills_by_cat)
        all_tokens.append(toks)
        all_tags.append(tags)
    df["tokens"] = all_tokens
    df["bio_tags"] = all_tags
    return df


# --- Train/val/test split — Stage 3 of 01_training.ipynb ---------------
def split_dataset(df: pd.DataFrame, test_size: float = 0.3, val_fraction_of_temp: float = 0.5,
                   random_state: int = 42):
    """Same two-step split as the notebooks: 70/15/15 train/val/test by
    default. IMPORTANT: random_state must match across train.py,
    build_role_database.py, and evaluate.py for the splits to line up
    (build_role_database.py needs to know which postings were NOT held out
    as the test set).
    """
    from sklearn.model_selection import train_test_split
    idx_train, idx_temp = train_test_split(df.index, test_size=test_size, random_state=random_state)
    idx_val, idx_test = train_test_split(idx_temp, test_size=val_fraction_of_temp, random_state=random_state)
    return idx_train, idx_val, idx_test


# --- Dataset / label alignment — Stage 4 of 01_training.ipynb ----------
def align_labels_with_tokens(word_ids, word_tags):
    labels = []
    prev_word_id = None
    for wid in word_ids:
        if wid is None:
            labels.append(PAD_TAG)
        elif wid != prev_word_id:
            labels.append(word_tags[wid])  # first subword of this word gets the real tag
        else:
            labels.append(PAD_TAG)  # continuation subwords are ignored by the loss
        prev_word_id = wid
    return labels


class SkillNERDataset(Dataset):
    def __init__(self, df: pd.DataFrame, indices, tokenizer, max_len: int = MAX_LEN):
        self.df = df
        self.indices = list(indices)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.df.loc[self.indices[i]]
        enc = self.tokenizer(
            row["tokens"], is_split_into_words=True, truncation=True,
            max_length=self.max_len, padding="max_length", return_tensors="pt",
        )
        word_ids = enc.word_ids(batch_index=0)
        labels = align_labels_with_tokens(word_ids, row["bio_tags"])
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# --- Entity-level P/R/F1 — Stage 6 of 01_training.ipynb ----------------
def entity_prf(all_pred_tags, all_true_tags):
    """all_pred_tags / all_true_tags: lists of per-example tag-id sequences,
    already stripped of PAD_TAG positions. Uses app.utils.helpers.tags_to_spans
    (the same span-decoding logic predictor.py uses at inference time).
    """
    tp = fp = fn = 0
    for pred, true in zip(all_pred_tags, all_true_tags):
        pred_spans = set(tags_to_spans(pred, TAG_LIST))
        true_spans = set(tags_to_spans(true, TAG_LIST))
        tp += len(pred_spans & true_spans)
        fp += len(pred_spans - true_spans)
        fn += len(true_spans - pred_spans)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
