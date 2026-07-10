import json
import logging
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parents[2] / "model" / "skill_ner_distilbert_best"
MAX_LEN = 100


class SkillPredictor:
    """Wraps the fine-tuned DistilBERT skill-NER model from the training
    notebook. Loaded once at app startup (see main.py's lifespan handler) and
    reused for every request.
    """

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.tag_list: List[str] = []

    def load(self) -> None:
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}\n"
                "Train the model with notebooks/01_training.ipynb (or "
                "scripts/train.py) and copy the saved checkpoint into "
                "model/skill_ner_distilbert_best/ before starting the API."
            )

        logger.info("Loading skill-NER model from %s on %s", self.model_dir, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir).to(self.device)
        self.model.eval()
        self.tag_list = self._load_tag_list()
        logger.info("Model loaded. Tags: %s", self.tag_list)

    def _load_tag_list(self) -> List[str]:
        # Prefer a standalone id2tag.json (per the project layout) if present,
        # otherwise fall back to the id2label mapping baked into the model
        # config at save time — both are produced by the training notebook.
        id2tag_path = self.model_dir.parent / "id2tag.json"
        if id2tag_path.exists():
            id2tag = json.loads(id2tag_path.read_text())
            return [id2tag[str(i)] for i in range(len(id2tag))]

        id2label = self.model.config.id2label
        return [id2label[i] for i in range(len(id2label))]

    def is_ready(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def predict_word_tags(self, words: List[str]) -> List[int]:
        """Given pre-tokenized words, return one predicted tag id per word."""
        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        word_ids = enc.word_ids(batch_index=0)

        logits = self.model(
            input_ids=enc["input_ids"].to(self.device),
            attention_mask=enc["attention_mask"].to(self.device),
        ).logits
        preds = logits.argmax(-1)[0].cpu().tolist()

        o_idx = self.tag_list.index("O")
        word_tags = [o_idx] * len(words)
        seen = set()
        for pos, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            word_tags[wid] = preds[pos]
        return word_tags


# Single shared instance. Loaded once via predictor.load() in main.py's
# lifespan handler, then imported wherever inference is needed.
predictor = SkillPredictor()
