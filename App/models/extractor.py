from typing import Dict, List

from app.models.predictor import predictor
from app.utils.helpers import tags_to_spans, tokenize


def extract_skills(text: str) -> List[Dict[str, str]]:
    """Run the fine-tuned skill-NER model over free text and return typed
    skill spans, e.g. [{"skill": "docker", "type": "TOOL"}, ...].

    Same logic as `extract_skills` in 06_Data_NLP.ipynb (Stage 3/4), just
    split out so it can back both CV parsing and job-posting parsing.
    """
    if not predictor.is_ready():
        raise RuntimeError("Model is not loaded yet.")

    words = tokenize(text)
    if not words:
        return []

    word_tags = predictor.predict_word_tags(words)
    spans = tags_to_spans(word_tags, predictor.tag_list)

    return [{"skill": " ".join(words[s:e]), "type": cat} for s, e, cat in spans]
