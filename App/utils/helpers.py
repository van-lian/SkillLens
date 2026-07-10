import re
from typing import List, Tuple


def tokenize(text: str) -> List[str]:
    """Word-level tokenizer. Must match the one used to build BIO tags in the
    training notebook (06_Data_NLP.ipynb, Stage 1), or predictions will be
    misaligned with the model's training distribution.
    """
    return re.findall(r"[a-z0-9+#.]+", str(text).lower())


def tags_to_spans(tag_ids: List[int], tag_list: List[str]) -> List[Tuple[int, int, str]]:
    """Convert a sequence of BIO tag ids into (start, end, category) spans.

    Mirrors `tags_to_spans` from the training notebook.
    """
    spans = []
    start, cat = None, None
    o_idx = tag_list.index("O")

    for i, t in enumerate(list(tag_ids) + [o_idx]):
        label = tag_list[t]
        if label.startswith("B-"):
            if start is not None:
                spans.append((start, i, cat))
            start, cat = i, label[2:]
        elif label.startswith("I-") and cat == label[2:]:
            continue
        else:
            if start is not None:
                spans.append((start, i, cat))
            start, cat = None, None

    return spans
