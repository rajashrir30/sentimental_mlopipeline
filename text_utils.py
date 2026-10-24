"""Text normalization shared by training and inference."""
from __future__ import annotations

import re


def clean_text(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text.lower())
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s'!?.,]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
