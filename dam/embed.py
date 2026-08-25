"""Sentence embeddings used by the description-based retrieval rules.

The embedder is a frozen pretrained encoder and is deliberately independent of
the agent: semantic similarity has to be measured by something that knows
nothing about the task's reward structure, otherwise the similarity baseline
would be smuggling in decision information.
"""

from __future__ import annotations

import numpy as np

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def _load():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: list[str], batch_size: int = 256) -> dict[str, np.ndarray]:
    """L2-normalised embeddings, keyed by text. Deduplicates first: the
    generator emits a few thousand distinct surface forms at most."""
    uniq = sorted(set(texts))
    model = _load()
    vecs = model.encode(uniq, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)
    return {t: v.astype(np.float32) for t, v in zip(uniq, vecs)}
