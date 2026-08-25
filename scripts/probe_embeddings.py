"""Diagnostic: where the decision-relevant information sits in the embedding.

The claim behind the whole setup is that the marker token, which alone
determines the correct action, is present in the sentence embedding but is not
what cosine similarity ranks by. This script measures both halves of that
claim:

  1. a linear probe on the frozen embeddings recovers the marker almost
     perfectly, so the information is there and is linearly accessible;
  2. among the nearest neighbours of a query under cosine similarity, the
     cluster is matched far more often than the marker, so the geometry that
     retrieval actually uses is organised by surface, not by decision.

Together they rule out the alternative reading that the similarity baseline
fails simply because the encoder cannot see the marker.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from dam.env import CLUSTER_VOCAB, MARKER_VOCAB, all_situation_texts
from dam.embed import embed_texts

RESULTS = Path("results")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    # A uniform subsample of the 8 x 3^6 x 8 possible surface forms; the full
    # enumeration adds nothing to either measurement and costs minutes.
    rng0 = np.random.default_rng(0)
    everything = all_situation_texts()
    texts = [everything[i] for i in
             rng0.choice(len(everything), size=8000, replace=False)]
    table = embed_texts(texts)
    x = np.stack([table[t] for t in texts])
    marker = np.array([MARKER_VOCAB.index(t.split()[-1]) for t in texts])
    # Cluster is identified by the project token, which is cluster-specific.
    proj_to_cluster = {p: i for i, v in enumerate(CLUSTER_VOCAB) for p in v["project"]}
    cluster = np.array([proj_to_cluster[t.split()[1]] for t in texts])

    report: dict[str, float] = {}
    for name, y in (("marker", marker), ("cluster", cluster)):
        xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3,
                                              random_state=0, stratify=y)
        clf = LogisticRegression(max_iter=2000, C=10.0).fit(xtr, ytr)
        report[f"probe_acc_{name}"] = float(clf.score(xte, yte))

    # Neighbourhood composition under cosine similarity.
    rng = np.random.default_rng(0)
    idx = rng.choice(len(texts), size=1000, replace=False)
    sims = x[idx] @ x.T
    sims[np.arange(len(idx)), idx] = -np.inf
    for k in (1, 4, 16):
        nn = np.argsort(-sims, axis=1)[:, :k]
        # Mean fraction of the k neighbours sharing the attribute, so the
        # chance level is 1/8 independently of k.
        report[f"nn{k}_same_marker"] = float(
            (marker[nn] == marker[idx][:, None]).mean())
        report[f"nn{k}_same_cluster"] = float(
            (cluster[nn] == cluster[idx][:, None]).mean())
    report["chance_same_marker"] = 1.0 / len(MARKER_VOCAB)
    report["chance_same_cluster"] = 1.0 / len(CLUSTER_VOCAB)

    (RESULTS / "embedding_probe.json").write_text(json.dumps(report, indent=2))
    for key, value in report.items():
        print(f"{key:26s} {value:.3f}")


if __name__ == "__main__":
    main()
