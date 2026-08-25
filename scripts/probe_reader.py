"""Two controls on the frozen reader, both prerequisites of the study.

A. Evidence integration at k = 4. If the reader could not exploit a relevant
   memory placed in its context, every retrieval rule would score the same and
   the comparison would be vacuous. Measured on three context types:

       solvable   one memory shares the latent rule, its action was correct;
       matching   one memory shares the latent rule, its action was wrong;
       unrelated  no memory shares the latent rule.

   Expected: high on `solvable`, below the 1/8 chance level on the other two,
   since a memory marked as a failure removes its action from consideration.

B. Calibration at k = 1. The decision-aware score evaluates every candidate in
   a context of exactly one memory, so it is only as good as the reader's
   sensitivity to a lone memory. Reported as the mean information gain
   H(pi(.|s)) - H(pi(.|s,m)) for the four kinds of single memory, and as the
   probability that a relevant successful memory outranks an irrelevant
   successful one. If that probability is near 0.5, the score ranks by the
   outcome flag rather than by relevance and the method cannot work.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dam.data import _situation
from dam.env import (
    N_ACTIONS,
    N_RULES,
    Experience,
    encode,
    serialize_memory,
    serialize_query,
)
from dam.model import pad_batch
from dam.train import load_reader

RESULTS = Path("results")
CONTEXT_TYPES = ("solvable", "matching", "unrelated")
EPS = 1e-9


def _other_rule(rule: int, rng: np.random.Generator) -> int:
    other = int(rng.integers(N_RULES))
    while other == rule:
        other = int(rng.integers(N_RULES))
    return other


def _memory(rule: int, action: int, perm: np.ndarray, rng) -> Experience:
    return Experience(_situation(rule, 0.0, rng), action,
                      action == int(perm[rule]), 0)


def _context(mode: str, k: int, rng: np.random.Generator):
    perm = rng.permutation(N_ACTIONS)[:N_RULES]
    rule = int(rng.integers(N_RULES))
    current = _situation(rule, 0.0, rng)
    memories = []
    for j in range(k):
        r_i = rule if (j == 0 and mode != "unrelated") else _other_rule(rule, rng)
        action = int(perm[r_i]) if (j == 0 and mode == "solvable") \
            else int(rng.integers(N_ACTIONS))
        memories.append(_memory(r_i, action, perm, rng))
    ids: list[int] = []
    for m in memories:
        ids += encode(serialize_memory(m))
    return ids + encode(serialize_query(current)), int(perm[rule])


def _entropy(reader, device, ids: list[int]) -> float:
    x, lens = pad_batch([ids], reader.cfg.max_len, device)
    p = reader.policy(x, lens)[0]
    return float(-(p * torch.log(p + EPS)).sum())


def integration(reader, device, n: int, k: int) -> dict:
    rng = np.random.default_rng(0)
    out = {}
    for mode in CONTEXT_TYPES:
        hits = 0
        for _ in range(n):
            ids, label = _context(mode, k, rng)
            x, lens = pad_batch([ids], reader.cfg.max_len, device)
            hits += int(reader.policy(x, lens)[0].argmax().item() == label)
        out[f"acc_{mode}"] = hits / n
    return out


def calibration(reader, device, n: int) -> dict:
    """Information gain of a single memory, by relevance and by outcome."""
    rng = np.random.default_rng(1)
    gains = {key: [] for key in
             ("match_ok", "match_bad", "other_ok", "other_bad")}
    wins = 0
    for _ in range(n):
        perm = rng.permutation(N_ACTIONS)[:N_RULES]
        rule = int(rng.integers(N_RULES))
        current = _situation(rule, 0.0, rng)
        query = encode(serialize_query(current))
        h0 = _entropy(reader, device, query)

        other = _other_rule(rule, rng)
        wrong = int(rng.integers(N_ACTIONS))
        while wrong == int(perm[rule]):
            wrong = int(rng.integers(N_ACTIONS))
        cases = {
            "match_ok": _memory(rule, int(perm[rule]), perm, rng),
            "match_bad": _memory(rule, wrong, perm, rng),
            "other_ok": _memory(other, int(perm[other]), perm, rng),
            "other_bad": _memory(other, (int(perm[other]) + 1) % N_ACTIONS,
                                 perm, rng),
        }
        ig = {}
        for key, mem in cases.items():
            ig[key] = h0 - _entropy(reader, device,
                                    encode(serialize_memory(mem)) + query)
            gains[key].append(ig[key])
        wins += int(ig["match_ok"] > ig["other_ok"])
    out = {f"ig_{key}": float(np.mean(v)) for key, v in gains.items()}
    out["p_relevant_outranks_irrelevant"] = wins / n
    return out


def main(n: int = 800, k: int = 4) -> None:
    RESULTS.mkdir(exist_ok=True)
    reader, device = load_reader()
    report = {"chance": 1.0 / N_ACTIONS, "k": k, "n": n}
    report.update(integration(reader, device, n, k))
    report.update(calibration(reader, device, n))
    (RESULTS / "reader_probe.json").write_text(json.dumps(report, indent=2))
    for key, value in report.items():
        print(f"{key:32s} {value}")


if __name__ == "__main__":
    main()
