"""Training-data synthesis for the reader.

The reader is trained once, on contexts drawn from a retrieval-agnostic
mixture, and is then frozen for every experiment. This separates two questions
that are easy to conflate:

    (a) can the agent exploit relevant experience placed in its context?
    (b) does a given retrieval rule place relevant experience there?

Only (b) is under study. Training the reader jointly with a retrieval rule
would make the comparison of retrieval rules a comparison of readers as well.

Contexts are therefore built by sampling the number of rule-matching memories,
their outcomes and their surface clusters at random, over a coupling parameter
drawn uniformly from [0, 1]. The reader consequently sees the whole range of
regimes and is not tuned to any of them.
"""

from __future__ import annotations

import numpy as np
import torch

from .env import (
    N_ACTIONS,
    N_CLUSTERS,
    N_RULES,
    Experience,
    Situation,
    encode,
    render_situation,
    serialize_memory,
    serialize_query,
)
from .model import pad_batch


# Structure of a training context.
#
# Two properties of the reader are needed downstream, and both have to be put
# into the training distribution deliberately.
#
# 1. Discrimination among successes. A context in which several memories record
#    a successful action but only one carries the current marker is the hard
#    case, and it is the case every retrieval rule produces at evaluation time.
#    Sampling each memory independently makes it rare, and a reader trained
#    that way learns to copy the action of *a* successful memory rather than of
#    the relevant one: measured accuracy falls from 1.00 to 0.16 as the number
#    of successful distractors goes from 0 to 3. Here the number of successful
#    distractors is sampled explicitly and covers the whole range.
#
# 2. Calibration at k = 1. The decision-aware scores in memory.py evaluate a
#    candidate in a context of one memory, so single-memory contexts have to be
#    common enough for that regime to be trained rather than extrapolated.
K_CHOICES = (1, 2, 3, 4, 5)
K_WEIGHTS = (0.30, 0.20, 0.20, 0.15, 0.15)
P_SOLVABLE = 0.55


def _situation(rule: int, rho: float, rng: np.random.Generator) -> Situation:
    cluster = rule if rng.random() < rho else int(rng.integers(N_CLUSTERS))
    text, tokens = render_situation(cluster, rule, rng)
    return Situation(cluster=cluster, rule=rule, text=text, tokens=tokens)


def _other_rule(rule: int, rng: np.random.Generator) -> int:
    other = int(rng.integers(N_RULES))
    while other == rule:
        other = int(rng.integers(N_RULES))
    return other


def _wrong_action(rule: int, perm: np.ndarray, rng: np.random.Generator) -> int:
    action = int(rng.integers(N_ACTIONS))
    while action == int(perm[rule]):
        action = int(rng.integers(N_ACTIONS))
    return action


def sample_example(rng: np.random.Generator) -> tuple[list[int], int]:
    """Returns (token ids, label action)."""
    rho = float(rng.random())
    perm = rng.permutation(N_ACTIONS)[:N_RULES]
    k = int(rng.choice(K_CHOICES, p=K_WEIGHTS))
    rule = int(rng.integers(N_RULES))
    cur = _situation(rule, rho, rng)

    solvable = rng.random() < P_SOLVABLE
    memories: list[Experience] = []
    if solvable:
        memories.append(_experience(rule, int(perm[rule]), perm, rho, rng))
    elif rng.random() < 0.5 and k > 0:
        # A same-marker failure: informative, since it excludes one action,
        # but not sufficient.
        memories.append(_experience(rule, _wrong_action(rule, perm, rng),
                                    perm, rho, rng))

    n_rest = k - len(memories)
    n_success = int(rng.integers(0, n_rest + 1)) if n_rest > 0 else 0
    for j in range(n_rest):
        other = _other_rule(rule, rng)
        action = int(perm[other]) if j < n_success             else _wrong_action(other, perm, rng)
        memories.append(_experience(other, action, perm, rho, rng))

    memories = [memories[i] for i in rng.permutation(len(memories))]
    ids: list[int] = []
    for m in memories:
        ids += encode(serialize_memory(m))
    ids += encode(serialize_query(cur))
    return ids, int(perm[rule])


def _experience(rule: int, action: int, perm: np.ndarray, rho: float,
                rng: np.random.Generator) -> Experience:
    return Experience(situation=_situation(rule, rho, rng), action=action,
                      correct=action == int(perm[rule]), step=0)


def make_batch(rng: np.random.Generator, batch_size: int, max_len: int,
               device: torch.device):
    seqs, labels = [], []
    for _ in range(batch_size):
        ids, y = sample_example(rng)
        seqs.append(ids)
        labels.append(y)
    x, lens = pad_batch(seqs, max_len, device)
    return x, lens, torch.tensor(labels, dtype=torch.long, device=device)
