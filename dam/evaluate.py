"""Online evaluation of retrieval rules on a frozen reader.

Protocol. The stream of situations is generated in advance from the
environment seed, so every retrieval rule is evaluated on exactly the same
sequence of situations (a paired comparison). The store starts empty and is
filled only with the agent's own experience, including its failures. Actions
are sampled from the policy rather than taken greedily: with binary feedback
the agent has to keep some exploration to escape a rule whose only recorded
attempts were wrong, and sampling provides it without an extra hyperparameter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .env import Environment, EnvConfig, Episode
from .memory import MemoryStore, RetrievalContext, retrieve
from .model import Reader, pad_batch


@dataclass
class RunResult:
    rule: str
    rho: float
    seed: int
    k: int
    capacity: int
    accuracy: np.ndarray          # per step, 0/1
    sufficient: np.ndarray        # retrieved set contains a same-rule success
    same_rule: np.ndarray         # retrieved set contains a same-rule memory
    entropy: np.ndarray           # policy entropy at the decision point
    seconds: float

    def summary(self, tail: float = 0.5) -> dict:
        n = len(self.accuracy)
        s = int(n * (1 - tail))
        return dict(
            rule=self.rule, rho=self.rho, seed=self.seed, k=self.k,
            capacity=self.capacity, steps=n,
            acc_all=float(self.accuracy.mean()),
            acc_tail=float(self.accuracy[s:].mean()),
            sufficient_tail=float(self.sufficient[s:].mean()),
            same_rule_tail=float(self.same_rule[s:].mean()),
            entropy_tail=float(self.entropy[s:].mean()),
            seconds=self.seconds,
        )


def run_stream(rule: str, reader: Reader, device: torch.device,
               embeddings: dict[str, np.ndarray], rho: float, seed: int,
               steps: int = 400, k: int = 4, capacity: int = 128) -> RunResult:
    env = Environment(EnvConfig(rho=rho, seed=seed))
    situations = [env.sample_situation() for _ in range(steps)]

    # Action sampling and any stochastic retrieval share a seed across rules,
    # so differences between rules are not differences in randomness.
    act_rng = np.random.default_rng(100_000 + seed)
    ctx = RetrievalContext(reader=reader, device=device, embeddings=embeddings,
                           rng=np.random.default_rng(200_000 + seed),
                           max_len=reader.cfg.max_len)
    store = MemoryStore(capacity=capacity)

    acc = np.zeros(steps, dtype=np.float32)
    suf = np.zeros(steps, dtype=np.float32)
    same = np.zeros(steps, dtype=np.float32)
    ent = np.zeros(steps, dtype=np.float32)
    t0 = time.time()

    for t, sit in enumerate(situations):
        mems = retrieve(rule, store, sit, k, ctx)
        ids, lens = pad_batch([Episode(memories=mems, situation=sit).token_ids()],
                              reader.cfg.max_len, device)
        pi = reader.policy(ids, lens)[0].cpu().numpy().astype(np.float64)
        pi = pi / pi.sum()
        action = int(act_rng.choice(len(pi), p=pi))
        exp = env.step(sit, action)
        store.write(exp)

        acc[t] = float(exp.correct)
        same[t] = float(any(m.situation.rule == sit.rule for m in mems))
        suf[t] = float(any(m.situation.rule == sit.rule and m.correct for m in mems))
        ent[t] = float(-(pi * np.log(pi + 1e-12)).sum())

    return RunResult(rule=rule, rho=rho, seed=seed, k=k, capacity=capacity,
                     accuracy=acc, sufficient=suf, same_rule=same,
                     entropy=ent, seconds=time.time() - t0)


def collect_stream_texts(rho: float, seed: int, steps: int) -> list[str]:
    env = Environment(EnvConfig(rho=rho, seed=seed))
    return [env.sample_situation().text for _ in range(steps)]
