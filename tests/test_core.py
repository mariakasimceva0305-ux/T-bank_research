"""Invariants the experiment depends on. Run with: pytest -q"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dam.env import (
    Environment,
    EnvConfig,
    Episode,
    STOI,
    all_situation_texts,
    encode,
    mutual_information,
    serialize_memory,
    serialize_query,
)
from dam.memory import MemoryStore, RULES, RetrievalContext, retrieve
from dam.model import ModelConfig, Reader, pad_batch


def _empirical_mi(rho: float, n: int = 40_000) -> float:
    env = Environment(EnvConfig(rho=rho, seed=0))
    pairs = np.array([(s.cluster, s.rule) for s in
                      (env.sample_situation() for _ in range(n))])
    joint = np.zeros((8, 8))
    np.add.at(joint, (pairs[:, 0], pairs[:, 1]), 1)
    joint /= joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    mask = joint > 0
    return float((joint[mask] * np.log2(joint[mask] / (px @ py)[mask])).sum())


@pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
def test_coupling_matches_closed_form(rho):
    """The knob does what the closed form says it does; the rho sweep is
    reported on an information axis, so this has to hold."""
    assert abs(_empirical_mi(rho) - mutual_information(rho)) < 0.05


def test_rho_extremes():
    env1 = Environment(EnvConfig(rho=1.0, seed=3))
    assert all(s.cluster == s.rule for s in
               (env1.sample_situation() for _ in range(500)))
    assert _empirical_mi(0.0) < 0.01


def test_rule_action_map_varies_across_instances():
    """No fixed rule -> action mapping can be memorised in the weights."""
    perms = {tuple(Environment(EnvConfig(seed=s)).perm) for s in range(20)}
    assert len(perms) > 15


def test_vocabulary_is_closed():
    for text in all_situation_texts()[::97]:
        assert all(tok in STOI for tok in text.split())
    env = Environment(EnvConfig(rho=0.4, seed=1))
    for _ in range(200):
        sit = env.sample_situation()
        exp = env.step(sit, 2)
        assert encode(serialize_memory(exp)) and encode(serialize_query(sit))


def test_sequences_fit_context_window():
    cfg = ModelConfig()
    env = Environment(EnvConfig(seed=0))
    sits = [env.sample_situation() for _ in range(7)]
    mems = [env.step(s, 0) for s in sits[:6]]
    ids = Episode(memories=mems, situation=sits[-1]).token_ids()
    assert len(ids) <= cfg.max_len


def _ctx(reader):
    texts = all_situation_texts()[:4000]
    rng = np.random.default_rng(0)
    table = {t: v for t, v in zip(
        texts, rng.normal(size=(len(texts), 16)).astype(np.float32))}
    for t in table:
        table[t] /= np.linalg.norm(table[t])
    return RetrievalContext(reader=reader, device=torch.device("cpu"),
                            embeddings=table, rng=np.random.default_rng(0),
                            max_len=reader.cfg.max_len), texts


FLOORED = ("influence_ig_floor",)   # may deliberately leave slots empty


def test_every_rule_respects_the_budget():
    reader = Reader().eval()
    ctx, _ = _ctx(reader)
    env = Environment(EnvConfig(rho=0.5, seed=0))
    store = MemoryStore(capacity=32)
    sits = []
    while len(store) < 32:
        sit = env.sample_situation()
        if sit.text not in ctx.embeddings:
            continue
        sits.append(sit)
        store.write(env.step(sit, int(env.rng.integers(8))))
    query = sits[0]
    for rule in RULES:
        got = retrieve(rule, store, query, 4, ctx)
        assert len(got) <= 4, rule
        if rule != "none" and rule not in FLOORED:
            assert len(got) == 4, rule
        assert len({id(g) for g in got}) == len(got), rule


def test_store_respects_capacity():
    env = Environment(EnvConfig(seed=0))
    store = MemoryStore(capacity=10)
    for _ in range(50):
        store.write(env.step(env.sample_situation(), 0))
    assert len(store) == 10


def test_policy_is_a_distribution():
    reader = Reader().eval()
    env = Environment(EnvConfig(seed=0))
    sit = env.sample_situation()
    ids, lens = pad_batch([Episode(memories=[], situation=sit).token_ids()],
                          reader.cfg.max_len, torch.device("cpu"))
    pi = reader.policy(ids, lens)
    assert pi.shape == (1, 8)
    assert torch.allclose(pi.sum(-1), torch.ones(1), atol=1e-5)


def test_padding_does_not_change_the_policy():
    """Decision scores are computed in batches of unequal length; padding must
    be inert, otherwise the scores would depend on batch composition."""
    reader = Reader().eval()
    env = Environment(EnvConfig(seed=0))
    sits = [env.sample_situation() for _ in range(3)]
    short = Episode(memories=[], situation=sits[0]).token_ids()
    long = Episode(memories=[env.step(sits[1], 1), env.step(sits[2], 2)],
                   situation=sits[0]).token_ids()
    alone = reader.policy(*pad_batch([short], reader.cfg.max_len,
                                     torch.device("cpu")))[0]
    batched = reader.policy(*pad_batch([short, long], reader.cfg.max_len,
                                       torch.device("cpu")))[0]
    assert torch.allclose(alone, batched, atol=1e-5)
