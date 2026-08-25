"""Synthetic sequential environment with a controllable coupling between
surface (semantic) structure and decision-relevant structure.

Every situation is generated from two independent latent factors:

    c  -- surface cluster: determines the vocabulary of the situation text and
          therefore dominates any sentence embedding of it (11 content tokens).
    r  -- latent rule: exposed in the text through a single low-mass marker
          token, and it alone determines the correct action.

The correct action is a*(r) = perm[r], where ``perm`` is a bijection
rules -> actions drawn independently for every environment instance. The agent
cannot know ``perm`` a priori: it is recoverable only from past experience in
which some situation with the same marker was acted upon and scored.

The coupling parameter ``rho`` sets P(c = r); with probability 1 - rho the
cluster is drawn uniformly and independently of the rule. It therefore
interpolates between two regimes:

    rho = 1  semantic similarity is a sufficient statistic for utility;
    rho = 0  similarity carries no information about the rule, while the
             information is still present in the text, in the marker.

Both regimes share one generator, one agent and one memory budget: only rho
changes. This is what isolates the effect under study.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

N_CLUSTERS = 8
N_RULES = 8
N_ACTIONS = 8

# Cluster-specific surface vocabulary. Each cluster owns a disjoint set of
# words, so texts from different clusters are far apart under any lexical or
# sentence-embedding metric, and texts from the same cluster are close.
CLUSTER_VOCAB: list[dict[str, list[str]]] = [
    dict(project=["atlas", "atlas_core", "atlas_edge"],
         team=["platform", "platform_ops", "platform_sre"],
         verb=["rotate", "reissue", "renew"],
         obj=["credentials", "certificates", "keypairs"],
         service=["billing", "invoicing", "checkout"],
         region=["dc_west", "dc_west_b", "dc_west_c"]),
    dict(project=["nimbus", "nimbus_edge", "nimbus_mesh"],
         team=["network", "network_eng", "network_ops"],
         verb=["drain", "cordon", "evict"],
         obj=["node_pool", "edge_pool", "worker_set"],
         service=["gateway", "ingress", "proxy"],
         region=["eu_north", "eu_north_c", "eu_north_d"]),
    dict(project=["quarry", "quarry_etl", "quarry_lake"],
         team=["data", "data_eng", "data_platform"],
         verb=["backfill", "replay", "recompute"],
         obj=["partition", "shard", "segment"],
         service=["warehouse", "lakehouse", "catalog"],
         region=["us_east", "us_east_2", "us_east_3"]),
    dict(project=["harbor", "harbor_reg", "harbor_hub"],
         team=["release", "release_mgmt", "release_ops"],
         verb=["promote", "publish", "sign"],
         obj=["artifact", "bundle", "manifest"],
         service=["registry", "mirror", "cdn"],
         region=["ap_south", "ap_south_1", "ap_south_2"]),
    dict(project=["ledger", "ledger_svc", "ledger_core"],
         team=["finance", "fin_systems", "fin_ops"],
         verb=["reconcile", "settle", "post"],
         obj=["batch", "ledger_run", "journal"],
         service=["payments", "clearing", "payouts"],
         region=["dc_north", "dc_north_a", "dc_north_b"]),
    dict(project=["sentry", "sentry_gate", "sentry_watch"],
         team=["security", "appsec", "secops"],
         verb=["revoke", "quarantine", "isolate"],
         obj=["token", "session", "grant"],
         service=["auth", "identity", "sso"],
         region=["eu_west", "eu_west_3", "eu_west_4"]),
    dict(project=["cinder", "cinder_vol", "cinder_blk"],
         team=["storage", "storage_ops", "storage_eng"],
         verb=["resize", "detach", "migrate"],
         obj=["volume", "snapshot", "replica"],
         service=["object_store", "block_store", "archive"],
         region=["us_west", "us_west_1", "us_west_2"]),
    dict(project=["vertex", "vertex_ml", "vertex_lab"],
         team=["research", "ml_infra", "ml_ops"],
         verb=["retrain", "recalibrate", "distill"],
         obj=["checkpoint", "estimator", "embedding_index"],
         service=["ranker", "scorer", "retriever"],
         region=["ap_east", "ap_east_2", "ap_east_3"]),
]

# Decision-relevant marker: one common, semantically bland token per rule,
# disjoint from every cluster vocabulary. It contributes a single token out of
# twelve to the surface form. The choice of token is not cosmetic. With rare
# pseudo-tokens ("t7f", "q2a") the encoder treats the marker as the most
# distinctive part of the sentence and cosine similarity ranks by it, which
# would collapse the two regimes; see scripts/probe_embeddings.py for the
# measurement that fixed this choice.
MARKER_VOCAB = ["one", "two", "three", "four", "five", "six",
                "seven", "eight"]

ACTION_VOCAB = [f"proc_{i}" for i in range(N_ACTIONS)]


@dataclass(frozen=True)
class Situation:
    cluster: int
    rule: int
    text: str          # natural form, the only thing the sentence encoder sees
    tokens: tuple[str, ...]  # content words, the only thing the reader sees


@dataclass(frozen=True)
class Experience:
    """One unit of experience: a situation, the action taken on it, and the
    binary feedback the environment returned. The correct action is never
    revealed directly."""

    situation: Situation
    action: int
    correct: bool
    step: int


@dataclass
class EnvConfig:
    rho: float = 1.0
    seed: int = 0
    n_clusters: int = N_CLUSTERS
    n_rules: int = N_RULES
    n_actions: int = N_ACTIONS


class Environment:
    """Stream of situations with a per-instance rule -> action bijection."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # Rule -> action bijection, resampled per instance so that no mapping
        # can be memorised in the agent's weights.
        self.perm = self.rng.permutation(cfg.n_actions)[: cfg.n_rules]
        self._step = 0

    def correct_action(self, rule: int) -> int:
        return int(self.perm[rule])

    def sample_situation(self) -> Situation:
        rule = int(self.rng.integers(self.cfg.n_rules))
        if self.rng.random() < self.cfg.rho:
            cluster = rule % self.cfg.n_clusters
        else:
            cluster = int(self.rng.integers(self.cfg.n_clusters))
        text, tokens = render_situation(cluster, rule, self.rng)
        return Situation(cluster=cluster, rule=rule, text=text, tokens=tokens)

    def step(self, situation: Situation, action: int) -> Experience:
        correct = action == self.correct_action(situation.rule)
        exp = Experience(situation=situation, action=action,
                         correct=correct, step=self._step)
        self._step += 1
        return exp


FIELDS = ("project", "team", "verb", "obj", "service", "region")


def render_situation(cluster: int, rule: int,
                     rng: np.random.Generator) -> tuple[str, tuple[str, ...]]:
    """Returns the natural surface form and the content tokens.

    The encoder is given the natural form, because semantic similarity has to
    be measured on something a text encoder would actually be shown. The reader
    is given only the content tokens: the connectives carry no information and
    every token in the reader's context costs attention over a memory record
    that is repeated k times."""
    v = CLUSTER_VOCAB[cluster]
    picked = tuple(v[f][int(rng.integers(len(v[f])))] for f in FIELDS)
    proj, team, verb, obj, svc, reg = picked
    text = (f"project {proj} team {team} request {verb} {obj} "
            f"for {svc} in {reg} tag {MARKER_VOCAB[rule]}")
    return text, picked + (MARKER_VOCAB[rule],)


def mutual_information(rho: float, n_clusters: int = N_CLUSTERS,
                       n_rules: int = N_RULES) -> float:
    """I(c; r) in bits for the generator above, with uniform rules and
    n_clusters == n_rules. Reported alongside rho so the sweep has an
    information-theoretic x-axis rather than an arbitrary knob."""
    if n_clusters != n_rules:
        raise ValueError("closed form assumes n_clusters == n_rules")
    n = n_rules
    p_match = rho + (1.0 - rho) / n           # P(c = r | r)
    p_other = (1.0 - rho) / n                 # P(c = c' != r | r)
    mi = 0.0
    for p, mult in ((p_match, 1), (p_other, n - 1)):
        if p > 0:
            mi += mult * p * math.log2(p * n)  # P(c) = 1/n is uniform
    return mi


def build_vocabulary() -> list[str]:
    """Deterministic word-level vocabulary covering every surface form the
    generator and the serializer can emit."""
    words: set[str] = set()
    for v in CLUSTER_VOCAB:
        for vals in v.values():
            words.update(vals)
    words.update(MARKER_VOCAB)
    words.update(ACTION_VOCAB)
    words.update(["project", "team", "request", "for", "in", "tag"])
    specials = ["<pad>", "<mem>", "<cur>", "<act>", "<fb>", "<pred>",
                "<ok>", "<bad>"]
    return specials + sorted(words)


VOCAB = build_vocabulary()
STOI = {w: i for i, w in enumerate(VOCAB)}
ACTION_IDS = [STOI[a] for a in ACTION_VOCAB]
PAD_ID = STOI["<pad>"]


def serialize_memory(exp: Experience) -> str:
    """Layout: <mem> <cluster words> <marker> <act> <action> <fb> <outcome>.

    The separators were removed in one configuration on the hypothesis that
    placing the action immediately after the marker would make the copy easier
    to learn. It did not help the reader arbitrate between several successful
    records (0.30 against a 0.25 chance level, unchanged), and it destroyed the
    property the retrieval score depends on: the information gain of a single
    relevant memory became indistinguishable from that of an irrelevant one
    (1.24 against 1.24, ranking at chance). The separators are kept.
    """
    fb = "<ok>" if exp.correct else "<bad>"
    body = " ".join(exp.situation.tokens)
    return f"<mem> {body} <act> {ACTION_VOCAB[exp.action]} <fb> {fb}"


def serialize_query(sit: Situation) -> str:
    return f"<cur> {' '.join(sit.tokens)} <pred>"


def encode(tokens: str) -> list[int]:
    return [STOI[t] for t in tokens.split()]


@dataclass
class Episode:
    """A single decision point: retrieved memories plus the current situation."""

    memories: list[Experience] = field(default_factory=list)
    situation: Situation | None = None

    def token_ids(self) -> list[int]:
        ids: list[int] = []
        for m in self.memories:
            ids += encode(serialize_memory(m))
        ids += encode(serialize_query(self.situation))
        return ids


def all_situation_texts() -> list[str]:
    """Every surface form the generator can emit (8 clusters x 2^6 field
    choices x 8 markers). Enumerated once so sentence embeddings can be
    precomputed and cached instead of being recomputed per run."""
    import itertools

    texts = []
    for cluster, v in enumerate(CLUSTER_VOCAB):
        for combo in itertools.product(*(v[f] for f in FIELDS)):
            proj, team, verb, obj, svc, reg = combo
            for marker in MARKER_VOCAB:
                texts.append(
                    f"project {proj} team {team} request {verb} {obj} "
                    f"for {svc} in {reg} tag {marker}"
                )
    return texts
