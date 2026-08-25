"""Memory store and the retrieval rules under comparison.

Every rule receives the same store, the same capacity M and the same number of
retrieved slots k, so the comparison is at equal memory budget. The rules
differ only in how they score a stored experience against the current
situation.

Two families are compared:

  description-based -- the score is a property of the stored text (cosine
      similarity of sentence embeddings, recency, diversity);
  decision-based -- the score is a property of the shift the memory induces in
      the agent's policy for the current situation.

The decision-based scores are computed exactly, because the action set is
closed and the policy is an explicit categorical distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .env import Experience, Situation, encode, serialize_memory, serialize_query
from .model import Reader, pad_batch

EPS = 1e-9


@dataclass
class MemoryStore:
    capacity: int
    items: list[Experience] = field(default_factory=list)

    def write(self, exp: Experience) -> None:
        self.items.append(exp)
        if len(self.items) > self.capacity:
            # First-in-first-out eviction, identical for every retrieval rule,
            # so capacity binds equally.
            self.items.pop(0)

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class RetrievalContext:
    reader: Reader
    device: torch.device
    embeddings: dict[str, np.ndarray]
    rng: np.random.Generator
    max_len: int = 160
    dedup_threshold: float = 0.95
    score_floor: float = 0.5   # used only by the influence_ig_floor ablation


def _emb(ctx: RetrievalContext, sit: Situation) -> np.ndarray:
    return ctx.embeddings[sit.text]


def _cosine(ctx: RetrievalContext, sit: Situation,
            items: list[Experience]) -> np.ndarray:
    q = _emb(ctx, sit)
    m = np.stack([_emb(ctx, e.situation) for e in items])
    return m @ q  # embeddings are pre-normalised


def policy_no_memory(ctx: RetrievalContext, sit: Situation) -> torch.Tensor:
    ids, lens = pad_batch([encode(serialize_query(sit))], ctx.max_len, ctx.device)
    return ctx.reader.policy(ids, lens)[0]


def policies_with_single_memory(ctx: RetrievalContext, sit: Situation,
                                items: list[Experience]) -> torch.Tensor:
    """pi(. | s, m) for every m in the store, one batched forward pass."""
    q = encode(serialize_query(sit))
    seqs = [encode(serialize_memory(e)) + q for e in items]
    out = []
    for i in range(0, len(seqs), 128):
        ids, lens = pad_batch(seqs[i: i + 128], ctx.max_len, ctx.device)
        out.append(ctx.reader.policy(ids, lens))
    return torch.cat(out, dim=0)


def _entropy_of(ctx: RetrievalContext, sit: Situation,
                contexts: list[list[Experience]]) -> np.ndarray:
    """Entropy of pi(. | s, C) for a batch of candidate contexts C."""
    q = encode(serialize_query(sit))
    seqs = []
    for ctxt in contexts:
        ids: list[int] = []
        for e in ctxt:
            ids += encode(serialize_memory(e))
        seqs.append(ids + q)
    out = []
    for i in range(0, len(seqs), 128):
        ids, lens = pad_batch(seqs[i: i + 128], ctx.max_len, ctx.device)
        p = ctx.reader.policy(ids, lens)
        out.append(-(p * torch.log(p + EPS)).sum(-1))
    return torch.cat(out).cpu().numpy()


def _greedy_context(ctx: RetrievalContext, sit: Situation,
                    items: list[Experience], k: int,
                    pool_size: int) -> list[Experience]:
    """Selects k memories by greedily minimising the decision uncertainty of
    the assembled context.

    Scoring a candidate in isolation and taking the top k does not work here,
    and the reason is a property of the reader rather than an implementation
    detail: with a single memory in context the reader has nothing to compare
    the marker against, so its entropy responds to the outcome flag far more
    than to relevance (see results/reader_probe.json). Scoring the marginal
    contribution of a candidate to a context that already holds the previous
    selections restores the comparison, and it optimises the object that
    actually determines the decision -- the assembled context -- instead of a
    sum of independent proxies for it.

    Cost is k * pool_size forward passes per decision, against pool_size for
    independent top-k selection.
    """
    pool = list(range(len(items)))
    if len(pool) > pool_size:
        pool = sorted(ctx.rng.choice(len(items), size=pool_size, replace=False))
    chosen: list[Experience] = []
    remaining = [items[i] for i in pool]
    while len(chosen) < k and remaining:
        entropies = _entropy_of(ctx, sit, [chosen + [c] for c in remaining])
        best = int(np.argmin(entropies))
        chosen.append(remaining.pop(best))
    return chosen


def _decision_scores(ctx: RetrievalContext, sit: Situation,
                     items: list[Experience]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (kl, information_gain, shift_directions).

    kl_m  = D_KL( pi(.|s,m) || pi(.|s) ), the magnitude of the decision shift.
    ig_m  = H( pi(.|s) ) - H( pi(.|s,m) ), the reduction in decision
            uncertainty; unlike the KL magnitude it is signed, and it is
            negative for memories that make the decision less determinate.
    shift = log pi(.|s,m) - log pi(.|s), used to detect memories that are
            distinct as text but equivalent as decisions.
    """
    p0 = policy_no_memory(ctx, sit)
    pm = policies_with_single_memory(ctx, sit, items)
    log_p0, log_pm = torch.log(p0 + EPS), torch.log(pm + EPS)
    kl = (pm * (log_pm - log_p0.unsqueeze(0))).sum(-1)
    h0 = -(p0 * log_p0).sum()
    hm = -(pm * log_pm).sum(-1)
    shift = log_pm - log_p0.unsqueeze(0)
    return (kl.cpu().numpy(), (h0 - hm).cpu().numpy(), shift.cpu().numpy())


def _contrastive_scores(ctx: RetrievalContext, sit: Situation,
                        cands: list[Experience], controls: list[Situation]
                        ) -> np.ndarray:
    """Situation-specific decision influence.

    The raw information gain of a memory confounds two things: how much it says
    about *this* situation, and how much it says about any situation at all. A
    memory recording that some action succeeded raises the reader's confidence
    even when its marker is irrelevant (results/reader_probe.json: 0.38 against
    0.41 for a relevant one), so ranking by raw gain ranks by outcome.

    Subtracting the gain the same memory produces on control situations removes
    that generic term and leaves the part that is specific to s:

        score(m | s) = IG(s, m) - mean_j IG(s_j, m)

    The controls are past situations drawn uniformly from the store, which uses
    no privileged information about the latent rule.
    """
    gain_here = _entropy_of(ctx, sit, [[]] * 1)[0] - _entropy_of(
        ctx, sit, [[c] for c in cands])
    generic = np.zeros_like(gain_here)
    for control in controls:
        base = _entropy_of(ctx, control, [[]] * 1)[0]
        generic += base - _entropy_of(ctx, control, [[c] for c in cands])
    return gain_here - generic / max(len(controls), 1)


def _topk_with_dedup(scores: np.ndarray, shift: np.ndarray | None, k: int,
                     threshold: float) -> list[int]:
    """Greedy selection by score, optionally skipping a candidate whose
    decision shift is nearly collinear with one already selected. Two
    experiences with different descriptions but the same effect on the policy
    are redundant under a fixed budget."""
    order = np.argsort(-scores)
    chosen: list[int] = []
    for idx in order:
        if len(chosen) == k:
            break
        if shift is not None and chosen:
            v = shift[idx]
            nv = np.linalg.norm(v) + EPS
            sel = shift[chosen]
            cos = (sel @ v) / ((np.linalg.norm(sel, axis=1) + EPS) * nv)
            if cos.max() > threshold:
                continue
        chosen.append(int(idx))
    if len(chosen) < k:  # dedup may leave slots open; fill by score
        for idx in order:
            if len(chosen) == k:
                break
            if int(idx) not in chosen:
                chosen.append(int(idx))
    return chosen


def retrieve(rule: str, store: MemoryStore, sit: Situation, k: int,
             ctx: RetrievalContext) -> list[Experience]:
    items = store.items
    if k == 0 or not items:
        return []
    k = min(k, len(items))

    if rule == "none":
        return []
    if rule == "recency":
        return items[-k:]
    if rule == "random":
        idx = ctx.rng.choice(len(items), size=k, replace=False)
        return [items[i] for i in idx]
    if rule == "similarity":
        sims = _cosine(ctx, sit, items)
        return [items[i] for i in np.argsort(-sims)[:k]]
    if rule == "similarity_success":
        # Controls for the possibility that the gain comes from conditioning on
        # successful outcomes rather than from decision-aware selection.
        # Successful memories are preferred but the k slots are always filled,
        # so this rule spends exactly the same budget as the others.
        sims = _cosine(ctx, sit, items)
        ok = np.array([e.correct for e in items])
        order = np.lexsort((-sims, ~ok))
        return [items[i] for i in order[:k]]
    if rule == "mmr":
        # Controls for the possibility that the gain comes from context
        # diversity rather than from decision relevance.
        sims = _cosine(ctx, sit, items)
        embs = np.stack([_emb(ctx, e.situation) for e in items])
        chosen: list[int] = []
        lam = 0.5
        for _ in range(k):
            best, best_v = None, -np.inf
            for i in range(len(items)):
                if i in chosen:
                    continue
                red = 0.0 if not chosen else float((embs[chosen] @ embs[i]).max())
                v = lam * sims[i] - (1 - lam) * red
                if v > best_v:
                    best, best_v = i, v
            chosen.append(int(best))
        return [items[i] for i in chosen]
    if rule.startswith("contrastive"):
        parts = rule.split("_")
        pool_size = int(parts[1]) if len(parts) > 1 else 32
        n_controls = int(parts[2]) if len(parts) > 2 else 2
        pool = list(range(len(items)))
        if len(pool) > pool_size:
            pool = sorted(ctx.rng.choice(len(items), size=pool_size,
                                         replace=False))
        cands = [items[i] for i in pool]
        controls = [items[i].situation for i in
                    ctx.rng.choice(len(items),
                                   size=min(n_controls, len(items)),
                                   replace=False)]
        scores = _contrastive_scores(ctx, sit, cands, controls)
        return [cands[i] for i in np.argsort(-scores)[:k]]
    if rule.startswith("greedy"):
        pool_size = int(rule.split("_")[-1]) if rule.split("_")[-1].isdigit() else 32
        return _greedy_context(ctx, sit, items, k, pool_size)
    if rule.startswith("influence"):
        # influence_ig_floor drops candidates scoring below half of the best
        # one, on the argument that the budget k is an upper bound rather than
        # a quota. It was measured and it loses (0.17 against 0.42 at equal
        # recall): on a real store the information gain of every successful
        # memory is within a few per cent of every other one, so the floor cuts
        # by rank rather than by relevance and simply shrinks the context.
        # A candidate pool smaller than the store trades accuracy for cost:
        # scoring is one forward pass per candidate, so the pooled variant is
        # what makes the rule affordable at a large store.
        pool = list(range(len(items)))
        if rule.endswith("_pool16") and len(items) > 16:
            pool = sorted(ctx.rng.choice(len(items), size=16, replace=False))
        cands = [items[i] for i in pool]
        kl, ig, shift = _decision_scores(ctx, sit, cands)
        scores = kl if rule == "influence_kl" else ig
        use_shift = None if rule == "influence_ig_nodedup" else shift
        idx = _topk_with_dedup(scores, use_shift, k, ctx.dedup_threshold)
        if rule == "influence_ig_floor" and ctx.score_floor > 0.0:
            best = float(max(scores[i] for i in idx))
            idx = [i for i in idx if scores[i] >= ctx.score_floor * best]
        return [cands[i] for i in idx]
    if rule == "oracle":
        # Upper bound: memories generated by the same latent rule, successful
        # ones first. Not implementable by an agent; it bounds the headroom.
        same = [e for e in items if e.situation.rule == sit.rule]
        same.sort(key=lambda e: (not e.correct, -e.step))
        if len(same) < k:
            rest = [e for e in items if e.situation.rule != sit.rule]
            same = same + rest[-(k - len(same)):]
        return same[:k]
    raise ValueError(f"unknown retrieval rule: {rule}")


RULES = [
    "none", "recency", "random", "similarity", "similarity_success", "mmr",
    "influence_kl", "influence_ig_nodedup", "influence_ig",
    "influence_ig_floor",
    "greedy_32", "contrastive_32_2", "contrastive_64_2",
    "oracle",
]
