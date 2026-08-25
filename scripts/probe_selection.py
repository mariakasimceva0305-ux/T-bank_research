"""Selection quality at a fixed store, with the feedback loop removed.

In the online protocol a retrieval rule shapes the store it later reads from:
a rule that decides badly early collects few successful experiences and then
has nothing good to retrieve. Online accuracy therefore confounds selection
quality with a self-reinforcing dynamic.

Here the store is built once by a fixed random behaviour policy and then
frozen. Every rule answers the same queries against the same store, so the
comparison isolates selection. Two quantities are reported:

    sufficient   the selected set contains a same-rule successful memory;
    accuracy     the reader's argmax on the selected set is correct.

The gap between them is the point of the diagnostic: a rule can retrieve a
sufficient memory and still lose, if the other slots it fills push the decision
elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dam.embed import embed_texts
from dam.env import Environment, EnvConfig, Episode
from dam.evaluate import collect_stream_texts
from dam.memory import RULES, MemoryStore, RetrievalContext, retrieve
from dam.model import pad_batch
from dam.train import load_reader

RESULTS = Path("results")


def main(rho: float = 0.0, seed: int = 0, store_steps: int = 400,
         queries: int = 200, k: int = 4, capacity: int = 128) -> None:
    RESULTS.mkdir(exist_ok=True)
    reader, device = load_reader()
    table = embed_texts(sorted(set(collect_stream_texts(
        rho, seed, store_steps + queries + 100))))

    env = Environment(EnvConfig(rho=rho, seed=seed))
    behaviour = np.random.default_rng(7)
    store = MemoryStore(capacity=capacity)
    for _ in range(store_steps):
        sit = env.sample_situation()
        store.write(env.step(sit, int(behaviour.integers(8))))
    qs = [env.sample_situation() for _ in range(queries)]

    report = {"rho": rho, "seed": seed, "k": k, "capacity": capacity,
              "queries": queries,
              "store_successes": int(sum(e.correct for e in store.items)),
              "store_size": len(store)}
    for rule in RULES:
        ctx = RetrievalContext(reader=reader, device=device, embeddings=table,
                               rng=np.random.default_rng(1),
                               max_len=reader.cfg.max_len)
        hits = suf = 0
        for q in qs:
            mems = retrieve(rule, store, q, k, ctx)
            x, lens = pad_batch(
                [Episode(memories=mems, situation=q).token_ids()],
                reader.cfg.max_len, device)
            hits += int(reader.policy(x, lens)[0].argmax().item()
                        == env.correct_action(q.rule))
            suf += int(any(m.situation.rule == q.rule and m.correct
                           for m in mems))
        report[rule] = {"accuracy": hits / len(qs), "sufficient": suf / len(qs)}
        print(f"{rule:22s} acc={hits/len(qs):.3f} suf={suf/len(qs):.3f}",
              flush=True)

    (RESULTS / f"selection_probe_rho{rho:g}.json").write_text(
        json.dumps(report, indent=2))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    main(**vars(ap.parse_args()))
