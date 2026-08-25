"""Runs every experiment reported in the study and writes results/runs.jsonl.

Stages
  main  all retrieval rules at rho in {1, 0}, k = 4        -- central question
  rho   coupling sweep for a subset of rules               -- hypothesis H2a
  k     budget sweep at rho in {1, 0}                      -- hypothesis H2b

All stages share one frozen reader and one precomputed embedding table.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dam.embed import embed_texts
from dam.env import mutual_information
from dam.evaluate import collect_stream_texts, run_stream
from dam.memory import RULES
from dam.train import load_reader

RESULTS = Path("results")
CURVES = RESULTS / "curves"

RHO_SWEEP_RULES = ["none", "recency", "similarity", "influence_ig", "oracle"]
K_SWEEP_RULES = ["similarity", "influence_ig", "influence_ig_floor"]


ALL_RHOS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _embeddings(cache: Path, seeds: int, steps: int) -> dict[str, np.ndarray]:
    """Embeddings for every situation that any stream in the study can emit.
    The streams are a deterministic function of (rho, seed), so this table is
    reproducible and is computed once."""
    if cache.exists():
        z = np.load(cache)
        return {t: z["vecs"][i] for i, t in enumerate(z["texts"])}
    texts = sorted({t for rho in ALL_RHOS for seed in range(seeds)
                    for t in collect_stream_texts(rho, seed, steps)})
    table = embed_texts(texts)
    np.savez_compressed(cache, texts=np.array(texts),
                        vecs=np.stack([table[t] for t in texts]))
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/reader.pt")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--capacity", type=int, default=128)
    ap.add_argument("--stages", default="main,rho,k")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    CURVES.mkdir(exist_ok=True)
    reader, device = load_reader(args.checkpoint)
    table = _embeddings(RESULTS / "embeddings.npz", args.seeds, args.steps)

    jobs: list[tuple[str, str, float, int, int]] = []
    stages = args.stages.split(",")
    if "main" in stages:
        for rho in (1.0, 0.0):
            for rule in RULES:
                for seed in range(args.seeds):
                    jobs.append(("main", rule, rho, 4, seed))
    if "rho" in stages:
        for rho in (0.25, 0.5, 0.75):        # 0.0 and 1.0 come from `main`
            for rule in RHO_SWEEP_RULES:
                for seed in range(args.seeds):
                    jobs.append(("rho", rule, rho, 4, seed))
    if "k" in stages:
        for rho in (1.0, 0.0):
            for k in (1, 2, 6):              # k = 4 comes from `main`
                for rule in K_SWEEP_RULES:
                    for seed in range(args.seeds):
                        jobs.append(("k", rule, rho, k, seed))

    # Resume: a run is identified by (stage, rule, rho, k, seed); anything
    # already in runs.jsonl is not recomputed, so an interrupted sweep can be
    # continued without duplicating rows.
    log = RESULTS / "runs.jsonl"
    done = set()
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                done.add((r["stage"], r["rule"], r["rho"], r["k"], r["seed"]))
    todo = [j for j in jobs if j not in done]
    print(f"{len(done)} runs already recorded, {len(todo)} to go")

    out = log.open("a", encoding="utf-8")
    t0 = time.time()
    for i, (stage, rule, rho, k, seed) in enumerate(todo, 1):
        res = run_stream(rule, reader, device, table, rho=rho, seed=seed,
                         steps=args.steps, k=k, capacity=args.capacity)
        row = res.summary()
        row.update(stage=stage, mi_bits=mutual_information(rho))
        out.write(json.dumps(row) + "\n")
        out.flush()
        if stage == "main" and k == 4:
            np.savez_compressed(
                CURVES / f"{rule}_rho{rho:.2f}_seed{seed}.npz",
                accuracy=res.accuracy, sufficient=res.sufficient,
                same_rule=res.same_rule, entropy=res.entropy)
        print(f"[{i}/{len(todo)}] {stage} {rule} rho={rho} k={k} seed={seed} "
              f"acc_tail={row['acc_tail']:.3f} suf={row['sufficient_tail']:.3f} "
              f"({res.seconds:.1f}s, total {time.time()-t0:.0f}s)", flush=True)
    out.close()


if __name__ == "__main__":
    main()
