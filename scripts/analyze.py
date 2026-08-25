"""Aggregates results/runs.jsonl into tables and figures.

All intervals are 95% Student-t intervals over seeds. Comparisons against the
similarity baseline are paired by seed, because the situation stream and the
action-sampling randomness are shared across retrieval rules within a seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("results")
FIGS = RESULTS / "figures"

RULE_LABEL = {
    "none": "no memory",
    "recency": "recency",
    "random": "random",
    "similarity": "similarity",
    "similarity_success": "similarity + success filter",
    "mmr": "similarity + MMR",
    "influence_kl": "influence (KL magnitude)",
    "influence_ig_nodedup": "influence (IG), no merge",
    "influence_ig": "influence (IG) + merge",
    "influence_ig_floor": "influence (IG) + merge + score floor",
    "greedy_32": "greedy context assembly",
    "contrastive_32_2": "contrastive influence, pool 32",
    "contrastive_64_2": "contrastive influence, pool 64",
    "oracle": "oracle (same rule)",
}
ORDER = list(RULE_LABEL)


def load() -> list[dict]:
    return [json.loads(line) for line in
            (RESULTS / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line]


def ci95(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()), 0.0
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(h)


def group(rows: list[dict], key: str = "acc_tail") -> dict:
    out: dict[tuple, list[float]] = {}
    for r in rows:
        out.setdefault((r["rule"], r["rho"], r["k"]), []).append(r[key])
    return out


def paired_delta(rows: list[dict], rule: str, ref: str, rho: float, k: int,
                 key: str = "acc_tail") -> tuple[float, float, float]:
    """Mean paired difference rule - ref, its 95% half-width, and the p-value
    of a two-sided paired t-test over seeds."""
    a = {r["seed"]: r[key] for r in rows if r["rule"] == rule
         and r["rho"] == rho and r["k"] == k}
    b = {r["seed"]: r[key] for r in rows if r["rule"] == ref
         and r["rho"] == rho and r["k"] == k}
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    m, h = ci95(d)
    p = float(stats.ttest_rel([a[s] for s in seeds], [b[s] for s in seeds]).pvalue)
    return m, h, p


def table_main(rows: list[dict]) -> str:
    rows = [r for r in rows if r["k"] == 4 and r["rho"] in (0.0, 1.0)]
    g_acc, g_suf = group(rows, "acc_tail"), group(rows, "sufficient_tail")
    lines = ["| retrieval rule | acc, rho=1 | sufficient@k, rho=1 "
             "| acc, rho=0 | sufficient@k, rho=0 | acc delta vs similarity, rho=0 |",
             "|---|---|---|---|---|---|"]
    for rule in ORDER:
        cells = [RULE_LABEL[rule]]
        for rho in (1.0, 0.0):
            for g in (g_acc, g_suf):
                v = g.get((rule, rho, 4))
                if v is None:
                    cells.append("--")
                else:
                    m, h = ci95(np.array(v))
                    cells.append(f"{m:.3f} ± {h:.3f}")
        if rule == "similarity":
            cells.append("--")
        else:
            m, h, p = paired_delta(rows, rule, "similarity", 0.0, 4)
            cells.append(f"{m:+.3f} ± {h:.3f} (p={p:.3f})")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_sweep(rows: list[dict], axis: str) -> str:
    if axis == "rho":
        rows = [r for r in rows if r["k"] == 4]
        values = sorted({r["rho"] for r in rows})
        rules = ["none", "recency", "similarity", "influence_ig", "oracle"]
        head = "| retrieval rule | " + " | ".join(f"rho={v:g}" for v in values) + " |"
        getter = lambda rule, v: group(rows).get((rule, v, 4))  # noqa: E731
    else:
        rows = [r for r in rows if r["rho"] in (0.0, 1.0)]
        values = sorted({r["k"] for r in rows})
        rules = ["similarity", "influence_ig", "influence_ig_floor"]
        head = "| retrieval rule (rho=0) | " + " | ".join(f"k={v}" for v in values) + " |"
        getter = lambda rule, v: group(rows).get((rule, 0.0, v))  # noqa: E731
    lines = [head, "|" + "---|" * (len(values) + 1)]
    for rule in rules:
        cells = [RULE_LABEL[rule]]
        for v in values:
            got = getter(rule, v)
            cells.append("--" if not got else "{:.3f} ± {:.3f}".format(*ci95(np.array(got))))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fig_main(rows: list[dict]) -> None:
    rows = [r for r in rows if r["k"] == 4 and r["rho"] in (0.0, 1.0)]
    g = group(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, rho, title in zip(axes, (1.0, 0.0),
                              ("rho = 1: similarity is a sufficient statistic",
                               "rho = 0: similarity is uninformative")):
        vals = [ci95(np.array(g.get((r, rho, 4), [np.nan]))) for r in ORDER]
        m = [v[0] for v in vals]
        h = [v[1] for v in vals]
        colors = ["#8c8c8c" if not r.startswith("influence") else "#1f6fb4"
                  for r in ORDER]
        colors[ORDER.index("oracle")] = "#c0392b"
        ax.bar(range(len(ORDER)), m, yerr=h, capsize=3, color=colors)
        ax.axhline(1 / 8, ls=":", c="k", lw=1)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([RULE_LABEL[r] for r in ORDER], rotation=35,
                           ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("accuracy, last 50% of stream")
    fig.tight_layout()
    fig.savefig(FIGS / "main.png", dpi=160)


def fig_rho(rows: list[dict]) -> None:
    rows = [r for r in rows if r["k"] == 4]
    g = group(rows)
    values = sorted({r["rho"] for r in rows})
    fig, ax = plt.subplots(figsize=(6, 4))
    for rule in ["none", "recency", "similarity", "influence_ig", "oracle"]:
        m, h = zip(*[ci95(np.array(g.get((rule, v, 4), [np.nan]))) for v in values])
        ax.errorbar(values, m, yerr=h, marker="o", capsize=3,
                    label=RULE_LABEL[rule])
    ax.set_xlabel("coupling rho   (I(c;r) = 0 .. 3 bits)")
    ax.set_ylabel("accuracy, last 50% of stream")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "rho_sweep.png", dpi=160)


def fig_k(rows: list[dict]) -> None:
    g = group(rows)
    values = sorted({r["k"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, rho in zip(axes, (1.0, 0.0)):
        for rule in ["similarity", "influence_ig", "influence_ig_floor"]:
            m, h = zip(*[ci95(np.array(g.get((rule, rho, v), [np.nan])))
                         for v in values])
            ax.errorbar(values, m, yerr=h, marker="o", capsize=3,
                        label=RULE_LABEL[rule])
        ax.set_xlabel("retrieved slots k")
        ax.set_title(f"rho = {rho:g}", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy, last 50% of stream")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "k_sweep.png", dpi=160)


def fig_curves() -> None:
    curves = RESULTS / "curves"
    if not curves.exists():
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    win = 60
    for ax, rho in zip(axes, (1.0, 0.0)):
        for rule in ["recency", "similarity", "influence_ig", "oracle"]:
            files = sorted(curves.glob(f"{rule}_rho{rho:.2f}_seed*.npz"))
            if not files:
                continue
            a = np.stack([np.load(f)["accuracy"] for f in files])
            sm = np.stack([np.convolve(row, np.ones(win) / win, mode="valid")
                           for row in a])
            m = sm.mean(0)
            h = stats.t.ppf(0.975, len(sm) - 1) * sm.std(0, ddof=1) / np.sqrt(len(sm))
            x = np.arange(len(m)) + win
            ax.plot(x, m, label=RULE_LABEL[rule])
            ax.fill_between(x, m - h, m + h, alpha=0.15)
        ax.axhline(1 / 8, ls=":", c="k", lw=1)
        ax.set_xlabel("step in the stream")
        ax.set_title(f"rho = {rho:g}", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"accuracy, running mean over {win} steps")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "learning_curves.png", dpi=160)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    rows = load()
    text = ["## Main comparison (k = 4, capacity 128, 5 seeds)", "",
            table_main(rows), "", "## Coupling sweep", "",
            table_sweep(rows, "rho"), "", "## Budget sweep at rho = 0", "",
            table_sweep(rows, "k"), ""]
    out = "\n".join(text)
    (RESULTS / "tables.md").write_text(out, encoding="utf-8")
    print(out)
    fig_main(rows)
    fig_rho(rows)
    fig_k(rows)
    fig_curves()
    print(f"figures written to {FIGS}")


if __name__ == "__main__":
    main()
