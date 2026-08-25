"""Trains the reader once; every experiment then uses it frozen."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import make_batch
from .model import ModelConfig, Reader, n_parameters


def train(steps: int = 3000, batch_size: int = 64, lr: float = 6e-4,
          seed: int = 0, out: str = "checkpoints/reader.pt",
          eval_every: int = 250) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig()
    model = Reader(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / steps))
    )
    rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(10_000 + seed)

    history, t0 = [], time.time()
    for step in range(steps):
        model.train()
        x, lens, y = make_batch(rng, batch_size, cfg.max_len, device)
        loss = F.cross_entropy(model(x, lens), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if (step + 1) % eval_every == 0 or step == 0:
            model.eval()
            with torch.no_grad():
                xv, lv, yv = make_batch(eval_rng, 256, cfg.max_len, device)
                logits = model(xv, lv)
                acc = (logits.argmax(-1) == yv).float().mean().item()
                vloss = F.cross_entropy(logits, yv).item()
            history.append(dict(step=step + 1, train_loss=loss.item(),
                                val_loss=vloss, val_acc=acc,
                                elapsed=time.time() - t0))
            print(f"step {step+1:5d}  loss {loss.item():.3f}  "
                  f"val_loss {vloss:.3f}  val_acc {acc:.3f}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), cfg=vars(cfg)), path)
    meta = dict(steps=steps, batch_size=batch_size, lr=lr, seed=seed,
                n_parameters=n_parameters(model), history=history)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"saved {path}  ({n_parameters(model)/1e6:.2f}M parameters)")
    return meta


def load_reader(path: str = "checkpoints/reader.pt") -> tuple[Reader, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(path, map_location=device, weights_only=False)
    model = Reader(ModelConfig(**blob["cfg"])).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, device


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="checkpoints/reader.pt")
    train(**vars(ap.parse_args()))
