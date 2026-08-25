"""Decoder-only transformer used as the agent's reader.

The reader consumes a serialized memory context followed by the current
situation and emits a distribution over a closed set of actions. The action set
being closed is a deliberate choice: the policy is then an exact categorical
distribution obtained by renormalising the logits of the action tokens, so the
decision-level quantities used by the retrieval rule in ``memory.py`` (KL
divergence, entropy) are computed exactly rather than estimated by sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import ACTION_IDS, PAD_ID, VOCAB


@dataclass
class ModelConfig:
    vocab_size: int = len(VOCAB)
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    max_len: int = 160
    dropout: float = 0.05


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                          dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, causal: torch.Tensor,
                pad_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=causal,
                         key_padding_mask=pad_mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class Reader(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.tok = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model,
                                padding_idx=PAD_ID)
        self.pos = nn.Embedding(self.cfg.max_len, self.cfg.d_model)
        self.drop = nn.Dropout(self.cfg.dropout)
        self.blocks = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layers)])
        self.ln_f = nn.LayerNorm(self.cfg.d_model)
        # Head restricted to the action set: the reader is a policy, not a
        # language model, so there is no reason to spend capacity on the rest
        # of the vocabulary.
        self.head = nn.Linear(self.cfg.d_model, len(ACTION_IDS))
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """ids: (B, T) right-padded. lengths: (B,). Returns action logits (B, A)."""
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device).unsqueeze(0)
        x = self.drop(self.tok(ids) + self.pos(pos))
        causal = torch.triu(torch.ones(t, t, device=ids.device, dtype=torch.bool),
                            diagonal=1)
        pad_mask = ids.eq(PAD_ID)
        for blk in self.blocks:
            x = blk(x, causal, pad_mask)
        x = self.ln_f(x)
        last = x[torch.arange(b, device=ids.device), lengths - 1]
        return self.head(last)

    @torch.no_grad()
    def policy(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Exact categorical policy over the action set, shape (B, A)."""
        return F.softmax(self.forward(ids, lengths), dim=-1)


def pad_batch(seqs: list[list[int]], max_len: int,
              device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    lens = [min(len(s), max_len) for s in seqs]
    t = max(lens)
    out = torch.full((len(seqs), t), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : lens[i]] = torch.tensor(s[: lens[i]], dtype=torch.long)
    return out.to(device), torch.tensor(lens, dtype=torch.long, device=device)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
