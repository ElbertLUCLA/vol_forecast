"""Mamba and LSTM regressors: daily-input window -> log variance forecast.

    x (B, L, C) -> Linear(C, d) -> backbone -> last time step (B, d) -> Linear(d, 1)

Mamba: transformers' MambaModel (Gu & Dao 2023), selective state-space layers. On machines
without the CUDA-only mamba_ssm kernels it runs its reference PyTorch implementation, which
works on CPU and on AMD GPUs (ROCm).
LSTM: torch.nn.LSTM, the control for "plain recurrence".

Both outputs are bounded, which matters for spike forecasts (cf. the XGBoost tree ceiling):
  Mamba ends with RMSNorm, y = g * x / rms(x), so ||x / rms(x)|| = sqrt(d) and
        head(y) <= b + sqrt(d) * ||w * g||   (attainable supremum)
  LSTM  h = o * tanh(c) lies in (-1, 1)^d, so head(h) < b + sum|w|.
"""
from __future__ import annotations

import math
import warnings

import torch
from torch import nn


class SeqRegressor(nn.Module):
    def __init__(self, arch: str, n_inputs: int, hidden_size: int, num_layers: int, mamba_cfg: dict | None = None):
        super().__init__()
        self.arch = arch
        if arch == "mamba":
            from transformers import MambaConfig, MambaModel
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()  # silence the "falling back to reference implementation" notes
            cfg = MambaConfig(hidden_size=hidden_size, num_hidden_layers=num_layers, vocab_size=2,
                              use_cache=False, **(mamba_cfg or {}))
            self.proj = nn.Linear(n_inputs, hidden_size)
            self.backbone = MambaModel(cfg)
            self.backbone.embeddings.requires_grad_(False)  # we feed inputs_embeds; the token table is unused
        elif arch == "lstm":
            self.proj = nn.Identity()
            self.backbone = nn.LSTM(n_inputs, hidden_size, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError(f"unknown arch {arch!r}")
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch == "mamba":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                h = self.backbone(inputs_embeds=self.proj(x)).last_hidden_state
        else:
            h, _ = self.backbone(x)
        return self.head(h[:, -1]).squeeze(-1)

    def output_ceiling(self) -> float:
        """Largest log forecast this network can output (see module docstring)."""
        w, b = self.head.weight.detach()[0], float(self.head.bias.detach()[0])
        if self.arch == "mamba":
            g = self.backbone.norm_f.weight.detach()
            return b + math.sqrt(w.numel()) * float(torch.linalg.vector_norm(w * g))
        return b + float(w.abs().sum())
