"""HARNet / HARNetSVJ in PyTorch (Reisenhofer, Bayer & Hautsch 2022, section 4 and eq. 29).

A port of HARNet/src/harnet/model.py (TensorFlow), with one extension: every
parameter has a leading *seed* dimension S, so S independently trained copies
(one per batch-sampling seed) run as a single vectorized model. Since Adam is
elementwise and the loss is a sum over seeds, each copy trains exactly as it
would alone.

Architecture for aggregation periods (j_1=1, j_2, ..., j_L):
  layer 1      : causal conv, kernel j_2, no dilation, init 1/j_2 (a j_2-day average)
  layer l >= 2 : causal conv, kernel j_{l+1}/j_l, dilation j_l, init j_l/j_{l+1}
  each conv    : no bias, ReLU, filters = number of averaged channels
                 (1 for HARNet; 2 for HARNetSVJ, RV and RS- averaged separately)
  output       : bias-free linear layer over [1, x_t, avg_{j_2}, ..., avg_{j_L} (, jump_t)],
                 initialized with the OLS coefficients of the matching HAR, then clipped
                 to [clip_min, clip_max].
With these initial values the network reproduces the HAR baseline exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

DTYPE = torch.float64  # tiny model; double precision keeps the init == OLS identity exact


def conv_specs(lags: list[int]) -> list[tuple[int, int]]:
    """(kernel_size, dilation) for each conv layer."""
    specs = [(lags[1], 1)]
    for a, b in zip(lags[1:], lags[2:]):
        specs.append((b // a, a))
    return specs


class HARNetEnsemble(nn.Module):
    """S copies of HARNet (n_jump=0) or HARNetSVJ (n_jump=1), sharing nothing but shape."""

    def __init__(
        self,
        lags: list[int],
        n_avg: int,
        n_jump: int,
        ols_coef: np.ndarray,
        n_seeds: int,
        clip_min: float,
        clip_max: float = 1000.0,
    ):
        super().__init__()
        self.lags = list(lags)
        self.n_avg = n_avg
        self.n_jump = n_jump
        self.n_seeds = n_seeds
        self.specs = conv_specs(self.lags)
        self.register_buffer("clip_min", torch.tensor(float(clip_min), dtype=DTYPE))
        self.register_buffer("clip_max", torch.tensor(float(clip_max), dtype=DTYPE))

        # Conv weights: (S, C_out, C_in, kernel), each output channel averages "its" input channel.
        self.convs = nn.ParameterList()
        for kernel, _ in self.specs:
            w = torch.zeros(n_seeds, n_avg, n_avg, kernel, dtype=DTYPE)
            for c in range(n_avg):
                w[:, c, c, :] = 1.0 / kernel
            self.convs.append(nn.Parameter(w))

        n_out = 1 + n_avg * len(self.lags) + n_jump
        coef = torch.as_tensor(np.asarray(ols_coef, dtype=float), dtype=DTYPE)
        if coef.shape != (n_out,):
            raise ValueError(f"expected {n_out} OLS coefficients, got {tuple(coef.shape)}")
        self.out = nn.Parameter(coef.expand(n_seeds, n_out).clone())

    @property
    def max_lag(self) -> int:
        return max(self.lags)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (S, B, T, C_in) -> regressors (S, B, F, T) in OLS column order."""
        x = x.transpose(-1, -2)  # (S, B, C, T)
        h = x[:, :, : self.n_avg]
        feats = [torch.ones_like(h[:, :, :1]), h]
        for w, (kernel, dilation) in zip(self.convs, self.specs):
            h = torch.relu(_causal_conv(h, w, dilation))
            feats.append(h)
        if self.n_jump:
            feats.append(x[:, :, self.n_avg :])
        return torch.cat(feats, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (S, B, T, C_in) -> clipped predictions (S, B, T). The first max_lag-1 steps lack history."""
        pred = torch.einsum("sbft,sf->sbt", self.features(x), self.out)
        return torch.clamp(pred, self.clip_min, self.clip_max)


def _causal_conv(h: torch.Tensor, w: torch.Tensor, dilation: int) -> torch.Tensor:
    """h: (S, B, C_in, T), w: (S, C_out, C_in, k) -> (S, B, C_out, T), zero-padded on the left."""
    kernel = w.shape[-1]
    T = h.shape[-1]
    pad = (kernel - 1) * dilation
    hp = torch.nn.functional.pad(h, (pad, 0))
    out = 0
    for i in range(kernel):
        # Tap i looks back (kernel - 1 - i) * dilation steps, as in Keras' causal Conv1D.
        start = i * dilation
        out = out + torch.einsum("soc,sbct->sbot", w[..., i], hp[..., start : start + T])
    return out


# ---------------------------------------------------------------- HAR baseline (initialization)

def har_regressors(x: np.ndarray, lags: list[int], n_avg: int, n_jump: int) -> np.ndarray:
    """Regressors at every position, in the network's order:
    [1, ch_0..ch_{n_avg-1}, mean_{j_2}(ch_0..), ..., mean_{j_L}(ch_0..), jump]. NaN before max_lag-1.
    """
    df = pd.DataFrame(x[:, :n_avg])
    cols = [np.ones(len(x))]
    for lag in lags:
        avg = df.rolling(lag, min_periods=lag).mean().to_numpy() if lag > 1 else df.to_numpy()
        cols.extend(avg.T)
    if n_jump:
        cols.extend(x[:, n_avg:].T)
    X = np.column_stack(cols)
    X[: max(lags) - 1] = np.nan
    return X


def ols_fit(x: np.ndarray, y: np.ndarray, positions: np.ndarray, lags: list[int], n_avg: int, n_jump: int) -> np.ndarray:
    """OLS of the scaled label on the HAR regressors over `positions` (paper's recommended init)."""
    X = har_regressors(x, lags, n_avg, n_jump)[positions]
    coef, *_ = np.linalg.lstsq(X, y[positions], rcond=None)
    return coef
