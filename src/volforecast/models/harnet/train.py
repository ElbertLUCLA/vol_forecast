"""Training, validation-based iteration selection, and refit for one HARNet cell.

Procedure (mirrors the XGBoost refit-and-freeze design):
  1. OLS-fit the HAR baseline on the train block and initialize every seed copy from it.
  2. Train on train-block labels (Adam, QLIKE, batches of 4 random windows x 5 labels),
     evaluating seed-averaged validation QLIKE every `eval_every` iterations.
     Iteration 0 is the OLS-initialized HAR itself.
  3. best_iter = the iteration with the lowest validation QLIKE.
  4. Refit: OLS on all purged pre-cutoff rows, re-initialize, train `best_iter`
     iterations on those rows with the same seeds, and freeze.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from volforecast.models.harnet.data import SeriesData, window_starts
from volforecast.models.harnet.model import DTYPE, HARNetEnsemble, ols_fit


def qlike_loss(y: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    ratio = y / pred
    return ratio - torch.log(ratio) - 1.0


def qlike(y: np.ndarray, pred: np.ndarray) -> float:
    ratio = np.asarray(y, dtype=float) / np.asarray(pred, dtype=float)
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def make_model(data: SeriesData, positions: np.ndarray, variant: dict, lags: list[int],
               n_seeds: int, clip_max: float) -> tuple[HARNetEnsemble, np.ndarray]:
    n_avg = 1 if variant["variant"] == "HARNet" else 2
    n_jump = len(data.channels) - n_avg
    coef = ols_fit(data.x, data.y, positions, lags, n_avg, n_jump)
    # Paper/reference: clip below at half the smallest training RV (scaled units).
    clip_min = 0.5 * float(np.min(data.x[positions, 0]))
    model = HARNetEnsemble(lags, n_avg, n_jump, coef, n_seeds, clip_min, clip_max)
    return model, coef


def predict(model: HARNetEnsemble, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Scaled predictions (S, len(positions)) for origins at `positions` (any sorted set)."""
    if len(positions) == 0:
        return np.empty((model.n_seeds, 0))
    lo, hi = int(positions.min()) - model.max_lag + 1, int(positions.max())
    seq = torch.as_tensor(x[lo : hi + 1], dtype=DTYPE)
    seq = seq.expand(model.n_seeds, 1, *seq.shape)
    with torch.no_grad():
        out = model(seq)[:, 0, :]  # (S, T)
    return out[:, positions - lo].numpy()


@dataclass
class TrainResult:
    model: HARNetEnsemble
    coef: np.ndarray
    curve: list[dict] = field(default_factory=list)  # eval history (iteration, val/train QLIKE)
    best_iter: int = 0
    seconds: float = 0.0


def train(
    data: SeriesData,
    fit_positions: np.ndarray,
    variant: dict,
    cfg: dict,
    seeds: list[int],
    n_iterations: int,
    val_positions: np.ndarray | None = None,
) -> TrainResult:
    tcfg = cfg["train"]
    torch.set_num_threads(int(tcfg.get("num_threads", 1)))
    torch.manual_seed(seeds[0])
    t0 = time.perf_counter()

    model, coef = make_model(data, fit_positions, variant, cfg["lags"], len(seeds), tcfg["clip_max"])
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"])

    L, B = tcfg["label_length"], tcfg["batch_size"]
    starts = window_starts(fit_positions, L)
    if len(starts) == 0:
        raise ValueError("No training windows; the fit block is shorter than label_length.")
    offsets = np.arange(-(model.max_lag - 1), L)  # window covers max_lag-1 history steps + L labels
    x_all = torch.as_tensor(data.x, dtype=DTYPE)
    y_all = torch.as_tensor(data.y, dtype=DTYPE)
    rngs = [np.random.default_rng(s) for s in seeds]  # one stream per seed: seed 42 is the same alone or in an ensemble

    result = TrainResult(model, coef)
    evaluate = val_positions is not None and len(val_positions) > 0
    best_state, best_val, since_best = None, np.inf, 0
    eval_every, patience = tcfg["eval_every"], tcfg.get("patience_evals")

    def record(it: int) -> bool:
        nonlocal best_state, best_val, since_best
        val_pred = predict(model, data.x, val_positions).mean(axis=0)
        val_q = qlike(data.y[val_positions], val_pred)
        result.curve.append({"iteration": it, "val_qlike": val_q})
        if val_q < best_val - 1e-12:
            best_val, since_best, result.best_iter = val_q, 0, it
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
        return patience is not None and since_best >= patience

    if evaluate:
        record(0)
    for it in range(1, n_iterations + 1):
        idx = np.stack([rng.choice(starts, size=B) for rng in rngs])  # (S, B)
        xb = x_all[idx[..., None] + offsets]                          # (S, B, T, C)
        yb = y_all[idx[..., None] + np.arange(L)]                     # (S, B, L)
        pred = model(xb)[..., -L:]
        loss = qlike_loss(yb, pred).mean(dim=(1, 2)).sum()           # sum over seeds -> independent grads
        opt.zero_grad()
        loss.backward()
        opt.step()
        if evaluate and it % eval_every == 0 and record(it):
            break

    if evaluate and best_state is not None:
        model.load_state_dict(best_state)
    result.seconds = time.perf_counter() - t0
    return result
