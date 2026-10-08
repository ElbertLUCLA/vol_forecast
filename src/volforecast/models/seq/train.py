"""Training loop for the sequence models: MSE on the log target, early stopping on validation.

Per cell (mirrors XGBoost's tune -> refit -> freeze):
  1. Tuning: for each lookback x learning rate, train seed 42 on the train block with
     early stopping on validation MSE (log). Pick the setting with the lowest best val MSE.
  2. Selection: train every seed on the train block with that setting; each seed's best
     epoch is recorded, and validation residuals of the seed-averaged forecast give the
     smearing factor S (out-of-sample, as for HAR/XGBoost).
  3. Refit: each seed is retrained from scratch on train + val for its own best-epoch count
     and frozen.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from volforecast.models.seq.data import SeqData
from volforecast.models.seq.model import SeqRegressor


@dataclass
class FitResult:
    model: SeqRegressor
    best_epoch: int
    best_val_mse: float
    curve: list[dict] = field(default_factory=list)
    seconds: float = 0.0


def make_model(cfg: dict, arch: str, n_inputs: int, seed: int) -> SeqRegressor:
    torch.manual_seed(seed)
    m = cfg["model"]
    return SeqRegressor(arch, n_inputs, m["hidden_size"], m["num_layers"], m.get("mamba"))


def predict(model: SeqRegressor, data: SeqData, positions: np.ndarray, lookback: int, device: str,
            batch: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(positions), batch):
            xb = torch.from_numpy(data.windows(positions[i : i + batch], lookback)).to(device)
            out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def fit(
    data: SeqData,
    cfg: dict,
    arch: str,
    seed: int,
    lookback: int,
    lr: float,
    fit_positions: np.ndarray,
    device: str,
    val_positions: np.ndarray | None = None,
    n_epochs: int | None = None,
) -> FitResult:
    """Train one model. With val_positions: early stopping, best weights restored.
    Without: exactly n_epochs epochs (the refit)."""
    t0 = time.perf_counter()
    tcfg = cfg["train"]
    model = make_model(cfg, arch, len(data.channels), seed).to(device)
    with torch.no_grad():  # start at the mean target so early epochs aren't spent learning the level
        model.head.bias.fill_(float(np.mean(data.y[fit_positions])))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                            weight_decay=tcfg["weight_decay"])
    gen = np.random.default_rng(seed)
    X = torch.from_numpy(data.windows(fit_positions, lookback)).to(device)
    Y = torch.from_numpy(data.y[fit_positions].astype(np.float32)).to(device)
    max_epochs = n_epochs if n_epochs is not None else tcfg["max_epochs"]
    B = tcfg["batch_size"]

    result = FitResult(model, best_epoch=max_epochs, best_val_mse=float("nan"))
    best_state, best_mse, since = None, np.inf, 0
    y_val = data.y[val_positions] if val_positions is not None else None
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.from_numpy(gen.permutation(len(fit_positions))).to(device)
        for i in range(0, len(order), B):
            idx = order[i : i + B]
            loss = torch.mean((model(X[idx]) - Y[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
        if val_positions is None:
            continue
        val_mse = float(np.mean((predict(model, data, val_positions, lookback, device) - y_val) ** 2))
        result.curve.append({"epoch": epoch, "train_loss_last": loss.item(), "val_mse": val_mse})
        if val_mse < best_mse - 1e-6:
            best_mse, since, result.best_epoch = val_mse, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= tcfg["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        result.best_val_mse = best_mse
    result.seconds = time.perf_counter() - t0
    return result
