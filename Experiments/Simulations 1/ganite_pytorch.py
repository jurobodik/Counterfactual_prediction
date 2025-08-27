"""
GANITE (Generative Adversarial Nets for ITE) — PyTorch reference implementation

Usage (Python):
    from ganite_pytorch import GANITE
    ganite = GANITE(input_dim=X.shape[1])
    ganite.fit(X, T, Y, num_epochs=200)
    y0_hat, y1_hat, ite_hat = ganite.predict(X_new)

Usage from R via reticulate:
    library(reticulate)
    gan <- import_from_path("ganite_pytorch", path=".")
    model <- gan$GANITE(input_dim=ncol(X))
    model$fit(X, T, Y, num_epochs=200L)
    pred <- model$predict(X_new)

Data expectations:
    X: numpy array (n, p), float32
    T: numpy array (n,), {0,1}
    Y: numpy array (n,), float32 (continuous outcome). For binary outcomes, set loss_y = 'bce'.

Notes:
    - This is a clean, minimal, readable template. Hyperparameters are conservative.
    - Stage 1 learns full potential outcomes; Stage 2 focuses on ITE.
    - Includes early stopping and simple metrics (factual RMSE). For counterfactual metrics, you’ll need semi-synthetic ground truth or sensitivity analyses.
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
# -----------------------------
# Utilities
# -----------------------------

def _to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return torch.tensor(x)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64), out_dim=1, dropout=0.1, act=nn.ReLU):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), act(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -----------------------------
# GANITE Modules
# -----------------------------

class CounterfactualGenerator(nn.Module):
    """G: generates missing potential outcomes.
    Input: [X, T, Y_f, Z] -> outputs [Y0_hat, Y1_hat]
    For factual arm, we copy Y_f; for counterfactual, use generated value.
    """
    def __init__(self, x_dim, z_dim=8, hidden=(128, 64), dropout=0.1):
        super().__init__()
        self.z_dim = z_dim
        self.mlp = MLP(x_dim + 1 + 1 + z_dim, hidden=hidden, out_dim=2, dropout=dropout)

    def forward(self, x, t, y_f):
        z = torch.randn(x.size(0), self.z_dim, device=x.device)
        inp = torch.cat([x, t.unsqueeze(1).float(), y_f.unsqueeze(1), z], dim=1)
        y_hat = self.mlp(inp)
        # replace factual with observed
        y0_hat = torch.where(t == 0, y_f, y_hat[:, 0])
        y1_hat = torch.where(t == 1, y_f, y_hat[:, 1])
        return y0_hat, y1_hat

class CounterfactualDiscriminator(nn.Module):
    """D: distinguishes factual vs generated given X, T, Y.
    Outputs probability that Y is factual.
    """
    def __init__(self, x_dim, hidden=(128, 64), dropout=0.1):
        super().__init__()
        self.clf = MLP(x_dim + 2, hidden=hidden, out_dim=1, dropout=dropout)

    def forward(self, x, t, y):
        logits = self.clf(torch.cat([x, t.unsqueeze(1).float(), y.unsqueeze(1)], dim=1)).squeeze(1)
        return logits

class ITEGenerator(nn.Module):
    """H: predicts ITE given X and generated potential outcomes.
    Input: [X, Y0_hat, Y1_hat] -> ITE_hat
    """
    def __init__(self, x_dim, hidden=(128, 64), dropout=0.1):
        super().__init__()
        self.mlp = MLP(x_dim + 2, hidden=hidden, out_dim=1, dropout=dropout)

    def forward(self, x, y0_hat, y1_hat):
        return self.mlp(torch.cat([x, y0_hat.unsqueeze(1), y1_hat.unsqueeze(1)], dim=1)).squeeze(1)

class ITEDiscriminator(nn.Module):
    """R: discriminator for ITE realism.
    Uses sign consistency with observed treatment effect proxies.
    """
    def __init__(self, x_dim, hidden=(128, 64), dropout=0.1):
        super().__init__()
        self.clf = MLP(x_dim + 1, hidden=hidden, out_dim=1, dropout=dropout)

    def forward(self, x, ite):
        return self.clf(torch.cat([x, ite.unsqueeze(1)], dim=1)).squeeze(1)

# -----------------------------
# Main wrapper
# -----------------------------

@dataclass
class GANITEConfig:
    lr_g: float = 1e-3
    lr_d: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 300
    patience: int = 20
    loss_y: str = 'mse'  # 'mse' or 'bce'
    lambda_adv: float = 1.0
    lambda_factual: float = 10.0
    lambda_ite_adv: float = 0.1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class GANITE:
    def __init__(self, input_dim: int, cfg: Optional[GANITEConfig] = None):
        self.cfg = cfg or GANITEConfig()
        self.G = CounterfactualGenerator(input_dim)
        self.D = CounterfactualDiscriminator(input_dim)
        self.H = ITEGenerator(input_dim)
        self.R = ITEDiscriminator(input_dim)
        self.to(self.cfg.device)

    # -------------------------
    # Helpers
    # -------------------------
    def to(self, device):
        self.G.to(device); self.D.to(device); self.H.to(device); self.R.to(device)
        return self

    def _yf_loss(self, y_pred, y_true):
        if self.cfg.loss_y == 'mse':
            return F.mse_loss(y_pred, y_true)
        elif self.cfg.loss_y == 'bce':
            return F.binary_cross_entropy_with_logits(y_pred, y_true)
        else:
            raise ValueError("loss_y must be 'mse' or 'bce'")

    # -------------------------
    # Fit
    # -------------------------
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray,
            num_epochs: Optional[int] = None, val_split: float = 0.2, seed: int = 7):
        cfg = self.cfg
        if num_epochs is None: num_epochs = cfg.num_epochs
        rng = np.random.RandomState(seed)
        n = X.shape[0]
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = int(n * val_split)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        def make_loader(I):
            x = _to_tensor(X[I]).float()
            t = _to_tensor(T[I]).long()
            y = _to_tensor(Y[I]).float()
            ds = TensorDataset(x, t, y)
            return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

        dl_tr = make_loader(tr_idx)
        dl_val = make_loader(val_idx)

        optG = torch.optim.Adam(self.G.parameters(), lr=cfg.lr_g)
        optD = torch.optim.Adam(self.D.parameters(), lr=cfg.lr_d)
        optH = torch.optim.Adam(self.H.parameters(), lr=cfg.lr_g)
        optR = torch.optim.Adam(self.R.parameters(), lr=cfg.lr_d)

        best_val = math.inf
        best_state = None
        patience = cfg.patience
        no_improve = 0

        bce_logits = nn.BCEWithLogitsLoss()

        for epoch in range(1, num_epochs + 1):
            self.G.train(); self.D.train(); self.H.train(); self.R.train()
            for x, t, y in dl_tr:
                x = x.to(cfg.device); t = t.to(cfg.device); y = y.to(cfg.device)

                # --------- Stage 1: Train D (discriminate factual vs generated)
                with torch.no_grad():
                    y0_hat, y1_hat = self.G(x, t, y)
                # Build dataset of (x, t, y_obs) labeled factual=1 and (x, 1-t, y_cf) labeled factual=0
                y_obs = torch.where(t == 1, y1_hat, y0_hat)
                y_cf  = torch.where(t == 1, y0_hat, y1_hat)
                t_obs = t
                t_cf = 1 - t

                d_logits_obs = self.D(x, t_obs.float(), y_obs)
                d_logits_cf  = self.D(x, t_cf.float(), y_cf)
                loss_D = bce_logits(d_logits_obs, torch.ones_like(d_logits_obs)) \
                       + bce_logits(d_logits_cf,  torch.zeros_like(d_logits_cf))
                optD.zero_grad(); loss_D.backward(); optD.step()

                # --------- Stage 1: Train G (fool D) + supervised factual loss
                y0_hat, y1_hat = self.G(x, t, y)
                y_obs = torch.where(t == 1, y1_hat, y0_hat)
                y_cf  = torch.where(t == 1, y0_hat, y1_hat)

                d_logits_cf = self.D(x, (1 - t).float(), y_cf)
                loss_adv = bce_logits(d_logits_cf, torch.ones_like(d_logits_cf))
                # factual supervised loss (predict y_obs close to true y)
                loss_factual = self._yf_loss(y_obs, y)
                loss_G = cfg.lambda_adv * loss_adv + cfg.lambda_factual * loss_factual
                optG.zero_grad(); loss_G.backward(); optG.step()

                # --------- Stage 2: Train R (ITE discriminator)
                with torch.no_grad():
                    y0_hat, y1_hat = self.G(x, t, y)
                    ite_hat = y1_hat - y0_hat
                    # proxy label: sign consistency with observed treatment effect relative to y
                    # delta_obs = (2t-1)*(y - y0_hat) approximates sign of ITE
                    delta_proxy = (2 * t.float() - 1.0) * (y - torch.where(t==1, y0_hat, y1_hat))
                r_logits = self.R(x, ite_hat)
                loss_R = bce_logits(r_logits, (delta_proxy > 0).float())
                optR.zero_grad(); loss_R.backward(); optR.step()

                # --------- Stage 2: Train H (ITE generator)
                y0_hat, y1_hat = self.G(x, t, y)
                ite_hat = self.H(x, y0_hat, y1_hat)
                r_logits = self.R(x, ite_hat)
                loss_H_adv = bce_logits(r_logits, torch.ones_like(r_logits))
                # tie ITE to difference in potential outcomes
                loss_H_align = F.mse_loss(ite_hat, (y1_hat - y0_hat).detach())
                loss_H = cfg.lambda_ite_adv * loss_H_adv + loss_H_align
                optH.zero_grad(); loss_H.backward(); optH.step()

            # ----- validation: factual RMSE
            self.G.eval();
            with torch.no_grad():
                se = 0.0; m = 0
                for x, t, y in dl_val:
                    x = x.to(cfg.device); t = t.to(cfg.device); y = y.to(cfg.device)
                    y0_hat, y1_hat = self.G(x, t, y)
                    y_obs = torch.where(t == 1, y1_hat, y0_hat)
                    se += F.mse_loss(y_obs, y, reduction='sum').item()
                    m += y.numel()
                rmse = math.sqrt(se / m)
            if rmse < best_val:
                best_val = rmse
                no_improve = 0
                best_state = {
                    'G': self.G.state_dict(), 'D': self.D.state_dict(),
                    'H': self.H.state_dict(), 'R': self.R.state_dict()
                }
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            self.G.load_state_dict(best_state['G'])
            self.D.load_state_dict(best_state['D'])
            self.H.load_state_dict(best_state['H'])
            self.R.load_state_dict(best_state['R'])
        return self

    # -------------------------
    # Predict
    # -------------------------
    def predict(self, X: np.ndarray, T: Optional[np.ndarray] = None, Y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.G.eval(); self.H.eval()
        x = _to_tensor(X).float().to(self.cfg.device)
        if T is None:
            # dummy for G; Y_f not used when T is None. Use zeros.
            t = torch.zeros(x.size(0), dtype=torch.long, device=self.cfg.device)
            y_f = torch.zeros(x.size(0), dtype=torch.float32, device=self.cfg.device)
        else:
            t = _to_tensor(T).long().to(self.cfg.device)
            y_f = _to_tensor(Y).float().to(self.cfg.device) if Y is not None else torch.zeros_like(t, dtype=torch.float32)
        with torch.no_grad():
            y0_hat, y1_hat = self.G(x, t, y_f)
            ite_hat = self.H(x, y0_hat, y1_hat)
        return y0_hat.cpu().numpy(), y1_hat.cpu().numpy(), ite_hat.cpu().numpy()

    # -------------------------
    # Convenience
    # -------------------------
    def fit_from_arrays(self, X, T, Y, **kwargs):
        return self.fit(X, T, Y, **kwargs)

# -----------------------------
# Simple preprocessing helpers
# -----------------------------

def standardize(train: np.ndarray, other: Optional[np.ndarray] = None):
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True) + 1e-8
    train_z = (train - mu) / sd
    if other is None:
        return train_z, mu, sd
    return (train - mu) / sd, (other - mu) / sd, mu, sd

if __name__ == "__main__":
    # Minimal smoke test on random data
    n, p = 2000, 10
    rng = np.random.RandomState(0)
    X = rng.randn(n, p).astype(np.float32)
    e = 1/(1+np.exp(-X[:,0]))
    T = (rng.rand(n) < e).astype(np.int64)
    # nonlinear outcomes
    Y0 = X[:,0] + 0.5*X[:,1] + rng.randn(n)*0.5
    Y1 = X[:,0]**2 - 0.5*X[:,2] + rng.randn(n)*0.5 + 1.0
    Y = np.where(T==1, Y1, Y0).astype(np.float32)

    model = GANITE(input_dim=p)
    model.fit(X, T, Y, num_epochs=50)
    y0h, y1h, iteh = model.predict(X)
    print("PEHE (synthetic, lower is better):", np.sqrt(np.mean(((Y1-Y0)-iteh)**2)))
