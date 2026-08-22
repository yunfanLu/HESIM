import json
import os
import random
import sys
from itertools import product
from math import ceil
from os import makedirs
from os.path import basename, join
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from absl.logging import debug, info, warning
from numpy.linalg import lstsq
from scipy.optimize import lsq_linear
from scipy.stats import norm
from tqdm import tqdm

from hesim.calibration.color_checker import _quad_bayer_map


def poly_feature_terms(order: int = 2):
    terms = []
    for i in range(order + 1):
        for j in range(order + 1):
            if (i, j) == (0, 0):
                terms.append((0, 0))
                continue
            if i + j <= order:
                terms.append((i, j))
    return terms


def plot_var_mean_scatter_by_channel(
    means: np.ndarray,
    vars_: np.ndarray,
    times: np.ndarray,
    betas4x4: np.ndarray,
    terms: list[tuple[int, int]],
    save_dir: str,
    num_samples: int = 1_000,
):
    G, H, W = means.shape
    CFA = _quad_bayer_map(H, W).ravel()
    chan_ids = {"R": 0, "Gr": 1, "Gb": 2, "B": 3}
    chan_col = {"R": "red", "Gr": "green", "Gb": "limegreen", "B": "blue"}
    M_flat = means.reshape(G, -1).transpose(1, 0)
    V_flat = vars_.reshape(G, -1).transpose(1, 0)

    mu_min, mu_max = M_flat.min(), M_flat.max()
    mu_line = np.linspace(mu_min, mu_max, 256)

    n_cols = 4
    n_rows = ceil(G / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

    for cname in ("R", "Gr", "Gb", "B"):
        cid = chan_ids[cname]
        color = chan_col[cname]
        axes = axes.flatten()

        def _poly(mu, t, beta_vec):
            y = np.zeros_like(mu, dtype=np.float32)
            for (i, j), b in zip(terms, beta_vec):
                y += b * ((mu * t) ** i) * (t**j)
            return y

        for g in range(G):
            ax = axes[g]
            t_g = float(times[g])

            idx = np.where(CFA == cid)[0]
            samp = np.random.choice(idx, size=min(num_samples, idx.size), replace=False)
            ax.scatter(M_flat[samp, g], V_flat[samp, g], s=1, alpha=0.01, color=color)

            line_style = ["-", "--", ":", "-."]
            markers = ["o", "s", "^", "x"]
            for channel_idx in range(4):
                beta_vec = betas4x4[cid, channel_idx, :]
                info(f"cid: {cid}, channel_idx: {channel_idx}, beta_vec: {beta_vec.shape}, {beta_vec}")
                ax.plot(
                    mu_line,
                    _poly(mu_line, t_g, beta_vec),
                    linewidth=1,
                    color=color,
                    marker=markers[channel_idx],
                    markersize=2,
                    markevery=10 + 10 * channel_idx,
                    linestyle=line_style[channel_idx],
                    label=rf"{cname}-{channel_idx//2},{channel_idx%2}",
                )
            xvmin, xvmax = np.percentile(M_flat[:, g], (0, 99))

            ax.set_xlim(xvmin, xvmax * 1.2)
            yvmin, yvmax = np.percentile(V_flat[:, g], (0, 99))

            ax.set_ylim(yvmin, yvmax * 1.2)
            ax.set_title(f"exp = {t_g:g} ms", fontsize=10)
            ax.grid(True, ls="--", alpha=0.2)
            ax.legend(fontsize=7, loc="upper left")

    fig.text(0.5, 0.04, "Mean $\\mu$", ha="center")
    fig.text(0.06, 0.5, "Variance", va="center", rotation="vertical")
    fig.suptitle(f"{cname} channel – Var vs Mean", fontsize=15)
    fig.tight_layout(rect=[0.08, 0.08, 1, 0.95])
    fig.savefig(join(save_dir, f"var_vs_mean.png"), dpi=300)
    plt.close(fig)


def var_prediction_by_fit_noise_poly_gpu_per_cfa(betas, terms, mu, t):
    """
    Compute Var_hat(x,y) for every pixel using the polynomial model
    fitted per-CFA-position.

        Var = Σ_k  β_k(pos) · μ^i · t^j    with (i,j)=terms[k]

    Returns
    -------
    var_hat : (H,W) np.float32
    """
    H, W = mu.shape
    cfa_h, cfa_w, n_terms = betas.shape
    assert (cfa_h, cfa_w) == (4, 4), "function assumes 4×4 Quad-Bayer"

    beta_full = np.empty((H, W, n_terms), dtype=np.float32)
    for r in range(cfa_h):
        for c in range(cfa_w):
            beta_full[r::cfa_h, c::cfa_w, :] = betas[r, c, :]

    mu_pow = {i: mu.astype(np.float32) ** i for i, _ in terms}
    t_pow = {j: (t**j) for _, j in terms}

    var_hat = np.zeros_like(mu, dtype=np.float32)
    for k, (i, j) in enumerate(terms):
        var_hat += beta_full[..., k] * mu_pow[i] * t_pow[j]

    return var_hat


def fit_with_row_interaction(means, vars_, times):
    info(f"fit_with_row_interaction: means:{means.shape}, vars_:{vars_.shape}, times:{times.shape}")
    G, H, W = means.shape
    P = H * W

    M = means.reshape(G, P).T
    V = vars_.reshape(G, P).T
    Tv = np.tile(times, (P, 1))

    rows = np.arange(H, dtype=np.float32) / (H - 1)
    row_map = np.repeat(rows[:, None], W, axis=1)
    row_flat = row_map.ravel()
    Rv = np.tile(row_flat[:, None], (1, G))

    X6 = np.stack(
        [
            M,
            np.ones_like(M),
            Tv,
            M * Rv,
            Rv,
            Tv * Rv,
        ],
        axis=2,
    )

    coeff = np.empty((P, 6), np.float32)
    for i in range(P):
        coeff[i], *_ = lstsq(X6[i], V[i], rcond=None)
    (beta_1, beta_2, beta_3, gamma_1, gamma_2, gamma_3) = coeff.T.reshape(6, H, W)
    return dict(beta_1=beta_1, beta_2=beta_2, beta_3=beta_3, gamma_1=gamma_1, gamma_2=gamma_2, gamma_3=gamma_3)


def fit_without_row_interaction(means, vars_, times):
    G, H, W = means.shape
    P = H * W
    M = means.reshape(G, P).T
    V = vars_.reshape(G, P).T
    Tv = np.repeat(times[None, :], P, 0)
    X = np.stack([M, np.ones_like(M), Tv], 2)
    beta_ = np.empty((P, 3), np.float32)
    for i in range(P):
        beta_[i], *_ = lstsq(X[i], V[i], rcond=None)
    beta_1, beta_2, beta_3 = beta_.T.reshape(3, H, W)
    return dict(beta_1=beta_1, beta_2=beta_2, beta_3=beta_3, gamma_1=0, gamma_2=0, gamma_3=0)


def fit_without_row_interaction_not_negative(means, vars_, times):
    G, H, W = means.shape
    P = H * W
    M = means.reshape(G, P).T
    V = vars_.reshape(G, P).T
    Tv = np.repeat(times[None, :], P, 0)
    X = np.stack([M, np.ones_like(M), Tv], 2)
    beta_ = np.empty((P, 3), np.float32)
    for i in range(P):

        res = lsq_linear(X[i], V[i], bounds=(0, np.inf))
        if not res.success:
            warning(f"lsq_linear for pixel {i} did not converge successfully: {res.message}")
        beta_[i] = res.x
    beta_1, beta_2, beta_3 = beta_.T.reshape(3, H, W)
    return dict(beta_1=beta_1, beta_2=beta_2, beta_3=beta_3, gamma_1=0, gamma_2=0, gamma_3=0)


def fit_noise_second_order_gpu(
    means: np.ndarray,
    vars_: np.ndarray,
    times: np.ndarray,
    device="cuda",
    ridge_lambda=1e-2,
    max_mem_frac=0.5,
    iqr_clip=1.5,
    clip_beta=None,
    return_r2=True,
) -> Dict[str, np.ndarray]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not found")

    means_t = torch.as_tensor(means, device=device, dtype=torch.float32)
    vars_t = torch.as_tensor(vars_, device=device, dtype=torch.float32)
    times_t = torch.as_tensor(times, device=device, dtype=torch.float32)

    G, H, W = means_t.shape
    P, n_feat = H * W, 6

    mu = means_t.reshape(G, P).T
    var = vars_t.reshape(G, P).T
    tmat = times_t.unsqueeze(0).repeat(P, 1)

    mu_c = mu - mu.mean(dim=1, keepdim=True)
    t_c = tmat - tmat.mean(dim=1, keepdim=True)

    if iqr_clip > 0:
        q1 = var.quantile(0.25, dim=1, keepdim=True)
        q3 = var.quantile(0.75, dim=1, keepdim=True)
        iqr = q3 - q1
        var = torch.clamp(var, q1 - iqr_clip * iqr, q3 + iqr_clip * iqr)

    feats = torch.stack([mu_c, t_c, torch.ones_like(mu_c), mu_c**2, t_c**2, mu_c * t_c], dim=2)

    free = torch.cuda.mem_get_info(device)[0]
    bytes_per_px = (G * (n_feat + 1) + n_feat**2) * 4
    B = min(int(free * max_mem_frac // bytes_per_px), 131072)
    B = max(B, 16384)

    coef_buf = torch.empty((P, n_feat), device=device)
    r2_buf = torch.empty(P, device=device) if return_r2 else None

    eye = ridge_lambda * torch.eye(n_feat, device=device).unsqueeze(0)

    for beg in tqdm(range(0, P, B), desc="GPU Ridge LS"):
        end = min(beg + B, P)
        X_blk = feats[beg:end]
        y_blk = var[beg:end]

        Xt = X_blk.transpose(1, 2)
        XtX = Xt @ X_blk + eye
        Xty = Xt @ y_blk.unsqueeze(-1)

        beta = torch.linalg.solve(XtX, Xty).squeeze(-1)
        coef_buf[beg:end] = beta

        if return_r2:
            y_hat = (X_blk @ beta.unsqueeze(-1)).squeeze(-1)
            ss_res = ((y_blk - y_hat) ** 2).sum(dim=1)
            ss_tot = ((y_blk - y_blk.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
            r2_buf[beg:end] = 1.0 - ss_res / (ss_tot + 1e-12)

        del X_blk, y_blk, beta, Xt, XtX, Xty
        if return_r2:
            del y_hat
        torch.cuda.empty_cache()

    coef_maps = coef_buf.cpu().numpy().reshape(H, W, n_feat)
    names = [f"beta_{i}" for i in range(1, n_feat + 1)]
    out = {k: coef_maps[..., i] for i, k in enumerate(names)}

    if clip_beta is not None:
        for m in out.values():
            m[m < 0] = 0
            np.clip(m, 0, clip_beta, out=m)

    if return_r2:
        out["r2"] = r2_buf.cpu().numpy().reshape(H, W)

    return out


def fit_noise_second_order_gpu_without_negative_remove(
    means: np.ndarray,
    vars_: np.ndarray,
    times: np.ndarray,
    device: str = "cuda",
    chunk: int = 3264 * 2448 // 4,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not found")
    means = torch.as_tensor(means, device=device, dtype=torch.float32)
    vars_ = torch.as_tensor(vars_, device=device, dtype=torch.float32)
    times = torch.as_tensor(times, device=device, dtype=torch.float32)
    G, H, W = means.shape
    P = H * W
    n_feat = 6
    mu = means.reshape(G, P).T
    var = vars_.reshape(G, P).T
    tmat = times.unsqueeze(0).repeat(P, 1)
    feats = torch.stack(
        [mu, tmat, torch.ones_like(mu), mu**2, tmat**2, mu * tmat],
        dim=2,
    )
    coef_buf = torch.empty((P, n_feat), device=device, dtype=torch.float32)
    for beg in tqdm(range(0, P, chunk), desc="GPU LS batching"):
        end = min(beg + chunk, P)
        X_blk = feats[beg:end]
        y_blk = var[beg:end].unsqueeze(-1)

        beta_blk = torch.linalg.lstsq(X_blk, y_blk).solution
        coef_buf[beg:end] = beta_blk.squeeze(-1)
    coef_maps = coef_buf.cpu().numpy().reshape(H, W, n_feat)
    names = [f"beta_{i}" for i in range(1, n_feat + 1)]
    return {k: coef_maps[..., i] for i, k in enumerate(names)}


def fit_noise_poly_gpu_per_cfa(
    means: np.ndarray,
    vars_: np.ndarray,
    times: np.ndarray,
    cfa_size: int = 4,
    poly_order: int = 3,
    device: str = "cuda",
    iqr_clip: float = 1.5,
):
    assert torch.cuda.is_available()
    G, H, W = means.shape

    cfa_map = np.empty((H, W), np.uint8)
    for r in range(H):
        for c in range(W):
            cfa_map[r, c] = (r % cfa_size) * cfa_size + (c % cfa_size)

    terms = poly_feature_terms(poly_order)
    n_feat = len(terms)
    betas = np.zeros((cfa_size, cfa_size, n_feat), dtype=np.float32)

    for r_off in range(cfa_size):
        for c_off in range(cfa_size):
            mask = cfa_map == (r_off * cfa_size + c_off)
            mu = means[:, mask].reshape(G, -1)
            var = vars_[:, mask].reshape(G, -1)
            mu = mu.flatten()
            var = var.flatten()
            t = np.repeat(times, mu.shape[0] // G)

            q1, q3 = np.percentile(var, [2, 98])
            iqr = q3 - q1
            low, hi = q1 - iqr_clip * iqr, q3 + iqr_clip * iqr
            keep = (var >= low) & (var <= hi)
            mu, var, t = mu[keep], var[keep], t[keep]

            mu_torch = torch.from_numpy(mu).to(device=device, dtype=torch.float32)
            t_torch = torch.from_numpy(t).to(device=device, dtype=torch.float32)
            I_torch = mu_torch * t_torch
            X_feat = []
            for i, j in terms:
                X_feat.append((I_torch**i) * (t_torch**j))
            X = torch.stack(X_feat, dim=1)
            y = torch.from_numpy(var).to(device=device, dtype=torch.float32)

            beta, *_ = torch.linalg.lstsq(X, y)
            betas[r_off, c_off] = beta[:n_feat].cpu().numpy()
    return betas, terms
