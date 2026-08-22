import csv
import json
from os import makedirs
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from absl import app
from absl.logging import info
from scipy.stats import norm
from tqdm import tqdm


def fit_evs_betas_single_pos(
    Ic_np,
    P_np,
    theta=0.75e-3,
    device="cuda",
    max_iter_adam=4000,
    num_bins: int = 300,
    perc_clip: tuple = (1.0, 99.0),
):

    Ic = np.asarray(Ic_np).astype(np.float64).ravel()
    Pe = np.asarray(P_np).astype(np.float64).ravel()
    assert Ic.shape == Pe.shape, "Ic_np and P_np must have the same shape."

    valid = np.isfinite(Ic) & np.isfinite(Pe)
    Ic, Pe = Ic[valid], Pe[valid]

    qx1, qx2 = np.percentile(Ic, list(perc_clip))
    qy1, qy2 = np.percentile(Pe, list(perc_clip))
    keep = (Ic >= qx1) & (Ic <= qx2) & (Pe >= qy1) & (Pe <= qy2)
    Ic, Pe = Ic[keep], Pe[keep]
    print(f"Ic range: {Ic.min():.8f} ~ {Ic.max():.8f}, P range: {Pe.min():.8f} ~ {Pe.max():.8f}, bins: {num_bins}")
    edges = np.linspace(Ic.min(), Ic.max(), num_bins + 1)
    bin_idx = np.digitize(Ic, edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)
    Ic_bins, Pe_bins = [], []
    for b in range(num_bins):
        m = bin_idx == b
        Ic_bins.append(float(np.mean(Ic[m])))
        Pe_bins.append(float(np.mean(Pe[m])))
    Ic_bins = np.asarray(Ic_bins, dtype=np.float64)
    Pe_bins = np.asarray(Pe_bins, dtype=np.float64)
    y_bins = -norm.ppf(Pe_bins)
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    Ic_t = torch.as_tensor(Ic_bins, dtype=torch.float64, device=dev)
    y_t = torch.as_tensor(y_bins, dtype=torch.float64, device=dev)
    eps = 1e-12
    sp = torch.nn.Softplus(beta=1.0)

    def sp_pos(x):
        return sp(x) + eps

    b0 = torch.nn.Parameter(torch.tensor(0.0, device=dev))
    b1 = torch.nn.Parameter(torch.tensor(0.0, device=dev))
    b2 = torch.nn.Parameter(torch.tensor(-5.0, device=dev))
    b3 = torch.nn.Parameter(torch.tensor(-2.0, device=dev))
    b4 = torch.nn.Parameter(torch.tensor(-2.0, device=dev))
    b5 = torch.nn.Parameter(torch.tensor(0.0, device=dev))
    params = [b0, b1, b2, b3, b4, b5]

    def predict_qinv(Ic_vec):
        beta0 = sp_pos(b0)
        beta1 = sp_pos(b1)
        beta2 = sp_pos(b2)
        beta3 = sp_pos(b3)
        beta4 = sp_pos(b4)
        rho = torch.tanh(b5)

        V = beta1 * Ic_vec + beta2
        V = torch.clamp(V, min=1e-9)

        denom_inside = 2.0 * ((beta3 * Ic_vec) ** 2 + beta4**2 + 2.0 * rho * (beta3 * Ic_vec) * beta4)
        denom = torch.sqrt(torch.clamp(denom_inside, min=1e-18))
        yhat = (beta0 * theta * V) / denom
        return yhat

    def loss_fn():
        yhat = predict_qinv(Ic_t)
        resid = yhat - y_t

        mse = torch.mean(resid * resid)
        reg = 1e-6 * sum(p.pow(2).sum() for p in params)
        return mse + reg

    opt_adam = torch.optim.Adam(params, lr=5e-2)
    for _ in range(max_iter_adam):
        opt_adam.zero_grad(set_to_none=True)
        l = loss_fn()
        l.backward()
        opt_adam.step()

    with torch.no_grad():
        beta0 = float(sp_pos(b0).item())
        beta1 = float(sp_pos(b1).item())
        beta2 = float(sp_pos(b2).item())
        beta3 = float(sp_pos(b3).item())
        beta4 = float(sp_pos(b4).item())
        beta5 = float(torch.tanh(b5).item())

    return np.array([beta0, beta1, beta2, beta3, beta4, beta5], dtype=np.float32)


def plot_ic_vs_p_per_cfa(
    I_c: np.ndarray,
    P_e: np.ndarray,
    cfa_map: np.ndarray,
    out_dir: str,
    cfa_size: int,
    num_bins: int = 260,
    samples_per_bin: int = 200,
    point_size: float = 1.0,
    seed: int = 42,
):
    assert I_c.shape == P_e.shape == cfa_map.shape, "I_c, P_e, and cfa_map must have the same shape."
    rng = np.random.default_rng(seed)
    colors = {0: "#1eff0e", 1: "#202ae8", 2: "#ed0e81", 3: "#1eff0e"}
    titles = {0: "CFA (0,0)", 1: "CFA (0,1)", 2: "CFA (1,0)", 3: "CFA (1,1)"}

    fig, axes = plt.subplots(cfa_size, cfa_size, figsize=(8, 6))
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = np.array([axes])

    for k in range(cfa_size * cfa_size):
        ax = axes[k]
        mask = cfa_map == k
        x = I_c[mask].astype(np.float64).ravel()
        y = P_e[mask].astype(np.float64).ravel()

        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if x.size == 0:
            ax.set_title(titles[k] + " (no data)")
            continue

        qx1, qx2 = np.percentile(x, [1, 99.2])
        qy1, qy2 = np.percentile(y, [1, 99.2])
        keep = (x >= qx1) & (x <= qx2) & (y >= qy1) & (y <= qy2)
        x, y = x[keep], y[keep]
        if x.size == 0:
            ax.set_title(titles[k] + " (no valid data)")
            continue

        bin_edges = np.linspace(x.min(), x.max(), num_bins + 1)
        x_sampled, y_sampled = [], []
        for i in range(num_bins):
            in_bin = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
            idxs = np.nonzero(in_bin)[0]
            n_in_bin = len(idxs)
            if n_in_bin == 0:
                continue
            n_pick = min(samples_per_bin, n_in_bin)
            sel = rng.choice(idxs, size=n_pick, replace=False)
            x_sampled.append(x[sel])
            y_sampled.append(y[sel])

        if len(x_sampled) == 0:
            ax.set_title(titles[k] + " (no sampled data)")
            continue

        x_sampled = np.concatenate(x_sampled)
        y_sampled = np.concatenate(y_sampled)

        ax.scatter(x_sampled, y_sampled, s=point_size, alpha=0.35, color=colors[k], edgecolors="none")
        ax.set_title(titles[k])
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xlabel("I_c (intensity)")
        ax.set_ylabel("P_e (event probability)")

    fig.suptitle("EVS Calibration: Ic vs Pe (binned sampling, per 2×2 CFA)", fontsize=13)
    scatter_path = join(out_dir, "evs_scatter_Ic_vs_Pe_perCFA_binned.png")
    print(f"Saving I_c vs. P_e plot to {scatter_path}")
    fig.savefig(scatter_path, dpi=300)
    plt.close(fig)


def save_evs_betas(betas_2x2_6, out_dir, cfa_size, theta=0.75e-3):
    """
    Save EVS calibration betas into JSON and CSV files.
    Args:
        betas_2x2_6: (2,2,6) array of β0...β5 for each EVS CFA position
        out_dir: output directory
        theta: comparator threshold (for reference in description)
    """
    betas_to_save = {
        "description": "EVS noise calibration. Fitted parameters β0...β5 per EVS 2×2 CFA position.",
        "formula": ("Q^{-1}(P) = [β0 * θ * (β1*Ic + β2)] / " "sqrt( 2 * ( (β3*Ic)^2 + β4^2 + 2*β5*β3*Ic*β4 ) )"),
        "theta": float(theta),
        "cfa_size": [cfa_size, cfa_size],
        "num_betas": 6,
        "betas": {},
    }
    names = ["beta0", "beta1", "beta2", "beta3", "beta4", "beta5"]
    for i in range(cfa_size):
        for j in range(cfa_size):
            key = f"CFA_{i}_{j}"
            pixel_terms = []
            for k, name in enumerate(names):
                coeff = float(betas_2x2_6[i, j, k])
                pixel_terms.append({"name": name, "index": k, "coefficient": coeff})
            betas_to_save["betas"][key] = pixel_terms
    with open(join(out_dir, "evs_noise_fit_results.json"), "w") as f:
        json.dump(betas_to_save, f, indent=4)
    csv_file = join(out_dir, "evs_noise_fit_results.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["CFA_row", "CFA_col"] + names)
        for i in range(cfa_size):
            for j in range(cfa_size):
                row = [i, j] + [float(betas_2x2_6[i, j, k]) for k in range(6)]
                writer.writerow(row)
    info("Saved EVS fit results to evs_noise_fit_results.json and evs_noise_fit_results.csv")


def plot_evs_ic_vs_qinvP(
    Ic: np.ndarray,
    P: np.ndarray,
    betas_2x2_6: np.ndarray,
    save_dir: str,
    cfa_size: int,
    theta: float = 0.75e-3,
    num_samples: int = 20000,
    seed: int = 42,
):
    assert Ic.ndim == 2 and P.shape == Ic.shape, "Ic and P must have matching (H, W) shapes."
    assert betas_2x2_6.shape == (cfa_size, cfa_size, 6), f"betas_2x2_6 must have shape ({cfa_size}, {cfa_size}, 6)."
    H, W = Ic.shape
    rng = np.random.default_rng(seed)

    rr = np.arange(H)[:, None] % 2
    cc = np.arange(W)[None, :] % 2
    cfa_map = (rr * 2 + cc).astype(np.uint8)

    colors_pred = {0: "#1f77b4", 1: "#2ca02c", 2: "#d62728", 3: "#9467bd"}
    colors_true = {0: "#9ecae1", 1: "#98df8a", 2: "#ff9896", 3: "#c5b0d5"}
    titles = {0: "CFA (0,0)", 1: "CFA (0,1)", 2: "CFA (1,0)", 3: "CFA (1,1)"}

    P_pred = np.empty_like(P, dtype=np.float64)
    Ic_f64 = Ic.astype(np.float64)
    for r_off in range(cfa_size):
        for c_off in range(cfa_size):
            mask = cfa_map == (r_off * 2 + c_off)
            b0, b1, b2, b3, b4, b5 = betas_2x2_6[r_off, c_off, :].astype(np.float64)
            Ic_sel = Ic_f64[mask]

            V = b1 * Ic_sel + b2
            V = np.clip(V, 1e-12, None)

            denom_inside = 2.0 * ((b3 * Ic_sel) ** 2 + (b4**2) + 2.0 * b5 * (b3 * Ic_sel) * b4)
            denom_inside = np.clip(denom_inside, 1e-18, None)
            yhat = (b0 * theta * V) / np.sqrt(denom_inside)
            P_pred[mask] = norm.sf(yhat)

    fig, axes = plt.subplots(cfa_size, cfa_size, figsize=(10, 7), constrained_layout=True)
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = np.array([axes])

    for k in range(cfa_size * cfa_size):
        ax = axes[k]
        mask = cfa_map == k
        x_all = Ic_f64[mask].ravel()
        y_true_all = P[mask].astype(np.float64).ravel()
        y_pred_all = P_pred[mask].ravel()

        valid = np.isfinite(x_all) & np.isfinite(y_true_all) & np.isfinite(y_pred_all)
        x_all, y_true_all, y_pred_all = x_all[valid], y_true_all[valid], y_pred_all[valid]
        n = x_all.size
        if n > num_samples:
            idx = rng.choice(n, size=num_samples, replace=False)
            x, y_true, y_pred = x_all[idx], y_true_all[idx], y_pred_all[idx]
        else:
            x, y_true, y_pred = x_all, y_true_all, y_pred_all

        nbins = 260
        edges = np.linspace(x.min(), x.max(), nbins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        bin_idx = np.digitize(x, edges, right=False) - 1
        bin_idx = np.clip(bin_idx, 0, nbins - 1)
        cnt = np.bincount(bin_idx, minlength=nbins)
        sum_true = np.bincount(bin_idx, weights=y_true, minlength=nbins)
        sum_pred = np.bincount(bin_idx, weights=y_pred, minlength=nbins)
        mean_true = np.divide(sum_true, cnt, out=np.full(nbins, np.nan), where=cnt > 0)
        mean_pred = np.divide(sum_pred, cnt, out=np.full(nbins, np.nan), where=cnt > 0)
        valid_bins = cnt > 0

        ax.plot(
            centers[valid_bins],
            mean_true[valid_bins],
            linestyle="--",
            linewidth=2.5,
            alpha=0.9,
            color=colors_true[k],
            label="True mean",
        )

        ax.plot(
            centers[valid_bins],
            mean_pred[valid_bins],
            linestyle="--",
            linewidth=2.5,
            alpha=0.9,
            color=colors_pred[k],
            label="Pred mean",
        )
        ax.scatter(x, y_true, s=2.0, alpha=0.35, color=colors_true[k], edgecolors="none", label="True P")
        ax.scatter(x, y_pred, s=2.0, alpha=0.7, color=colors_pred[k], edgecolors="none", label="Pred P")
        y_98 = np.percentile(np.concatenate([y_true, y_pred]), 98.5)
        ax.set_ylim(0, y_98)
        ax.set_title(titles[k])
        ax.set_xlabel("I_c (intensity)")
        ax.set_ylabel("P (probability)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle("EVS: True vs Predicted Probability per 2×2 CFA", fontsize=14)
    out_path = join(save_dir, "evs_true_vs_pred_P_perCFA.png")
    print(f"Saving comparison plot to {out_path}")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def evs_calibrate(I_c, P_e, out_dir, N, theta=0.75e-3, device="cuda", iqr_clip=1.5, sensor="gen2"):
    makedirs(out_dir, exist_ok=True)
    H, W = I_c.shape
    eps = 1e-6
    P = np.clip(P_e, eps, 1 - eps)
    y = -norm.ppf(P)
    if sensor == "gen2":
        cfa_size = 1
    elif sensor == "eiger":
        cfa_size = 2

    cfa_map = np.empty((H, W), np.uint8)
    for r in range(H):
        for c in range(W):
            cfa_map[r, c] = (r % cfa_size) * cfa_size + (c % cfa_size)

    plot_ic_vs_p_per_cfa(I_c, P_e, cfa_map, out_dir, cfa_size)

    betas = np.zeros((cfa_size, cfa_size, 6), dtype=np.float32)
    for r_off in range(cfa_size):
        for c_off in range(cfa_size):
            mask = (np.arange(H)[:, None] % cfa_size == r_off) & (np.arange(W)[None, :] % cfa_size == c_off)

            Ic = I_c[mask].astype(np.float32)
            P = P_e[mask].astype(np.float32)
            beta_rc = fit_evs_betas_single_pos(Ic, P, theta=theta, device=device)
            betas[r_off, c_off, :] = beta_rc.astype(np.float32)

    np.save(join(out_dir, "evs_betas.npy"), betas)
    plot_evs_ic_vs_qinvP(I_c, P_e, betas_2x2_6=betas, save_dir=out_dir, theta=theta, cfa_size=cfa_size)
    save_evs_betas(betas, out_dir, cfa_size)
    return betas
