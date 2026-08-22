from __future__ import annotations

import json
import os
import random
import sys
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

from hesim.calibration.aps_noise_calibration_with_multiple_groups_fitting import (
    fit_noise_second_order_gpu,
    fit_noise_second_order_gpu_without_negative_remove,
    fit_with_row_interaction,
    fit_without_row_interaction,
    fit_without_row_interaction_not_negative,
)
from hesim.calibration.color_checker import _quad_bayer_map
from hesim.hisp.black_level_currection import get_black_level_corrector
from hesim.io import analytic_file_name, aps_raw_stack, find_aps_folder, read_raw, sorted_raws
from hesim.visualization import visualize_matrix_with_histogram

_ROOT = "./calibration_data/ARGB_ERGB_Eiger/"
_GROUPS = [
    ("resolution_board_color_checker_exp01ms_20240516153213128", 1),
    ("resolution_board_color_checker_exp02ms_20240516153221655", 2),
    ("resolution_board_color_checker_exp05ms_20240516153235224", 5),
    ("resolution_board_color_checker_exp10ms_20240516153244340", 10),
    ("resolution_board_color_checker_exp20ms_20240516153253743", 20),
    ("resolution_board_color_checker_exp40ms_20240516153312041", 40),
    ("resolution_board_color_checker_exp50ms_20240516153331327", 50),
    ("resolution_board_color_checker_exp80ms_20240516153343982", 80),
]


def _exposure_time_relation_with_intensity(means, times, out_prefix, with_blc, plot_vis):

    global_means = np.mean(means, axis=(1, 2))

    G, H, W = means.shape
    T_vec = times.reshape(G, 1, 1)

    S_t = T_vec.sum(0)
    S_tt = (T_vec**2).sum(0)
    S_ty = (T_vec * means).sum(0)
    S_y = means.sum(0)

    denom = G * S_tt - S_t**2
    slope = (G * S_ty - S_t * S_y) / denom
    intercept = (S_y - slope * S_t) / G

    pred = slope[None, :, :] * T_vec + intercept[None, :, :]
    sse = ((means - pred) ** 2).sum(0)
    sst = ((means - means.mean(0)) ** 2).sum(0) + 1e-12
    r2_map = 1.0 - sse / sst
    r2_map = np.clip(r2_map, 0, 1)

    good_ratio_95 = float(np.mean(r2_map > 0.95))
    good_ratio_99 = float(np.mean(r2_map > 0.99))
    info(f"Pixels with R²>0.95 : {good_ratio_95*100:.2f}%")
    info(f"Pixels with R²>0.99 : {good_ratio_99*100:.2f}%")

    if plot_vis:
        plt.figure(figsize=(4, 3))
        vmin, vmax = 0.95, 1.0
        plt.imshow(r2_map, cmap="viridis", vmin=vmin, vmax=vmax)
        cbar = plt.colorbar()
        cbar.set_label("R\u00b2")
        plt.title(r"R\u00b2 of intensity vs $\Delta t$")
        plt.axis("off")
        plt.text(
            0.01,
            0.01,
            f"R²>0.95  : {good_ratio_95*100:.2f}%",
            color="black",
            fontsize=10,
            transform=plt.gca().transAxes,
        )
        plt.text(
            0.01,
            0.08,
            f"R²>0.99  : {good_ratio_99*100:.2f}%",
            color="black",
            fontsize=10,
            transform=plt.gca().transAxes,
        )
        plt.tight_layout()
        plt.savefig(join(out_prefix, f"mean_vs_exposure_R2_with_blc_{with_blc}.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(4, 3))
        plt.imshow(intercept, cmap="hot")
        plt.colorbar(label="Intercept")
        plt.title("Per-pixel intercept of mean vs exposure")
        plt.axis("off")
        plt.savefig(join(out_prefix, f"intercept_with_blc_{with_blc}.png"), dpi=300)
        plt.close()

        np.save(join(out_prefix, f"intercept_with_blc_{with_blc}.npy"), intercept)

        visualize_matrix_with_histogram(
            intercept,
            title=f"Intercept",
            filename=join(out_prefix, f"intercept_with_blc_{with_blc}_w_histogram.png"),
            clip_percentile_low=1,
            clip_percentile_high=99,
        )

        plt.figure(figsize=(8, 6))
        plt.imshow(slope, cmap="hot")
        plt.colorbar(label="slope")
        plt.title("Per-pixel slope vs exposure")
        plt.axis("off")
        plt.savefig(f"{out_prefix}_slope_{with_blc}.png", dpi=800)
        plt.close()
        np.save(f"{out_prefix}_slope_with_blc_{with_blc}.npy", slope)

        fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(5, 3), sharex=False, constrained_layout=True)

        ax_full.plot(times, global_means, "o-", lw=1.5)
        _annotate(ax_full, times, global_means)
        ax_full.set_title("Global mean vs Exposure Time – full range")
        ax_full.set_xlabel("Exposure time (ms)")
        ax_full.set_ylabel("Global mean")
        ax_full.grid(True)

        mask = times <= 10
        ax_zoom.plot(times[mask], global_means[mask], "s-", color="tab:orange", lw=1.5)
        _annotate(ax_zoom, times[mask], global_means[mask])
        ax_zoom.set_title("Zoom-in (≤ 10 ms)")
        ax_zoom.set_xlabel("Exposure time (ms)")
        ax_zoom.set_ylabel("Global mean")
        ax_zoom.grid(True)

        fig.suptitle(f"Global mean vs Exposure  (with_blc = {with_blc})", fontsize=14)
        plt.tight_layout()
        fig.savefig(f"{out_prefix}_global_mean_vs_exposure_with_blc_{with_blc}.png", dpi=800)
        plt.close(fig)

    return slope, intercept


def _annotate(ax, xs, ys, y_off=0, x_off=0.3):
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.annotate(
            f"{y:.3f}",
            xy=(x, y),
            xytext=(x, y + y_off),
            fontsize=8,
            ha="right",
            va="bottom",
        )


def _quad_bayer_splitting(H, W, beta_1, beta_2, beta_3):
    CFA = _quad_bayer_map(H, W)
    names = ["R", "Gr", "Gb", "B"]
    color_color_channelnel = {}
    for cid, n in enumerate(names):
        m = CFA == cid
        color_color_channelnel[n] = dict(
            beta_shot=float(np.median(beta_1[m])),
            beta_read=float(np.median(beta_2[m])),
            beta_dcsn=float(np.median(beta_3[m])),
        )
    return color_color_channelnel


def _vis_noise_with_beta_gamma(idx_eval, times, means, vars_, noise_row, noise_base, out_dir, vis=True):
    def _save_heat(im, fname, title):
        v1, v2 = np.percentile(im, (5, 95))
        plt.figure(figsize=(6, 6))
        plt.imshow(im, cmap="cool", vmin=v1, vmax=v2)
        cbar = plt.colorbar()

        plt.title(title, fontsize=9)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(join(out_dir, fname), dpi=600)
        plt.close()

    t_eval = times[idx_eval]
    var_gt = vars_[idx_eval]
    mu_eval = means[idx_eval]

    H, W = var_gt.shape
    r_norm = np.arange(H, dtype=np.float32) / (H - 1)
    r_mat = np.repeat(r_norm[:, None], W, 1)

    def _predict(noise_dict, name):
        beta_1, beta_2, beta_3 = noise_dict["beta_1"], noise_dict["beta_2"], noise_dict["beta_3"]
        gamma_1, gamma_2, gamma_3 = noise_dict["gamma_1"], noise_dict["gamma_2"], noise_dict["gamma_3"]

        mu_related = (beta_1 + gamma_1 * r_mat) * mu_eval
        content_related = beta_2 + gamma_2 * r_mat
        time_related = (beta_3 + gamma_3 * r_mat) * t_eval
        _save_heat(
            mu_related, f"{name}_vis_noise_with_beta_gamma-mu_{idx_eval}.png", "(beta_1 + gamma_1 * r_mat) * mu_eval"
        )
        _save_heat(
            content_related, f"{name}_vis_noise_with_beta_gamma-content_{idx_eval}.png", "(beta_2 + gamma_2 * r_mat)"
        )
        _save_heat(
            time_related, f"{name}_vis_noise_with_beta_gamma-time_{idx_eval}.png", "(beta_3 + gamma_3 * r_mat) * t_eval"
        )
        gamma_related = gamma_1 * r_mat * mu_eval + gamma_2 * r_mat + gamma_3 * r_mat * t_eval
        _save_heat(gamma_related, f"{name}_vis_noise_with_beta_gamma-gamma_{idx_eval}.png", "Gamma related term")
        return (beta_1 + gamma_1 * r_mat) * mu_eval + (beta_2 + gamma_2 * r_mat) + (beta_3 + gamma_3 * r_mat) * t_eval

    var_hat_row = _predict(noise_row, "var_hat_row")
    var_hat_base = _predict(noise_base, "var_hat_base")

    for k in ("beta_1", "beta_2", "beta_3", "gamma_1", "gamma_2", "gamma_3"):
        _save_heat(noise_row[k], f"noise_row-{k}.png", k)
    for k in ("beta_1", "beta_2", "beta_3"):
        _save_heat(noise_base[k], f"noise_base-{k}.png", k)

    if vis:
        figs = dict(
            gt=var_gt,
            hat_r=var_hat_row,
            hat_b=var_hat_base,
            err_r=var_gt - var_hat_row,
            err_b=var_gt - var_hat_base,
        )
        fig, ax = plt.subplots(1, 5, figsize=(15, 4))
        for a, (name, img) in zip(ax.flat, figs.items()):
            v1, v2 = np.percentile(img, (2, 98))
            im = a.imshow(img, cmap="cool", vmin=v1, vmax=v2)
            plt.colorbar(im)
            a.set_title(name)
        plt.suptitle(f"Var maps @ {t_eval:.0f} ms", fontsize=12)
        plt.tight_layout()
        plt.savefig(join(out_dir, f"var_compare_{int(t_eval)}ms.png"), dpi=320)
        plt.close()


def _gaussian_noise_stripping(means, vars_, times, plot_vis, out_prefix):

    G, H, W = means.shape
    P = H * W
    noise_w_row = fit_with_row_interaction(means, vars_, times)
    noise_wo_row = fit_without_row_interaction(means, vars_, times)

    np.savez_compressed(f"{out_prefix}-noise_models.npz", noise_with_row=noise_w_row, noise_without_row=noise_wo_row)

    for i in range(G):
        t_eval = int(times[i])
        out_dir = join(out_prefix, f"{t_eval}ms")
        makedirs(out_dir, exist_ok=True)
        _vis_noise_with_beta_gamma(i, times, means, vars_, noise_w_row, noise_wo_row, out_dir, vis=True)

    beta_1 = noise_wo_row["beta_1"]
    beta_2 = noise_wo_row["beta_2"]
    beta_3 = noise_wo_row["beta_3"]

    color_channel = _quad_bayer_splitting(H, W, beta_1, beta_2, beta_3)

    if plot_vis:

        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        for a, im, t in zip(
            ax, [beta_1, beta_2, beta_3], [r"$\beta_{\rm shot}$", r"$\beta_{\rm read}$", r"$\beta_{\rm DCSN}$"]
        ):
            vmin, vmax = np.percentile(im, (1, 99))
            im_obj = a.imshow(np.clip(im, vmin, vmax), cmap="cool")
            plt.colorbar(im_obj)
            a.set_title(t)
            a.axis("off")
        plt.tight_layout()
        fig.savefig(f"{out_prefix}_beta_maps.png", dpi=800)
        plt.close(fig)

        M = means.reshape(G, P).T
        V = vars_.reshape(G, P).T
        samp = np.random.choice(P, 200000, replace=False)
        mu__s = M[samp].flatten()
        v_s = V[samp].flatten()
        plt.figure(figsize=(8, 8))
        plt.scatter(mu__s, v_s, s=1, alpha=0.05)
        mu__line = np.linspace(mu__s.min(), mu__s.max(), 256)
        plt.plot(mu__line, color_channel["Gr"]["beta_shot"] * mu__line + color_channel["Gr"]["beta_read"], "g-", lw=2)
        plt.plot(mu__line, color_channel["R"]["beta_shot"] * mu__line + color_channel["R"]["beta_read"], "r-", lw=2)
        plt.plot(
            mu__line,
            color_channel["Gb"]["beta_shot"] * mu__line + color_channel["Gb"]["beta_read"],
            "g.",
            lw=1,
            alpha=0.05,
        )
        plt.plot(mu__line, color_channel["B"]["beta_shot"] * mu__line + color_channel["B"]["beta_read"], "b-", lw=2)
        plt.xlabel("mean")
        plt.ylabel("variance")
        plt.title("Random-pixel Var-mu_  (color median fit)")
        plt.savefig(f"{out_prefix}_var_vs_mean_scatter.png", dpi=800)
        plt.close()

    return beta_1, beta_2, beta_3, color_channel


def _vis_image_noise_with_calibration(
    root: str,
    groups: Sequence[Tuple[str, float]],
    beta_1: np.ndarray,
    beta_2: np.ndarray,
    beta_3: np.ndarray,
    slope: np.ndarray,
    intercept: np.ndarray,
    noise_vis_dir: str,
    with_blc: bool,
    n_sample: int = 10,
) -> None:
    """
    For each exposure group pick *n_sample* RAW frames, strip the estimated
    clean signal, obtain the *residual noise*, plot its histogram and overlay
    the calibrated Gaussian PDF.

    Parameters
    ----------
    root            : calibration root folder
    groups          : [(folder, t_ms), ...]      (same order as `times`)
    beta_1,beta_2,beta_3 : H×W maps from calibration
    intercept       : H×W fixed-pattern offset ( Δt·N_FP + N_BLE + N_row^d )
    times           : (G,) exposures   – must align with `groups`
    noise_vis_dir   : folder to save figures
    n_sample        : how many images per exposure to visualise
    """
    makedirs(noise_vis_dir, exist_ok=True)
    H, W = beta_1.shape
    CFA = _quad_bayer_map(H, W)
    black_level_corrector = get_black_level_corrector() if with_blc else None

    for g_idx, (folder, t_ms) in enumerate(groups):
        aps_dir = find_aps_folder(join(root, folder))
        frames = sorted_raws(aps_dir)
        rng = random.Random(0xC0FFEE)
        sel_paths = rng.sample(frames, min(n_sample, len(frames)))
        info(f"Processing {len(sel_paths)} images from {folder} at {t_ms} ms exposure")

        I_clean = slope * t_ms
        var_pred = beta_1 * I_clean + beta_2 + beta_3 * t_ms
        sigma_pred = np.sqrt(np.clip(var_pred, 1e-8, None))
        sigma_median = np.median(sigma_pred)
        predicted_noise = np.random.normal(loc=0, scale=sigma_pred)

        for p in sel_paths:
            raw_o = read_raw(p).astype(np.float32)
            if with_blc:
                raw = black_level_corrector(raw_o, t_ms)
            else:
                raw = raw_o

            residual = raw - I_clean - intercept

            fig, ax = plt.subplots(1, 4, figsize=(24, 6))
            fig.suptitle(f"{basename(p)}-noise splitting (t={t_ms}ms)", fontsize=13)

            im0 = ax[0].imshow(raw_o, cmap="gray")
            ax[0].set_title(f"0. RAW image")
            ax[0].axis("off")
            plt.colorbar(im0)

            im1 = ax[1].imshow(raw, cmap="gray")
            ax[1].set_title(f"1. BLC{with_blc}")
            ax[1].axis("off")
            plt.colorbar(im1)

            im2 = ax[2].imshow(raw - intercept, cmap="gray")
            ax[2].set_title(f"2. Remove intercept")
            ax[2].axis("off")
            plt.colorbar(im2)

            im3 = ax[3].imshow(raw - intercept - residual, cmap="gray")
            ax[3].set_title(f"3. Remove residual (Clearn)")
            ax[3].axis("off")
            plt.colorbar(im3)

            plt.tight_layout()
            fn = join(noise_vis_dir, f"{t_ms}-{basename(p).rsplit('.',1)[0]}_noise_splitting.png")
            plt.savefig(fn, dpi=500)
            plt.close(fig)

            fig, axes = plt.subplots(1, 4, figsize=(16, 5))
            fig.suptitle(f"{basename(p)}  (t = {t_ms} ms)", fontsize=13)

            for cid, name in enumerate(["R", "Gr", "Gb", "B"]):
                ax0 = axes[cid]
                pix = residual[CFA == cid].ravel()
                mean, var = np.mean(pix), np.var(pix)
                info(f"{name} color_channelnel: mean={mean:.4E}, var={var:.4E}")
                vmin, vmax = np.percentile(pix, (0.4, 99.6))
                ax0.hist(pix, bins=256, density=True, alpha=0.4, label="measured", range=(vmin, vmax))
                ax0.legend(fontsize=8)

                pred_noise = predicted_noise[CFA == cid].ravel()
                info(
                    f"{name} color_channelnel predicted noise: mean={np.mean(pred_noise):.4E}, var={np.var(pred_noise):.4E}"
                )
                vmin, vmax = np.percentile(pix, (0.4, 99.6))
                ax0.hist(pred_noise, bins=256, density=True, alpha=0.4, label="predicted", range=(vmin, vmax))
                x = np.linspace(pix.min(), pix.max(), 256)
                pdf = norm.pdf(x, loc=0, scale=sigma_median)
                ax0.plot(x, pdf, "r-", lw=1.5, label="fitted PDF-median")
                ax0.set_title(f"{name} {t_ms}:m({mean:.4E}),v({var:.4E})~p_v({np.var(predicted_noise):.4E}),")
                ax0.legend(fontsize=8)

            plt.tight_layout()
            fn = join(noise_vis_dir, f"{t_ms}-{basename(p).rsplit('.',1)[0]}_noise_hist.png")
            plt.savefig(fn, dpi=500)
            plt.close(fig)


def _vis_var(G, vars_, groups, times, out_prefix):
    G, H, W = vars_.shape
    for i in range(G):
        var = vars_[i]
        t = int(times[i])
        folder = groups[i][0]
        var_row_mean = var.mean(1)
        var_wo_row = var - var_row_mean[:, None]

        fig, axs = plt.subplots(1, 3, figsize=(12, 5))
        im1 = axs[0].imshow(var, cmap="hot", vmin=0, vmax=np.percentile(var, 99))

        plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
        axs[0].set_title(f"Variance for {t} ms $\\Delta t$\n{var.min():.4e} to {var.max():.4e}")
        axs[0].axis("off")
        var_row_mean = np.tile(var_row_mean[:, None], (1, W))
        im2 = axs[1].imshow(var_row_mean, cmap="hot", vmin=0, vmax=np.percentile(var_row_mean, 99))

        plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
        axs[1].set_title(f"Row noise var for {t} ms $\\Delta t$\n{var_row_mean.min():.4e} to {var_row_mean.max():.4e}")
        axs[1].axis("off")
        im3 = axs[2].imshow(var_wo_row, cmap="hot", vmin=0, vmax=np.percentile(var_wo_row, 99))
        plt.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04)
        axs[2].set_title(
            f"Variance w/o row noise var for {t} ms $\\Delta t$\n{var_wo_row.min():.4e} to {var_wo_row.max():.4e}"
        )
        axs[2].axis("off")
        plt.tight_layout()
        fig.savefig(join(out_prefix, f"{basename(folder)}_{t}ms_var_map.png"), dpi=600)
        plt.close(fig)


def _vis_intercept(intercept, plot_vis, with_blc, out_prefix):

    H, W = intercept.shape
    row_noise = intercept.mean(1)

    fix_noise_wo_row = intercept - row_noise[:, None]
    if plot_vis:
        fig, ax = plt.subplots(1, 4, figsize=(20, 4))
        im0 = ax[0].imshow(np.tile(row_noise[:, None], (1, W)), cmap="gray")
        ax[0].set_title(f"ADC Row Noise")
        ax[0].axis("off")
        plt.colorbar(im0)
        im1 = ax[1].hist(row_noise.ravel(), bins=255, color="b", alpha=0.7)
        ax[1].set_title(f"Row Noise Histogram\n{row_noise.min():.4e},{row_noise.max():.4e}")
        im2 = ax[2].imshow(fix_noise_wo_row, cmap="gray")
        ax[2].set_title(f"fix_noise_wo_row")
        ax[2].axis("off")
        plt.colorbar(im2)
        vmin, vmax = np.percentile(fix_noise_wo_row, (0.2, 99.8))
        im3 = ax[3].hist(fix_noise_wo_row.ravel(), bins=255, color="b", alpha=0.7, range=(vmin, vmax))
        fix_noise_wo_row_mean = fix_noise_wo_row.mean()
        fix_noise_wo_row_std = fix_noise_wo_row.std()

        x = np.linspace(vmin, vmax, 500)
        gauss = np.exp(-0.5 * ((x - fix_noise_wo_row_mean) / fix_noise_wo_row_std) ** 2) / (
            fix_noise_wo_row_std * np.sqrt(2 * np.pi)
        )
        ax[3].plot(x, gauss, "r--", label="Fitted Gaussian")
        textstr = f"μ = {fix_noise_wo_row_mean:.8f}\nσ = {fix_noise_wo_row_std:.8f}"
        props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8)
        ax[3].text(
            0.5,
            0.55,
            textstr,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )

        ax[3].set_title(f"Fix Noise wo_row Histogram\n{fix_noise_wo_row.min():.4e},{fix_noise_wo_row.max():.4e}")
        fig.suptitle(f"Row Noise and Fix Noise without Row (with_blc={with_blc})", fontsize=14)
        fig.legend()
        plt.tight_layout()
        fig.savefig(f"{out_prefix}_row_noise_and_fix_noise_without_row_with_blc_{with_blc}.png", dpi=800)
        plt.close(fig)


def calibrate_noise_with_multiple_groups(
    out_prefix: str,
    root: str = _ROOT,
    groups: Sequence[Tuple[str, float]] = _GROUPS,
    with_blc: bool = True,
    plot_vis: bool = True,
) -> Dict[str, np.ndarray]:
    """multi-exposure three-term fit, **plus per-color_channelnel stats**"""
    stks, means, vars_, times = [], [], [], []
    for folder, t in groups:
        debug(f"Processing folder: {folder} with exposure time: {t} ms")
        aps_folder = find_aps_folder(join(root, folder))

        stk = aps_raw_stack(aps_folder, t, with_blc)
        stks.append(stk)
        means.append(stk.mean(0))
        var = stk.var(0, ddof=0)
        vars_.append(var)
        times.append(t)
        info(f"Processed {folder} with exposure time {t} ms, shape: {stk.shape}")

    means = np.stack(means)
    G, H, W = means.shape
    vars_ = np.stack(vars_)
    vars_row_ = vars_.mean(2)
    var_wo_row_ = vars_ - vars_row_[:, :, None]
    times = np.asarray(times, np.float32)
    np.savez_compressed(
        f"{out_prefix}_means_var_times.npz",
        means=means,
        vars_=vars_,
        times=times,
    )
    T = np.asarray(times, np.float32)[:, None, None]
    slope, intercept = _exposure_time_relation_with_intensity(means, times, out_prefix, with_blc, plot_vis)
    _vis_var(G, vars_, groups, times, out_prefix)

    row_noise = intercept.mean(1)
    _vis_intercept(intercept, plot_vis, with_blc, out_prefix)

    means_wo_bais = means - intercept[None, :, :]
    info(f"means_wo_bais:{means_wo_bais.shape}. slope:{slope.shape}, T:{T.shape}")
    beta_1, beta_2, beta_3, color_channel = _gaussian_noise_stripping(means_wo_bais, vars_, times, plot_vis, out_prefix)

    if plot_vis:
        noise_vis_ualization_folder = f"{out_prefix}/noise_vis/"
        makedirs(noise_vis_ualization_folder, exist_ok=True)
        _vis_image_noise_with_calibration(
            root=root,
            groups=groups,
            beta_1=beta_1,
            beta_2=beta_2,
            beta_3=beta_3,
            slope=slope,
            intercept=intercept,
            noise_vis_dir=noise_vis_ualization_folder,
            with_blc=with_blc,
        )

    np.savez_compressed(
        f"{out_prefix}.npz",
        beta_shot=beta_1,
        beta_read=beta_2,
        beta_dcsn=beta_3,
        intercept=intercept,
        times=times,
        row_noise=row_noise,
        color_channel_stats=color_channel,
        with_blc=with_blc,
    )
    with open(f"{out_prefix}_summary.json", "w") as f:
        json.dump(dict(per_color_channelnel=color_channel), f, indent=2)
