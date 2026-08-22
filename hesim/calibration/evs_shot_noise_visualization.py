#!/usr/bin/env python3

"""
Robust inverse visualization from EVS probability (no dark prior).
Input: evs_shot_noise.npz with keys:
  - positive_rate, negative_rate, evs_rate, total_frames[, H, W]
Output: denoised visuals + histograms + debug npy.
"""

import argparse
import os
from os.path import abspath, dirname, join

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter as _mf


def _median_filter_compat(x: np.ndarray, ksize: int) -> np.ndarray:
    k = max(3, int(ksize) | 1)
    x = np.asarray(x, dtype=np.float32)
    return _mf(x, size=k, mode="nearest")


def imsave_uint8(path, img01):
    img01 = np.clip(img01, 0.0, 1.0).astype(np.float32)
    img8 = (img01 * 255.0 + 0.5).astype(np.uint8)
    os.makedirs(dirname(path), exist_ok=True)
    cv2.imwrite(path, img8)


def save_hist_png(path, x, bins=512, vrange=None, title="Histogram"):
    x = np.asarray(x, dtype=np.float32)
    if vrange is None:
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
    else:
        lo, hi = vrange
    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    ax.hist(x.ravel(), bins=bins, range=(lo, hi), log=True, color="black")
    ax.set_title(f"{title}\n(Visual Range [{lo:.2e}, {hi:.2e}])")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count (log)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def robust_min_max(x, lo=0.0, hi=99.0):
    vmin = np.nanpercentile(x, lo)
    vmax = np.nanpercentile(x, hi)
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(x))
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(x))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def to01_linear(x, lo=0.0, hi=99.0):
    vmin, vmax = robust_min_max(x, lo, hi)
    y = (x - vmin) / (vmax - vmin)
    return np.clip(y, 0.0, 1.0)


def to01_logit(p, eps=1e-4):
    p = np.clip(p, eps, 1.0 - eps)
    y = np.log(p / (1.0 - p))
    return to01_linear(y, 0.0, 99.0)


def to01_rank(x):
    flat = x.ravel()
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=flat.size, endpoint=True, dtype=np.float32)
    return ranks.reshape(x.shape)


def invert_poisson_like(p, alpha=1.0, eps=1e-6):

    p = np.clip(p, 0.0, 1.0 - eps)
    return -np.log(1.0 - p) / max(alpha, eps)


def winsorize(x, lo=0.2, hi=99.8):
    lo_v = np.nanpercentile(x, lo)
    hi_v = np.nanpercentile(x, hi)
    return np.clip(x, lo_v, hi_v)


def local_mad_replace(x, ksize=7, thr=3.0):
    x = np.asarray(x, dtype=np.float32)
    m = _median_filter_compat(x, ksize)
    abs_dev = _median_filter_compat(np.abs(x - m), ksize)
    mad = abs_dev * 1.4826 + 1e-12
    z = np.abs(x - m) / mad
    y = x.copy()
    mask = z > thr
    y[mask] = m[mask]
    return y


def bilateral_smooth(x, diameter=7, sigma_color=0.06, sigma_space=3.0):

    xf = np.clip(x.astype(np.float32), 0.0, 1.0)
    out = cv2.bilateralFilter(xf, int(diameter), sigma_color * 255.0, sigma_space)
    return np.clip(out, 0.0, 1.0)


def pre_smooth(x, mode="median", ksize=5, sigma=1.0):
    x = np.asarray(x, dtype=np.float32)
    if mode == "none":
        return x
    if mode == "median":
        return _median_filter_compat(x, ksize)
    else:
        k = max(3, int(ksize) | 1)
        return cv2.GaussianBlur(x, (k, k), sigma)


def load_evs_npz(npz_path):
    data = np.load(npz_path)
    pos = data["positive_rate"].astype(np.float32)
    neg = data["negative_rate"].astype(np.float32)
    evs = data["evs_rate"].astype(np.float32)
    total_frames = int(data["total_frames"])
    H = int(data["H"]) if "H" in data.files else pos.shape[0]
    W = int(data["W"]) if "W" in data.files else pos.shape[1]
    return pos, neg, evs, total_frames, H, W


def main():
    ap = argparse.ArgumentParser("Robust inverse visualize without dark prior")
    ap.add_argument(
        "--npz", default="meta/ARGB_ERGB_Eiger_EVS_SHOT_Noise/evs_shot_noise.npz", help="path to evs_shot_noise.npz"
    )
    ap.add_argument("--outdir", default="meta/ARGB_ERGB_Eiger_EVS_SHOT_Noise/vis", help="output dir")
    ap.add_argument(
        "--method",
        default="logit",
        choices=["linear", "logit", "rank", "poisson"],
        help="inverse mapping for intensity",
    )

    ap.add_argument("--pre-smooth", default="median", choices=["none", "median", "gaussian"], help="pre smoothing")
    ap.add_argument("--pre-ksize", type=int, default=5, help="kernel for pre-smooth")
    ap.add_argument("--pre-sigma", type=float, default=1.0, help="sigma for gaussian")
    ap.add_argument("--winsor", type=float, default=0.2, help="winsorize percentile (lo=winsor, hi=100-winsor); 0=off")
    ap.add_argument("--mad-ksize", type=int, default=7, help="MAD kernel (odd>=3), 0=off")
    ap.add_argument("--mad-thr", type=float, default=3.0, help="MAD z-score threshold")
    ap.add_argument("--bilateral", action="store_true", default=True, help="apply bilateral")
    ap.add_argument("--bilateral-dia", type=int, default=7, help="bilateral diameter")
    ap.add_argument("--bilateral-sigc", type=float, default=0.06, help="bilateral sigma_color in [0..1]")
    ap.add_argument("--bilateral-sigs", type=float, default=3.0, help="bilateral sigma_space")
    ap.add_argument("--alpha", type=float, default=1.0, help="alpha for poisson-like inversion")
    args = ap.parse_args()

    pos, neg, evs, total_frames, H, W = load_evs_npz(args.npz)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[INFO] Loaded {args.npz} | H={H}, W={W}, frames={total_frames}")

    imsave_uint8(join(args.outdir, "positive_rate.png"), to01_linear(pos))
    imsave_uint8(join(args.outdir, "negative_rate.png"), to01_linear(neg))
    imsave_uint8(join(args.outdir, "evs_rate_raw.png"), to01_linear(evs))
    save_hist_png(join(args.outdir, "hist_evs_rate_raw.png"), evs, title="EVS rate (raw)")

    evs_clean = evs.copy()

    evs_clean = pre_smooth(evs_clean, mode=args.pre_smooth, ksize=args.pre_ksize, sigma=args.pre_sigma)

    if args.winsor and args.winsor > 0:
        evs_clean = winsorize(evs_clean, lo=args.winsor, hi=100.0 - args.winsor)

    if args.mad_ksize and args.mad_ksize >= 3 and (args.mad_ksize % 2 == 1):
        evs_clean = local_mad_replace(evs_clean, ksize=args.mad_ksize, thr=args.mad_thr)

    if args.bilateral:
        evs_clean = bilateral_smooth(
            evs_clean, diameter=args.bilateral_dia, sigma_color=args.bilateral_sigc, sigma_space=args.bilateral_sigs
        )

    if args.method == "linear":
        intensity01 = to01_linear(evs_clean)
    elif args.method == "logit":
        intensity01 = to01_logit(evs_clean)
    elif args.method == "rank":
        intensity01 = to01_rank(evs_clean)
    else:
        inv = invert_poisson_like(evs_clean, alpha=args.alpha)
        intensity01 = to01_linear(inv)

    imsave_uint8(join(args.outdir, f"intensity_{args.method}_robust.png"), intensity01)
    save_hist_png(
        join(args.outdir, f"hist_intensity_{args.method}.png"),
        intensity01,
        vrange=(0.0, 1.0),
        title=f"Intensity ({args.method})",
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        pos_ratio_given_event = np.divide(
            pos, pos + neg, out=np.zeros_like(pos, dtype=np.float32), where=(pos + neg) > 0
        )
    imsave_uint8(join(args.outdir, "pos_ratio_given_event_linear.png"), to01_linear(pos_ratio_given_event))
    imsave_uint8(join(args.outdir, "pos_ratio_given_event_rank.png"), to01_rank(pos_ratio_given_event))

    np.save(join(args.outdir, "debug_evs_clean.npy"), evs_clean.astype(np.float32))
    np.save(join(args.outdir, "debug_pos_rate.npy"), pos.astype(np.float32))
    np.save(join(args.outdir, "debug_neg_rate.npy"), neg.astype(np.float32))
    np.save(join(args.outdir, "debug_evs_rate_raw.npy"), evs.astype(np.float32))

    print(f"[DONE] Saved to: {args.outdir}")


if __name__ == "__main__":
    main()
