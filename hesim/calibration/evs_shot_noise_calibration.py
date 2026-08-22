from __future__ import annotations

import csv
import glob
import json
import os
import random
from os.path import basename, join
from typing import Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from absl.logging import info
from numpy.linalg import lstsq
from tqdm import tqdm

from hesim.calibration.color_checker import get_checker_blocks_homography, load_checker_json
from hesim.calibration.evs_noise_visualization import plot_frame_event_positive_negative_counts
from hesim.io import evs_raw_stack, find_evs_folder, read_evs_raw, sorted_raws


def _noise_regression_by_color_checker_2(stk, out_dir, roi_masks):

    rate_light = np.mean(np.abs(stk), 0)
    pos_light = np.mean(stk > 0, 0)
    neg_light = np.mean(stk < 0, 0)

    GRAY_IDX = [0, 4, 8, 12, 16, 20]
    L_GRAY = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])

    rate_gray_tot = np.array([rate_light[roi_masks[m]].mean() for m in GRAY_IDX])
    rate_gray_pos = np.array([pos_light[roi_masks[m]].mean() for m in GRAY_IDX])
    rate_gray_neg = np.array([neg_light[roi_masks[m]].mean() for m in GRAY_IDX])

    for idx in range(len(GRAY_IDX)):
        info(f"Gray-patch[{idx}](total, pos, neg): {rate_gray_tot[idx], rate_gray_pos[idx], rate_gray_neg[idx]}")

    A = np.vstack([L_GRAY, np.ones_like(L_GRAY)]).T
    k_shot, r_dark = lstsq(A, rate_gray_tot, rcond=None)[0]
    info("Gray-patch fit  k=%.4f  r₀=%.4f", k_shot, r_dark)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(L_GRAY, rate_gray_tot, c="C0", label="total")
    ax.scatter(L_GRAY, rate_gray_pos, c="C2", marker="^", label="pos")
    ax.scatter(L_GRAY, rate_gray_neg, c="C3", marker="v", label="neg")
    x_ = np.linspace(0, 1, 100)
    ax.plot(x_, k_shot * x_ + r_dark, "k--", lw=1.2, label=f"fit k={k_shot:.3f}")
    ax.set_xlabel("Relative luminance (gray row)")
    ax.set_ylabel("event rate  (Hz / pix)")
    ax.set_title("Gray-patch event-rate vs luminance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(join(out_dir, "gray_patch_rate_fit.png"), dpi=900)
    plt.close(fig)

    np.savez_compressed(
        join(out_dir, "evs_gray_fit.npz"),
        k_shot=k_shot,
        r_dark=r_dark,
        rate_gray_tot=rate_gray_tot,
        rate_gray_pos=rate_gray_pos,
        rate_gray_neg=rate_gray_neg,
    )


def calibrate_evs_shot_noise_by_color_checker(stk, chart_json, out_dir) -> None:

    H, W = 1632, 1224
    pts = load_checker_json(chart_json)
    blocks, _ = get_checker_blocks_homography(pts, (6, 4), roi_scale=0.4)
    roi_masks = []
    for quad in blocks:

        quad = quad / 2
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, quad[:, ::-1].astype(np.int32), 1)
        roi_masks.append(m.astype(bool))
        info(f"quad({quad}): {np.sum(m)}")

    plot_frame_event_positive_negative_counts(stk, out_dir, prefix=f"light")
    _noise_regression_by_color_checker_2(stk, out_dir, roi_masks)


def calibrate_evs_shot_noise_pixel_level(evs_stk, pixel_intensity):
    pass
