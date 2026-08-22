from __future__ import annotations

import os
from os.path import basename, join

import matplotlib.pyplot as plt
import numpy as np
from absl.logging import info
from numpy.linalg import lstsq
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde


def plot_count_eveny_frame(stk, testdata):
    N, H, W = stk.shape

    pos_cnt = (stk > 0).reshape(N, -1).sum(1)
    neg_cnt = (stk < 0).reshape(N, -1).sum(1)

    bins = 100
    range_max = int(np.percentile(np.concatenate([pos_cnt, neg_cnt]), 99.5))
    range_all = (0, range_max)

    pos_hist, pos_bins = np.histogram(pos_cnt, bins=bins, range=range_all, density=True)
    neg_hist, neg_bins = np.histogram(neg_cnt, bins=bins, range=range_all, density=True)
    bin_centers = (pos_bins[:-1] + pos_bins[1:]) / 2

    pos_hist = gaussian_filter1d(pos_hist, sigma=2)
    neg_hist = gaussian_filter1d(neg_hist, sigma=2)
    plt.figure(figsize=(4, 4))
    plt.plot(bin_centers, pos_hist, label="Positive Events per Frame", color="tab:red", linewidth=1.2)
    plt.plot(bin_centers, neg_hist, label="Negative Events per Frame", color="tab:blue", linewidth=1.2)

    plt.xlabel("Number of Events per Frame", fontsize=11)
    plt.ylabel("Probability Density", fontsize=11)
    plt.title("Distribution of Per-Frame Event Counts", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(testdata, "event_count_distribution_curve.png"), dpi=500)
    plt.show()
    plt.close()


def plot_frame_event_positive_negative_counts(stk: np.ndarray, out_dir: str, prefix: str = "dark") -> None:
    N, H, W = stk.shape
    pos_cnt = (stk > 0).reshape(stk.shape[0], -1).sum(1)
    neg_cnt = (stk < 0).reshape(stk.shape[0], -1).sum(1)

    plt.figure(figsize=(4, 4))
    bins = max(250, int(len(pos_cnt) / 25))

    plt.hist(pos_cnt, bins=bins, alpha=0.7, color="#d62728", label="Positive events")
    plt.hist(neg_cnt, bins=bins, alpha=0.7, color="#1f77b4", label="Negative events")
    plt.xlabel("Number of events per frame", fontweight="medium")
    plt.ylabel("Frequency", fontweight="medium")
    plt.title("Event Distribution Histogram")
    plt.legend(loc="upper right", frameon=True, framealpha=0.9)
    plt.grid(axis="y", alpha=0.6)
    plt.tight_layout(pad=2.0)
    plt.savefig(join(out_dir, f"{prefix}_frame_event_hist.png"), dpi=800)
    plt.close()

    plt.figure(figsize=(4, 4))
    plt.scatter(pos_cnt, neg_cnt, s=6, alpha=0.4)

    pos_mean = pos_cnt.mean()
    neg_mean = neg_cnt.mean()
    plt.scatter(pos_mean, neg_mean, s=20, c="red", marker="x", label="mean point")
    lim = max(pos_cnt.max(), neg_cnt.max()) * 1.05
    plt.plot([0, lim], [0, lim], "k--", lw=1)
    plt.xlabel("# pos events / Frame")
    plt.ylabel("# neg events / Frame")
    plt.title("EventFPS-wise +/– events")
    plt.tight_layout()
    plt.savefig(join(out_dir, f"{prefix}_pos_vs_neg_scatter.png"), dpi=800)
    plt.close()

    return (pos_mean, neg_mean)


def positive_vs_negative_rate_scatter(
    positive_rate: np.ndarray,
    negative_rate: np.ndarray,
    out_path: str,
    max_points,
    all_positive_rate,
    all_negative_rate,
    all_event_rate,
) -> None:

    noise_mask = (positive_rate > 0) | (negative_rate > 0)
    pos_noise = positive_rate[noise_mask]
    neg_noise = negative_rate[noise_mask]

    total_points = pos_noise.shape[0]
    selected_indices = np.random.choice(total_points, size=min(max_points, total_points), replace=False)
    pos_sel = pos_noise[selected_indices]
    neg_sel = neg_noise[selected_indices]

    equal_mask = np.isclose(neg_sel, pos_sel, atol=1e-6)
    neg_gt_pos = ((neg_sel > pos_sel) & (~equal_mask)).mean() * 100.0
    neg_eq_pos = equal_mask.mean() * 100.0
    neg_lt_pos = ((neg_sel < pos_sel) & (~equal_mask)).mean() * 100.0

    plt.figure(figsize=(4, 4))
    plt.scatter(pos_sel[neg_sel > pos_sel], neg_sel[neg_sel > pos_sel], s=2, color="red", alpha=0.5, label="Neg > Pos")
    plt.scatter(
        pos_sel[neg_sel < pos_sel], neg_sel[neg_sel < pos_sel], s=2, color="green", alpha=0.5, label="Pos > Neg"
    )
    plt.plot([-0.05, 0.25], [-0.05, 0.25], color="blue", linestyle="--", linewidth=1.2, label="y = x")
    plt.legend(loc="upper right", fontsize=8)

    USE_contour = True
    if USE_contour:

        points = np.stack([pos_sel, neg_sel], axis=1).T
        info(f"points:{points.shape}")
        max_kde_points = 50000
        if points.shape[1] > max_kde_points:
            sampled_idx = np.random.choice(points.shape[1], size=max_kde_points, replace=False)
            points = points[:, sampled_idx]
        kde = gaussian_kde(points, bw_method="scott")
        nbins = 100
        xgrid = np.linspace(0, 0.3, nbins)
        ygrid = np.linspace(0, 0.3, nbins)
        X, Y = np.meshgrid(xgrid, ygrid)
        grid_coords = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(grid_coords).reshape(nbins, nbins)

        Z_flat = Z.flatten()
        idx = np.argsort(Z_flat)[::-1]
        Z_cumsum = np.cumsum(Z_flat[idx])
        Z_cumsum /= Z_cumsum[-1]

        levels = []
        used_cutoffs = set()
        for p in [0.9, 0.95, 0.98, 0.99]:
            cutoff = Z_flat[idx][np.searchsorted(Z_cumsum, p)]
            info(f"cutoff:{cutoff}")
            if not np.isnan(cutoff) and cutoff not in used_cutoffs:
                levels.append(cutoff)
                used_cutoffs.add(cutoff)
        levels = sorted(set(levels))
        contour = plt.contour(
            X, Y, Z, levels=levels, colors=["#cccccc", "#999999", "#555555", "#000000"], linewidths=1.2, alpha=0.8
        )
        fmt = {l: f"{int(p*100)}%" for l, p in zip(contour.levels, [0.9, 0.95, 0.98, 0.99])}
        plt.clabel(contour, inline=True, fontsize=8, fmt=fmt)

    plt.xlabel("Positive Event Rate")
    plt.ylabel("Negative Event Rate")
    plt.title("Pixel-wise Event Rate Distribution")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.axis("equal")
    plt.plot([-0.05, 0.25], [-0.05, 0.25], color="blue", linestyle="--", linewidth=1.2, label="y = x")
    clip_x = 0.15
    clip_y = 0.15

    text_str = (
        f"Noise: ({all_event_rate*100:.6e}%)\n"
        f"--P(Pos): {all_positive_rate/all_event_rate * 100:.6e}%\n"
        f"--P(Neg): {all_negative_rate/all_event_rate * 100:.6e}%\n"
    )
    plt.text(clip_x, clip_y, text_str, fontsize=9, color="blue", bbox=dict(facecolor="white", alpha=0.7))

    plt.xlim(0.0000001, 0.3)
    plt.ylim(0.0000001, 0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "positive_vs_negative_rate_scatter.png"), dpi=800)
    plt.show()
    plt.close()
