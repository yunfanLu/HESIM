import json
import os
from glob import glob
from os.path import isdir, join

import matplotlib.pyplot as plt
import numpy as np
from absl.logging import info
from tqdm import tqdm

from hesim.io import aps_raw_stack, read_raw
from hesim.visualization import visualize_matrix_with_histogram

__all__ = ["calibrate_black_level_with_multiple_exposures"]


DEBUG = True


def calibrate_black_level_with_multiple_exposures(exp_dirs, out_dir):
    T, stk = _load_stack_by_exposure(exp_dirs)

    a_map, b_map = _fit_linear_per_pixel(T, stk)
    info(f"Slope DN/ms : min={a_map.min():.4f},  median={np.median(a_map):.4f},  max={a_map.max():.4f}")
    info(f"Offset DN   : min={b_map.min():.2f},  median={np.median(b_map):.2f}, max={b_map.max():.2f}")
    _save_maps(a_map, b_map, out_dir)
    info(f"Maps saved to {out_dir}")
    _evaluate_fit_error(T, stk, a_map, b_map, out_dir)
    return a_map, b_map, T, stk


def fixed_noise_split_to_row_noise_and_black_level_noise(b):
    row = np.mean(b, axis=1)
    blc = b - row[:, None]
    return row, blc


def calibrate_black_level(folder, vis_dir=None, verbose=True):
    """
    Calibrates black level (fixed pattern noise) for Quad Bayer RAW sensor data.
    Performs spatial, temporal, row/column, and block statistics and visualization.
    Args:
        folder: Directory containing multiple dark frames (.raw files).
        save_vis: Save visualizations to disk.
        verbose: info summary statistics.
    Returns:
        result_dict: Dictionary containing all key mean/std/statistics/visualization data.
    """

    quad_block_shape = (4, 4)

    files = sorted(glob(join(folder, "*.raw")), key=lambda x: int(x.split("_")[-3]))
    assert len(files) > 0, f"No raw files found in {folder}."

    raws = []
    for f in tqdm(files, desc=f"Reading RAWs ({len(files)}) from {folder}"):
        raw = read_raw(f).astype(np.float32)
        raws.append(raw)
    raws = np.stack(raws, axis=0)
    N, H, W = raws.shape
    info(f"Loaded {N} frames of size {H}x{W} from {folder}")
    info(f"  Raw in: {raws.dtype}, {raws.min()}-{raws.max()}. Mean: {raws.mean():.2f}, Std: {raws.std():.2f}")

    mean_map = np.mean(raws, axis=0)
    std_map = np.std(raws, axis=0)

    visualize_matrix_with_histogram(mean_map, filename=join(vis_dir, "mean_map-hist.png"))

    block_h, block_w = quad_block_shape
    block_stats = {}
    block_means = []
    block_stds = []
    for iy in range(block_h):
        for ix in range(block_w):
            block = mean_map[iy::block_h, ix::block_w]
            mean = block.mean()
            std = block.std()
            block_stats[f"Block({iy},{ix})"] = {"mean": float(mean), "std": float(std)}
            block_means.append(mean)
            block_stds.append(std)
            if verbose:
                info(f"Quad block ({iy},{ix}): mean={mean:.8f}, std={std:.8f}")

    row_mean = mean_map.mean(axis=1)
    col_mean = mean_map.mean(axis=0)
    row_std = mean_map.std(axis=1)
    col_std = mean_map.std(axis=0)
    if verbose:
        info(f"Row mean: {row_mean.max():.2f}-{row_mean.min():.2f}")
        info(f"Col mean: {col_mean.max():.2f}-{col_mean.min():.2f}")

    frame_mean = raws.reshape(N, -1).mean(axis=1)
    frame_std = raws.reshape(N, -1).std(axis=1)
    drift = frame_mean.max() - frame_mean.min()
    if verbose:
        info(f"Max temporal frame mean drift: {drift:.3f} DN")

    if isdir(vis_dir):
        outdir = vis_dir

        plt.figure(figsize=(4, 4))
        plt.imshow(mean_map, cmap="hot")
        plt.colorbar()
        plt.title("Black Level Mean Map")
        plt.savefig(join(outdir, "blc_mean_map.png"), dpi=800)
        plt.close()
        plt.figure(figsize=(4, 4))
        plt.imshow(mean_map, cmap="hot", vmin=0, vmax=0.1)
        plt.colorbar()
        plt.title(f"Black Level Mean Map \n {mean_map.min():.4f} - {mean_map.max():.4f}")
        plt.savefig(join(outdir, "blc_mean_map-fix-0-0_1.png"), dpi=800)
        plt.close()

        plt.figure(figsize=(4, 3))
        plt.imshow(std_map, cmap="hot")
        plt.colorbar()
        plt.title("Black Level Std Map")
        plt.savefig(join(outdir, "blc_std_map.png"), dpi=800)
        plt.close()

        plt.figure(figsize=(4, 3))
        plt.plot(row_mean)
        plt.title("Row Mean")
        plt.xlabel("Row")
        plt.ylabel("Mean DN")
        plt.savefig(join(outdir, "row_mean.png"), dpi=800)
        plt.close()
        plt.figure(figsize=(4, 3))
        plt.plot(col_mean)
        plt.title("Column Mean")
        plt.xlabel("Column")
        plt.ylabel("Mean DN")
        plt.savefig(join(outdir, "col_mean.png"), dpi=800)
        plt.close()

        plt.figure()
        plt.plot(row_std)
        plt.title("Row Std")
        plt.xlabel("Row")
        plt.ylabel("Std DN")
        plt.savefig(join(outdir, "row_std.png"))
        plt.close()
        plt.figure()
        plt.plot(col_std)
        plt.title("Column Std")
        plt.xlabel("Column")
        plt.ylabel("Std DN")
        plt.savefig(join(outdir, "col_std.png"))
        plt.close()

        plt.figure()
        plt.plot(frame_mean, label="Mean")
        plt.plot(frame_std, label="Std")
        plt.title("Frame Mean & Std Over Time")
        plt.xlabel("Frame Index")
        plt.ylabel("DN")
        plt.legend()
        plt.savefig(join(outdir, "frame_seq_mean_std.png"))
        plt.close()

        plt.figure(figsize=(4, 3))
        plt.hist(mean_map.ravel(), bins=255, color="b", alpha=0.7)
        plt.title(f"Black Level Mean Histogram, [{mean_map.min():.2f},{mean_map.max():.2f}]")
        plt.savefig(join(outdir, "blc_mean_hist.png"), dpi=800)
        plt.close()
        plt.figure(figsize=(4, 3))
        plt.hist(std_map.ravel(), bins=255, color="r", alpha=0.7)
        plt.title("Black Level Std Histogram")
        plt.savefig(join(outdir, "blc_std_hist.png"), dpi=800)
        plt.close()

    result = {
        "global_mean": float(mean_map.mean()),
        "global_std": float(mean_map.std()),
        "mean_map": mean_map,
        "std_map": std_map,
        "block_stats": block_stats,
        "block_means": np.array(block_means),
        "block_stds": np.array(block_stds),
        "row_mean": row_mean,
        "col_mean": col_mean,
        "row_std": row_std,
        "col_std": col_std,
        "frame_mean": frame_mean,
        "frame_std": frame_std,
        "frame_drift": drift,
        "file_list": files,
    }
    return result


def _evaluate_fit_error(T, stk, a_map, b_map, out_dir):
    """
    Computes pixel-wise and global fitting errors between measured and predicted dark frames.

    Args:
        T       : (E,) exposure times
        stk     : (E, H, W) dark frame stack
        a_map   : (H, W) slope per pixel
        b_map   : (H, W) offset per pixel
        out_dir : path to save visualizations and error metadata

    Returns:
        error_stats : dict containing MAE, RMSE per exposure and overall
    """
    E, H, W = stk.shape

    pred = a_map[None, :, :] * T[:, None, None] + b_map[None, :, :]
    residuals = pred - stk

    mae_per_T = np.mean(np.abs(residuals), axis=(1, 2))
    rmse_per_T = np.sqrt(np.mean(residuals**2, axis=(1, 2)))

    os.makedirs(out_dir, exist_ok=True)
    for i, t in enumerate(T):

        visualize_matrix_with_histogram(
            residuals[i],
            f"Residual Map @ {t:.0f}ms (Pred - Actual)",
            filename=join(out_dir, f"residual_map_{int(t)}ms.png"),
        )

    error_stats = {
        "exposures_ms": T.tolist(),
        "mae_per_exposure": mae_per_T.tolist(),
        "rmse_per_exposure": rmse_per_T.tolist(),
        "mae_global": float(np.mean(mae_per_T)),
        "rmse_global": float(np.sqrt(np.mean(residuals**2))),
        "slope_stats": {
            "min": float(a_map.min()),
            "median": float(np.median(a_map)),
            "max": float(a_map.max()),
            "mean": float(a_map.mean()),
            "98%ile": float(np.percentile(a_map, 98)),
            "2%ile": float(np.percentile(a_map, 2)),
            "abs>0.5_count": int(np.sum(np.abs(a_map) > 0.5)),
            "abs>1.0_count": int(np.sum(np.abs(a_map) > 1.0)),
        },
        "offset_stats": {
            "min": float(b_map.min()),
            "median": float(np.median(b_map)),
            "max": float(b_map.max()),
            "mean": float(b_map.mean()),
            "98%ile": float(np.percentile(b_map, 98)),
            "2%ile": float(np.percentile(b_map, 2)),
            "abs>18_count": int(np.sum(np.abs(b_map) > 18)),
            "abs>22_count": int(np.sum(np.abs(b_map) > 22)),
        },
    }
    with open(join(out_dir, "fit_error_stats.json"), "w") as f:
        json.dump(error_stats, f, indent=2)

    return error_stats


def _load_stack_by_exposure(exp_dirs):
    exposures = sorted([int(k) for k in exp_dirs.keys()])
    sample = read_raw(glob(join(exp_dirs[str(exposures[0])], "*.raw"))[0])
    H, W = sample.shape
    stk = np.empty((len(exposures), H, W), dtype=np.float32)

    for idx, t in enumerate(exposures):

        frames = aps_raw_stack(exp_dirs[str(t)], t, with_blc=False)

        stk[idx] = np.mean(frames, axis=0)
        info(f"  {t} ms -> DN range [{stk[idx].min():.1f}, {stk[idx].max():.1f}]")
    return np.array(exposures, dtype=np.float32), stk


def _fit_linear_per_pixel(T, stk):
    """
    Least-squares fit B = a*T + b per pixel.
    Parameters
    ----------
    T   : (E,) float32   exposures
    stk : (E,H,W) float32  mean dark frames
    Returns
    -------
    a_map : (H,W) slope   DN / ms
    b_map : (H,W) offset  DN
    """

    E, H, W = stk.shape
    T_vec = T.reshape(E, 1, 1)
    ones = np.ones_like(T_vec)

    S_TT = np.sum(T_vec * T_vec, axis=0)
    S_T1 = np.sum(T_vec, axis=0)
    S_11 = E
    det = S_TT * S_11 - S_T1**2

    S_yT = np.sum(stk * T_vec, axis=0)
    S_y1 = np.sum(stk, axis=0)

    a_map = (S_11 * S_yT - S_T1 * S_y1) / det
    b_map = (S_TT * S_y1 - S_T1 * S_yT) / det
    return a_map, b_map


def _save_maps(a_map, b_map, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.save(join(out_dir, "dark_slope_map.npy"), a_map)
    np.save(join(out_dir, "dark_offset_map.npy"), b_map)

    for name, m, cmap in [("Slope", a_map, "bwr"), ("Offset", b_map, "bwr")]:

        visualize_matrix_with_histogram(
            m,
            f"Dark-current {name} map\n {m.min():.4f} to {m.max():.4f}",
            filename=join(out_dir, f"dark_{name}_map.png"),
        )

        plt.figure(figsize=(5, 5))
        plt.hist(m.ravel(), bins=255, color="b", alpha=0.7, density=True, label="Empirical distribution")
        plt.title(f"{name} Histogram\n {m.min():.2f} to {m.max():.2f}")
        plt.xlabel(f"DN-{name}")
        plt.ylabel("Probability Density")
        plt.savefig(join(out_dir, f"dark_{name}_hist.png"), dpi=600)

    b_map_row = b_map.mean(axis=1)
    b_map_residual = b_map - b_map_row[:, np.newaxis]

    plt.figure(figsize=(5, 5))
    plt.hist(b_map_row.ravel(), bins=255, color="b", density=True, alpha=0.7)
    plt.title(f"Row Mean Distribution\nRange: {b_map_row.min():.4f} to {b_map_row.max():.4f}")
    plt.xlabel(f"DN-b_map_row")
    plt.ylabel("Probability Density")
    plt.savefig(join(out_dir, f"dark_{name}_hist-row.png"), dpi=600)

    b_map_residual_mean = np.mean(b_map_residual)
    b_map_residual_std = np.std(b_map_residual)
    info(f"b_map_residual_mean: {b_map_residual_mean:.4f}, b_map_residual_std: {b_map_residual_std:.4f}")
    plt.figure(figsize=(5, 5))
    counts, bins, _ = plt.hist(
        b_map_residual.ravel(),
        bins=255,
        density=True,
        color="steelblue",
        alpha=0.6,
        edgecolor="black",
        label="Empirical distribution",
        range=(-0.012, 0.012),
    )
    x = np.linspace(-0.012, 0.012, 500)
    gauss = np.exp(-0.5 * ((x - b_map_residual_mean) / b_map_residual_std) ** 2) / (
        b_map_residual_std * np.sqrt(2 * np.pi)
    )
    plt.plot(x, gauss, "r--", label="Fitted Gaussian")

    textstr = f"μ = {b_map_residual_mean:.8f}\nσ = {b_map_residual_std:.8f}"
    props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8)
    plt.text(
        0.5,
        0.55,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )

    plt.xlabel("BLC-corrected Residual (DN)")
    plt.ylabel("Probability Density")
    plt.title(f"BLC-corrected Residual\nRange: {b_map_residual.min():.4f} to {b_map_residual.max():.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(out_dir, f"dark_{name}_hist-residual.png"), dpi=600)
    plt.close()
