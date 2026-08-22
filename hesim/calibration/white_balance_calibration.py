import cv2
import numpy as np
from absl.logging import debug, info

from hesim.calibration.color_checker import (
    GREEN16,
    _qb16_map,
    extract_qb16_means,
    extract_rgb_from_quad,
    extract_rgb_from_rgb,
)


def calibrate_wb_quad_bayer_with_16position(raw, blocks, gray_idx=(0, 4, 8, 12, 16, 20)):
    """
    Auto white balance (AWB) calibration for Quad-Bayer RAW (16-channel).
    Uses gray patches to normalize 16-channel gains to green reference.

    Args:
        raw: (H,W) numpy float32 RAW image.
        blocks: list of 24 (4,2) ROIs (row,col) from color checker.
        gray_idx: indices of grayscale patches (default: vertical left column).

    Returns:
        gains16: ndarray (16,), gain for each pixel type [P0-P15].
    """
    idx_map = _qb16_map(*raw.shape)
    means = [extract_qb16_means(raw, blocks[k], idx_map) for k in gray_idx]
    means = np.stack(means, axis=0)
    mean16 = np.nanmean(means, axis=0)
    g_ref = np.nanmean(mean16[GREEN16])
    gains16 = g_ref / mean16
    gains16 = np.clip(gains16, 0.01, 16.0)
    return gains16


def calibrate_wb_quad_bayer_with_RGB(raw, blocks, gray_idx=(4,)):
    """
    Simpler AWB for Quad-Bayer RAW, estimating per-channel RGB gains.

    Returns:
        gains: np.ndarray (3,) in R, G, B order.
    """
    rgbs = []
    for idx in gray_idx:
        R, G1, B, G2 = extract_rgb_from_quad(raw, blocks[idx])
        G = np.hstack((G1, G2))
        info(f"       Gains[{idx}]: R={R.mean()/G.mean():.2f}, G=1.00, B={B.mean()/G.mean():.2f}")
        if R.size and G.size and B.size:
            rgbs.append([R.mean(), G.mean(), B.mean()])
        else:
            rgbs.append([np.nan, np.nan, np.nan])
    rgbs = np.asarray(rgbs)
    mean_R, mean_G, mean_B = np.nanmean(rgbs, axis=0)
    gains = np.array([mean_R, mean_G, mean_B], dtype=np.float32)
    gains /= mean_G
    return gains


def calibrate_wb_quad_bayer_on_rgb3channel(
    img_rgb,
    blocks,
    gray_idx=(0, 4, 8, 12, 16, 20),
):
    """
    Compute per-image RGB gains **after demosaic**.

    The gain for each grey patch k is:
        g_R_k = R̅_k / G̅_k
        g_B_k = B̅_k / G̅_k

    The final gains are the (nan-robust) average of the individual
    grey-patch gains.  Green is the reference and normalised to 1.

    Parameters
    ----------
    img_rgb : ndarray (H, W, 3)  float32
    blocks  : list/ndarray (24, 4, 2) colour-checker ROIs  (row, col)
    gray_idx: tuple  indices of grey patches (default left column)

    Returns
    -------
    gains_rgb : ndarray (3,)   [R_gain, G_gain(==1), B_gain]
    """
    gains_R, gains_B = [], []

    for idx in gray_idx:
        R, G, B = extract_rgb_from_rgb(img_rgb, blocks[idx])
        if R.size and G.size and B.size:
            g_R = np.nanmean(R) / np.nanmean(G)
            g_B = np.nanmean(B) / np.nanmean(G)
            debug(f"  Patch {idx:2d}:  R/G={g_R:.3f}  B/G={g_B:.3f}")
            gains_R.append(g_R)
            gains_B.append(g_B)

    g_R_final = np.nanmean(gains_R)
    g_B_final = np.nanmean(gains_B)

    gains_rgb = np.array([g_R_final, 1.0, g_B_final], dtype=np.float32)
    return gains_rgb
