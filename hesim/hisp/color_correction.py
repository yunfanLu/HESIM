from functools import lru_cache

import numpy as np
from numpy.linalg import lstsq

from hesim.hisp.color_checker import SRGB_REF, get_checker_blocks_homography
from hesim.hisp.white_balance_calibration import extract_rgb_from_rgb


def patch_means_rgb(rgb, blocks):
    """Return (24,3) mean RGB (linear) from the 24 quadrilateral ROIs."""
    means = []
    for quad in blocks:
        R, G, B = extract_rgb_from_rgb(rgb, quad)
        means.append([R.mean(), G.mean(), B.mean()])
    return np.asarray(means, np.float32)


def linearise_srgb(srgb):
    """sRGB (0-255 or 0-1) → linear RGB 0-1 (IEC 61966-2-1)."""
    srgb = np.asarray(srgb, np.float32) / 255.0 if srgb.max() > 1 else srgb
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return lin.astype(np.float32)


def compute_ccm(meas_rgb24, ref_rgb24, with_bias=True):
    """
    Return a 3×3 (or 3×4 with bias) colour-correction matrix M s.t.

        ref ≈ M · meas_lin     (row-vector convention)

    Parameters
    ----------
    meas_rgb24 : (24,3)  measured patch means AFTER WB & demosaic (linear 0-1)
    ref_rgb24  : (24,3)  reference patch colours (linear 0-1)
    with_bias  : bool    add a constant column (recommended)

    Returns
    -------
    M : (3,3) or (3,4)  float32
    """
    meas = np.asarray(meas_rgb24, np.float32)
    ref = np.asarray(ref_rgb24, np.float32)
    if with_bias:
        meas = np.hstack([meas, np.ones((meas.shape[0], 1), np.float32)])

    M_t, *_ = lstsq(meas, ref, rcond=None)
    return M_t.T.astype(np.float32)


def apply_ccm(img_rgb, M):
    """
    Apply CCM to an RGB image (linear domain).

    img_rgb : (H,W,3) float32  in 0-1 (**linear**)
    M       : (3,3) or (3,4)

    Returns
    -------
    img_corr : same shape & dtype as input, clipped to 0-1
    """
    h, w, _ = img_rgb.shape
    flat = img_rgb.reshape(-1, 3)

    if M.shape[1] == 4:
        flat = np.hstack([flat, np.ones((flat.shape[0], 1), flat.dtype)])

    out = flat @ M.T
    out = np.clip(out, 0.0, 1.0)
    return out.reshape(h, w, 3)


def color_currection_by_color_checker(rgb_wb, vertex_pts):
    blocks, _ = get_checker_blocks_homography(vertex_pts, (6, 4), 0.4)
    meas = patch_means_rgb(rgb_wb, blocks)
    ref = linearise_srgb(SRGB_REF.reshape(-1, 3))
    M = compute_ccm(meas, ref, with_bias=True)
    rgb_cc = apply_ccm(rgb_wb, M)
    rgb_cc = np.clip(rgb_cc, 0.0, 1.0)
    return rgb_cc, M
