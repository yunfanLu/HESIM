from __future__ import annotations

import os
from os.path import join
from typing import Dict, List, Tuple

import numpy as np

from hesim.hisp.color_checker import SRGB_REF, get_checker_blocks_homography, load_checker_json
from hesim.hisp.color_correction import apply_ccm, compute_ccm, linearise_srgb, patch_means_rgb
from hesim.hisp.white_balance import apply_awb_on_rgb3channel
from hesim.hisp.white_balance_calibration import calibrate_wb_quad_bayer_on_rgb3channel
from hesim.io import read_raw


def _patch_means_after_awb(rgb: np.ndarray, blocks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    means    : (24,3) patch means after per-image AWB (linear RGB 0-1)
    wb_rgb   : entire white-balanced image (for QA)
    """
    gains = calibrate_wb_quad_bayer_on_rgb3channel(rgb, blocks)
    rgb_wb = apply_awb_on_rgb3channel(rgb, gains)
    return patch_means_rgb(rgb_wb, blocks), rgb_wb


def _estimate_cct_from_white(rgb: np.ndarray, blocks: np.ndarray) -> float:
    """
    Estimate CCT (Kelvin) from first grey patch (1E) in *raw* linear-RGB.
    Compatible with colour-science >= 0.4.
    """

    import colour

    srgb = colour.models.RGB_COLOURSPACES["sRGB"]
    patch_rgb = patch_means_rgb(rgb, blocks)[0].astype(np.float64)
    patch_rgb /= np.clip(patch_rgb.max(), 1e-8, None)

    XYZ = colour.RGB_to_XYZ(
        patch_rgb,
        srgb.whitepoint,
        srgb.whitepoint,
        srgb.matrix_RGB_to_XYZ,
    )

    xy = colour.XYZ_to_xy(XYZ)
    cct = colour.xy_to_CCT(xy, method="hernandez1999")
    return float(cct)


def _solve_bucket_ccm(list_meas: List[np.ndarray], with_bias: bool = True) -> np.ndarray:
    X = np.vstack(list_meas)
    Y = np.tile(linearise_srgb(SRGB_REF.reshape(-1, 3)), (len(list_meas), 1))
    return compute_ccm(X, Y, with_bias=with_bias)


def calibrate_ccm_space(
    case_dirs: List[str],
    expos_ms: List[float],
    blc,
    demosaic,
    ref_CCTs: Tuple[float, float, float] = (2500.0, 5000.0, 6500.0),
    roi_scale: float = 0.4,
    with_bias: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Returns
    {"CCT_low": M_low, "CCT_mid": M_mid, "CCT_high": M_high}
      (keys are the *Kelvin* floats)
    """
    buckets: Dict[float, List[np.ndarray]] = {c: [] for c in ref_CCTs}

    for case_dir, exp in zip(case_dirs, expos_ms):

        raw_p = join(case_dir, next(f for f in os.listdir(case_dir) if f.endswith(".raw")))
        json_p = join(case_dir, next(f for f in os.listdir(case_dir) if f.endswith(".json")))

        rgb = demosaic(blc(read_raw(raw_p), exp)).astype(np.float32).clip(0, 1)
        blocks, _ = get_checker_blocks_homography(load_checker_json(json_p), (6, 4), roi_scale)

        cct = _estimate_cct_from_white(rgb, blocks)

        ref = min(ref_CCTs, key=lambda r: abs(r - cct))

        means, _ = _patch_means_after_awb(rgb, blocks)
        buckets[ref].append(means)

    ccm_space = {}
    for ref_cct, meas_list in buckets.items():
        if len(meas_list) == 0:
            continue
        ccm_space[ref_cct] = _solve_bucket_ccm(meas_list, with_bias=with_bias)

    return ccm_space


def interpolate_ccm(ccm_space: Dict[float, np.ndarray], cct_input: float) -> np.ndarray:
    """
    Linear interpolation in *mired* space (1 / CCT).
    """
    refs = sorted(ccm_space.keys())
    mirs = [1.0 / k for k in refs]
    mi = 1.0 / cct_input

    if mi <= mirs[0]:
        return ccm_space[refs[0]]
    if mi >= mirs[-1]:
        return ccm_space[refs[-1]]

    for j in range(len(refs) - 1):
        if mirs[j + 1] <= mi <= mirs[j]:
            g = (mi - mirs[j + 1]) / (mirs[j] - mirs[j + 1])
            return g * ccm_space[refs[j]] + (1 - g) * ccm_space[refs[j + 1]]

    raise RuntimeError("Interpolation logic fell through.")


def colour_correct_image(
    rgb_raw: np.ndarray,
    blocks_rc: np.ndarray,
    ccm_space: Dict[float, np.ndarray],
) -> np.ndarray:
    """
    One-shot helper – returns colour-corrected RGB image (linear).
    """

    gains = calibrate_wb_quad_bayer_on_rgb3channel(rgb_raw, blocks_rc)
    rgb_wb = apply_awb_on_rgb3channel(rgb_raw, gains)

    cct_est = _estimate_cct_from_white(rgb_raw, blocks_rc)

    M = interpolate_ccm(ccm_space, cct_est)
    rgb_corr = apply_ccm(rgb_wb, M)
    return rgb_corr
