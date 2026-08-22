import numpy as np
from absl.logging import info

from hesim.meta import MASK_B, MASK_G, MASK_R


def apply_awb_16(raw, gains16):
    """
    Apply 16-channel AWB gains to RAW image.

    Args:
        raw: (H,W) numpy float32 RAW image.
        gains16: (16,) gain vector.

    Returns:
        corrected RAW image.
    """
    raw_corr = raw.copy()
    h, w = raw.shape
    for i in range(16):
        dx, dy = i // 4, i % 4
        raw_corr[dx::4, dy::4] *= 1.0 / gains16[i]
    return raw_corr


def apply_awb_rgb(raw, gains_rgb):
    """
    GGRR  1100
    GGRR  1100
    BBGG  3322
    BBGG  3322
    Apply RGB gains to standard Quad Bayer layout.

    Args:
        raw: (H,W) float32 image.
        gains_rgb: (3,) [R,G,B] gains.

    Returns:
        corrected raw.
    """
    if raw.ndim != 2:
        raise ValueError("raw must be 2-D (H, W)")

    g_R, g_G, g_B = map(float, gains_rgb)
    raw_corr = raw.astype(np.float32, copy=True)

    raw_corr[MASK_R] *= 1.0 / g_R
    raw_corr[MASK_B] *= 1.0 / g_B
    raw_corr[MASK_G] *= 1.0 / g_G

    return raw_corr


def apply_awb_on_rgb3channel(img_rgb, gains_rgb, clip=True):
    """
    Apply RGB gains to a demosaiced image.

    Parameters
    ----------
    img_rgb   : (H, W, 3) float32
    gains_rgb : (3,)       [g_R, g_G(=1), g_B]
    clip      : bool       whether to clip to input min/max

    Returns
    -------
    corrected : same dtype / shape as input
    """
    g_R, g_G, g_B = map(float, gains_rgb)

    out = img_rgb.astype(np.float32, copy=True)
    out[:, :, 0] /= g_R
    out[:, :, 1] /= g_G
    out[:, :, 2] /= g_B

    if clip:
        lo, hi = img_rgb.min(), img_rgb.max()
        np.clip(out, lo, hi, out)

    return out.astype(img_rgb.dtype)
