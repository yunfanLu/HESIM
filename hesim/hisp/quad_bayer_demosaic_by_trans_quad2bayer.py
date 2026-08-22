import cv2
import numpy as np

__all__ = ["quad_bayer_demosaic_by_trans_quad2bayer"]


def _bin_2x2_cluster(raw: np.ndarray) -> np.ndarray:
    """
    Mean-pool the Quad-Bayer RAW to Bayer resolution (H/2, W/2).
    The four pixels that share the same colour are averaged; we keep
    **one sample per colour** exactly like a normal Bayer sensor.
    """
    H, W = raw.shape
    assert H % 2 == 0 and W % 2 == 0, "image size must be even"

    return raw.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3)).astype(np.float32)


def _pixel_shuffle_quad2bayer(raw):
    h, w = raw.shape
    assert h % 2 == 0 and w % 2 == 0, "shape must be even"

    out = np.empty_like(raw)

    out[0::4] = raw[0::4]
    out[1::4] = raw[2::4]
    out[2::4] = raw[1::4]
    out[3::4] = raw[3::4]

    bayer = np.empty_like(out)
    bayer[:, 0::4] = out[:, 0::4]
    bayer[:, 1::4] = out[:, 2::4]
    bayer[:, 2::4] = out[:, 1::4]
    bayer[:, 3::4] = out[:, 3::4]
    return bayer


def quad_bayer_demosaic_by_trans_quad2bayer(raw4, trans_quad2bayer_type="binning", algorithm="Malvar"):
    """
    Convert Quad-Bayer RAW to RGB using two-pass strategy:
      1. 2×2 averaging → Bayer CFA
      2. Colour-Science demosaic
      3. Simple 2× up-scale (nearest-neighbour) back to original size.
    """

    if trans_quad2bayer_type == "binning":
        bayer = _bin_2x2_cluster(raw4.astype(np.float32))
    elif trans_quad2bayer_type == "pixel_shuffle":
        bayer = _pixel_shuffle_quad2bayer(raw4.astype(np.float32))
    else:
        raise ValueError(f"Unknown conversion: {trans_quad2bayer_type}")

    from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004, demosaicing_CFA_Bayer_Menon2007

    if algorithm.lower() == "malvar":
        rgb = demosaicing_CFA_Bayer_Malvar2004(bayer, pattern="GRBG")
    else:
        rgb = demosaicing_CFA_Bayer_Menon2007(bayer, pattern="GRBG")

    if trans_quad2bayer_type == "binning":
        rgb = cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC).astype(np.float32)

    return rgb
