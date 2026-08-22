import numpy as np

"""
GGRR  0123
GGRR  4567
BBGG  89ab
BBGG  cdef
"""

SENSOR_H, SENSOR_W = 3264, 2448


def SENSOR_COLOR_MASK():
    H, W = SENSOR_H, SENSOR_W
    rr = np.arange(H)[:, None] % 4
    cc = np.arange(W)[None, :] % 4

    mask_R = (rr < 2) & (cc >= 2)
    mask_B = (rr >= 2) & (cc < 2)
    mask_G = ~(mask_R | mask_B)
    return mask_R, mask_G, mask_B


def to_srgb_gamma(img_lin):

    x = np.clip(img_lin, 0.0, 1.0)

    a = 0.055
    thr = 0.0031308
    lin_mask = x <= thr
    srgb = np.empty_like(x)
    srgb[lin_mask] = 12.92 * x[lin_mask]
    srgb[~lin_mask] = (1 + a) * np.power(x[~lin_mask], 1 / 2.4) - a

    return (srgb * 255 + 0.5).astype(np.uint8)


MASK_R, MASK_G, MASK_B = SENSOR_COLOR_MASK()
