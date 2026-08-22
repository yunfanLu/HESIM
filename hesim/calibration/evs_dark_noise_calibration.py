from typing import Optional

import numpy as np


def detect_hot_pixels(dark_event: np.ndarray, rate_threshold: float = 0.01) -> np.ndarray:

    ev_rate = np.abs(dark_event).astype(float)

    return ev_rate > rate_threshold


def dark_event_activate_rate(rate_dark, top_pct: float = 0.05, rate_thr: Optional[float] = None):

    hot_pix = rate_dark > 0.01

    rate_sorted = np.sort(rate_dark.ravel())[::-1]
    P = rate_sorted.size
    prop = np.arange(P) / P
    x_idx = np.arange(1, rate_sorted.size + 1)

    if rate_thr is not None:
        keep = rate_sorted > rate_thr
    else:
        k = max(1, int(np.ceil(top_pct * P)))
        keep = np.zeros(P, bool)
        keep[:k] = True
    rate_keep = rate_sorted[keep]
    prop_keep = prop[keep]
    return hot_pix, rate_keep, prop_keep
