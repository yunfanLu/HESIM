"""
Generic RAW-domain denoiser.

Typical call inside the pipeline
--------------------------------
from hisp.denoise import get_denoiser

The function is intentionally *stateless* so it can be created once in
setUp() and reused for every frame.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Literal, Optional

import cv2
import numpy as np
import torch
import torchvision as tv
from bm3d import BM3DProfile, bm3d, bm3d_rgb

Method = Literal["bm3d", "nlm", "none"]


def denoise(img, method: Method = "bm3d", **kwargs):
    """
    Parameters
    ----------
    method : "bm3d" | "nlm" | "dncnn" | "none"
        - **bm3d**  : best quality, slow, single-channel.  Needs   `pip install bm3d`.
        - **nlm**   : OpenCV fast non-local means (works on uint8).
        - **dncnn** : PyTorch DnCNN (pre-trained σ≈25/255).  Needs GPU / CPU PyTorch.
        - **none**  : identity (useful for ablation).
    **kwargs      : algorithm-specific parameters,
        bm3d : `sigma`  (default 0.01 in [0,1]),  `profile`
        nlm  : `h`, `templateWindowSize`, `searchWindowSize`
        dncnn: `device`  ("cuda"/"cpu")
    """
    method = method.lower()

    if method == "none":
        return img
    elif method == "bm3d":
        sigma = float(kwargs.get("sigma", 0.01))
        profile = BM3DProfile()
        sigma = max(1e-4, min(sigma, 0.2))
        if img.ndim == 2:
            return bm3d(img, sigma, profile=profile)
        elif img.ndim == 3 and img.shape[2] == 3:
            return bm3d_rgb(img, sigma, profile=profile)
        else:
            raise ValueError("BM3D expects (H,W) or (H,W,3) array.")

    elif method == "nlm":
        h_val = kwargs.get("h", 10.0)
        h_color = kwargs.get("hColor", h_val)
        tsize = kwargs.get("templateWindowSize", 7)
        wsize = kwargs.get("searchWindowSize", 21)

        img8 = np.clip(img, 0, 1) * 255.0
        img8 = img8.astype(np.uint8)
        if img.ndim == 2:
            out8 = cv2.fastNlMeansDenoising(img8, None, h=h_val, templateWindowSize=tsize, searchWindowSize=wsize)
        else:
            out8 = cv2.fastNlMeansDenoisingColored(
                img8, None, h=h_color, hColor=h_color, templateWindowSize=tsize, searchWindowSize=wsize
            )
        return out8.astype(np.float32) / 255.0
    raise ValueError(f"Unknown denoise method '{method}'")
