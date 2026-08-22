from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class AutoColorCorrectionConfig:

    clip_percent: float = 1.0
    target_mean: float = 0.5
    target_std: float = 0.25
    min_std: float = 1e-3


class ColorCorrectionFactory:

    def __init__(self, method: Literal["retinex", "robust"] = "robust"):
        self.method = method
        self._config = AutoColorCorrectionConfig()

    def auto_ccm(self, rgb: np.ndarray) -> np.ndarray:
        if self.method == "retinex":
            return self._retinex_enhance(rgb)
        elif self.method == "robust":
            return self._auto_color_correction(rgb, self._config)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _retinex_enhance(self, img: np.ndarray, sigma_list=[15, 80, 250]) -> np.ndarray:
        img_float = img.astype(np.float32) + 1.0
        retinex = np.zeros_like(img_float)
        for sigma in sigma_list:
            blurred = cv2.GaussianBlur(img_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
            retinex += np.log10(img_float) - np.log10(blurred + 1.0)
        retinex /= len(sigma_list)
        r_min, r_max = retinex.min(), retinex.max()
        if r_max > r_min:
            normalized = (retinex - r_min) / (r_max - r_min) * 255.0
        else:
            normalized = np.zeros_like(retinex)
        out = np.clip(normalized, 0, 255).astype(np.float32)
        return out / 255.0

    def _auto_color_correction(
        self,
        rgb: np.ndarray,
        config: Optional[AutoColorCorrectionConfig] = None,
    ) -> np.ndarray:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("'rgb' must be an array of shape (H, W, 3)")

        if config is None:
            config = AutoColorCorrectionConfig()

        rgb = np.asarray(rgb, np.float32)
        corrected = np.empty_like(rgb, dtype=np.float32)

        for c in range(3):
            channel = rgb[..., c]
            stretched = self._robust_normalise(channel, config.clip_percent)
            corrected[..., c] = self._match_moments(stretched, config.target_mean, config.target_std, config.min_std)

        return np.clip(corrected, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _robust_normalise(channel: np.ndarray, clip_percent: float) -> np.ndarray:
        if clip_percent <= 0.0:
            cmin, cmax = channel.min(), channel.max()
        else:
            lower, upper = np.percentile(channel, [clip_percent, 100.0 - clip_percent])
            cmin, cmax = float(lower), float(upper)

        if cmax - cmin < 1e-8:
            return np.clip(channel - cmin, 0.0, 1.0)
        stretched = (channel - cmin) / (cmax - cmin)
        return np.clip(stretched, 0.0, 1.0)

    @staticmethod
    def _match_moments(channel: np.ndarray, target_mean: float, target_std: float, min_std: float) -> np.ndarray:
        mean = channel.mean()
        std = channel.std()
        gain = target_std / max(std, min_std)
        bias = target_mean - gain * mean
        return gain * channel + bias
