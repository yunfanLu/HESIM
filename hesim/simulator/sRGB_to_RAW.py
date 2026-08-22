"""
sRGB -> (approx.) APS RAW degrader driven by calibration
Deps: numpy, opencv-python
"""

import json
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from absl.logging import info
from PIL import Image

"""
simulator:
  target: src.hybrid_evs.Simulator
  meta:
    ISO: 985
    frame_rate: 500
    awb_gain:
    - 2.6426714354585625
    - 1.0
    - 1.763630761259051
    cam2rgb:
    - - 1.1484375
      - -0.048828125
      - -0.099609375
    - - -0.013671875
      - 1.315429688
      - -0.30078125
    - - 0.037109375
      - -0.40234375
      - 1.364257813
    gamma_power: 2.2
  gt_shutter_range:
  - 0
"""


def _maybe(path: str, loader):
    return loader(path) if os.path.isfile(path) else None


def load_calibration_meta(meta_dir: str) -> Dict:
    j = lambda p: os.path.join(meta_dir, p)
    params = dict(
        black_level_a_map=_maybe(j("black_level_a_map.npy"), np.load),
        black_level_b_map=_maybe(j("black_level_b_map.npy"), np.load),
    )
    npz = np.load(j("betas_calibration_with_fit_noise_poly_gpu_per_cfa.npz"))
    params["noise_betas_4x4"] = npz["betas"]
    params["noise_terms_PO"] = npz["terms"]
    return params


def inverse_gamma_and_oetf(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    assert x.max() <= 1.0 and x.min() >= 0.0, "Input sRGB should be in [0,1]"
    y = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return y.astype(np.float32)


def inverse_color_matrix_and_wb(
    linear_rgb: np.ndarray, ccm_3x3: np.ndarray, wb_rgb: Tuple[float, float, float]
) -> np.ndarray:
    H, W, _ = linear_rgb.shape
    M = np.asarray(ccm_3x3 if ccm_3x3 is not None else np.eye(3), np.float32)
    inv_ccm = np.linalg.inv(M)
    wb = np.asarray(wb_rgb if wb_rgb is not None else (1.0, 1.0, 1.0), np.float32)
    rgb = linear_rgb.reshape(-1, 3).astype(np.float32)
    rgb = (rgb @ inv_ccm.T) / wb.reshape(1, 3)
    return rgb.reshape(H, W, 3).astype(np.float32)


def _cfa_map(pattern: str = "RGGB", quad: bool = False) -> np.ndarray:
    pattern = pattern.upper()
    if not quad:
        tbl = {
            "RGGB": np.array([[0, 1], [1, 2]], np.int32),
            "BGGR": np.array([[2, 1], [1, 0]], np.int32),
            "GRBG": np.array([[1, 0], [2, 1]], np.int32),
            "GBRG": np.array([[1, 2], [0, 1]], np.int32),
        }
    else:
        tbl = {
            "RGGB": np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 2, 2], [1, 1, 2, 2]], np.int32),
            "BGGR": np.array([[2, 2, 1, 1], [2, 2, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]], np.int32),
            "GRBG": np.array([[1, 1, 0, 0], [1, 1, 2, 2], [0, 0, 1, 1], [2, 2, 1, 1]], np.int32),
            "GBRG": np.array([[1, 1, 2, 2], [1, 1, 0, 0], [2, 2, 1, 1], [0, 0, 1, 1]], np.int32),
        }
    if pattern not in tbl:
        raise ValueError(f"Unsupported CFA pattern: {pattern}")
    return tbl[pattern]


def re_mosaicing(sensor_rgb: np.ndarray, pattern: str = "RGGB", quad: bool = False) -> np.ndarray:
    H, W, _ = sensor_rgb.shape
    cfa = _cfa_map(pattern, quad)
    bh, bw = cfa.shape
    raw = np.zeros((H, W), np.float32)
    for dy in range(bh):
        for dx in range(bw):
            ch = cfa[dy, dx]
            raw[dy:H:bh, dx:W:bw] = sensor_rgb[dy:H:bh, dx:W:bw, ch]
    return raw


def add_fixed_pattern_and_dark(
    mu: np.ndarray,
    black_level_a_map: Optional[np.ndarray] = None,
    black_level_b_map: Optional[np.ndarray] = None,
    exposure_ms: float = 0.0,
) -> np.ndarray:
    if black_level_a_map is not None:
        mu = mu + exposure_ms * black_level_a_map
    if black_level_b_map is not None:
        mu = mu + black_level_b_map
    return mu


def _tile_betas_to_image(betas_4x4: np.ndarray, H: int, W: int) -> np.ndarray:
    bh, bw, K = betas_4x4.shape
    assert (bh, bw) == (4, 4), "expect (4,4,K)"
    ty, tx = int(np.ceil(H / bh)), int(np.ceil(W / bw))
    return np.tile(betas_4x4, (ty, tx, 1))[:H, :W, :]


def predict_variance_poly(
    mu_dn: np.ndarray, exposure_ms: float, betas_4x4: np.ndarray, terms_PO: np.ndarray
) -> np.ndarray:
    H, W = mu_dn.shape
    K = terms_PO.shape[0]
    betas_img = _tile_betas_to_image(betas_4x4, H, W)
    feats = np.empty((H, W, K), np.float32)
    for k in range(K):
        P, O = terms_PO[k]
        feats[..., k] = (mu_dn ** float(P)) * ((exposure_ms) ** float(O))
    var_hat = np.sum(betas_img * feats, axis=-1)
    return np.clip(var_hat, 0.0, None).astype(np.float32)


def add_gaussian_noise_from_var(
    mu_with_fixed: np.ndarray, var_map: np.ndarray, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    sigma = np.sqrt(var_map, dtype=np.float32) if hasattr(np, "sqrt") else np.sqrt(var_map)
    noise = rng.standard_normal(mu_with_fixed.shape).astype(np.float32) * sigma
    return (mu_with_fixed + noise).astype(np.float32), noise


def quantize_to_adc(signal_dn: np.ndarray, bit_depth: int = 10) -> np.ndarray:

    step = 1.0 / (2**bit_depth - 1)

    quantize_noise = np.random.uniform(-0.5 * step, 0.5 * step, signal_dn.shape).astype(np.float32)
    noisy_signal = np.clip(signal_dn + quantize_noise, 0.0, 1.0)
    sig = np.round(noisy_signal * (2**bit_depth - 1))
    return sig.astype(np.uint16)


def srgb_to_raw(
    srgb_bgr_or_rgb: np.ndarray, camera_params: Dict, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    img = srgb_bgr_or_rgb
    lin = inverse_gamma_and_oetf(img)
    info(f"[0] Inverse gamma and OETF, lin: {lin.min():.8f} - {lin.max():.8f}")
    sensor_rgb = inverse_color_matrix_and_wb(
        lin, camera_params.get("color_matrix", None), camera_params.get("wb", None)
    )
    sensor_rgb = np.clip(sensor_rgb, 0.0, 1.0)
    info(f"[1] Inverse CCM and WB, sensor_rgb: {sensor_rgb.min():.8f} - {sensor_rgb.max():.8f}")
    mosaic_clearn = re_mosaicing(
        sensor_rgb, pattern=camera_params.get("cfa_pattern", "RGGB"), quad=camera_params.get("quad_bayer", False)
    )
    info(f"[2] Re-mosaicing to RAW, mosaic: {mosaic_clearn.min():.8f} - {mosaic_clearn.max():.8f}")
    mu_fixed = add_fixed_pattern_and_dark(
        mosaic_clearn,
        black_level_a_map=camera_params.get("black_level_a_map", None),
        black_level_b_map=camera_params.get("black_level_b_map", None),
        exposure_ms=camera_params["exposure_ms"],
    )
    info(f"[3] Add fixed pattern in dark, mu_fixed: {mu_fixed.min():.8f} - {mu_fixed.max():.8f}")
    betas_4x4 = camera_params.get("noise_betas_4x4", None)
    terms_PO = camera_params.get("noise_terms_PO", None)
    var_hat = predict_variance_poly(mosaic_clearn, camera_params["exposure_ms"], betas_4x4, terms_PO)
    image_noisy, dynamic_noise = add_gaussian_noise_from_var(mu_fixed, var_hat, rng=rng)
    info(f"[4] Add Gaussian noise from predicted variance, noisy: {image_noisy.min():.8f} - {image_noisy.max():.8f}")
    raw_dn = quantize_to_adc(image_noisy, bit_depth=camera_params.get("bit_depth", 8))
    info(f"[5] Quantize to ADC, raw_dn: {raw_dn.min()} - {raw_dn.max()}")
    return raw_dn


class sRGBToRAWGenerator:
    def __init__(self, meta_dir: str, seed: Optional[int] = 1234):
        calib = load_calibration_meta(meta_dir)
        ccm = np.array(
            [
                [1.1484375, -0.048828125, -0.099609375],
                [-0.013671875, 1.315429688, -0.30078125],
                [0.037109375, -0.40234375, 1.364257813],
            ],
            dtype=np.float32,
        )
        wb = np.array([2.6426714354585625, 1.0, 1.763630761259051], np.float32)
        cam = dict(
            color_matrix=ccm,
            wb=wb,
            cfa_pattern="RGGB",
            quad_bayer=True,
            exposure_ms=10.0,
            exposure_scale=1.0,
            bit_depth=8,
        )
        cam.update({k: v for k, v in calib.items() if v is not None})
        self.camera_params = cam
        for k, v in cam.items():
            info(
                f"sRGBToRAWGenerator: camera_params[{k}] shape/type: {v.shape if isinstance(v, np.ndarray) else type(v)}"
            )
            info(f"   value (sample): {v if not isinstance(v, np.ndarray) else v.flatten()[:5]}")
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    def srgb2raw_from_path(self, srgb_image_path: str) -> np.ndarray:
        I0 = cv2.cvtColor(cv2.imread(srgb_image_path), cv2.COLOR_BGR2RGB)
        I0 = cv2.rotate(I0, cv2.ROTATE_90_CLOCKWISE)
        I0 = cv2.resize(I0, (2448, 3264))
        I0 = I0.astype(np.float32) / 255.0
        return self.srgb2raw(I0)

    def set_exposure_time_ms(self, exposure_ms: float) -> None:
        if exposure_ms <= 0:
            raise ValueError("exposure_ms must be positive.")
        self.camera_params["exposure_ms"] = float(exposure_ms)

    @torch.no_grad()
    def srgb2raw(self, srgb_image: np.ndarray) -> np.ndarray:
        return srgb_to_raw(srgb_image, self.camera_params, rng=self.rng)


def get_eiger_sRGB_to_RAW_generator() -> sRGBToRAWGenerator:
    meta_dir = "meta/ARGB_ERGB_Eiger_I_dt/"
    gen = sRGBToRAWGenerator(meta_dir, seed=9527)
    return gen


def get_gen2_sRGB_to_RAW_generator() -> sRGBToRAWGenerator:
    meta_dir = "meta/ARGB_EW_GEN2_I_dt/"
    gen = sRGBToRAWGenerator(meta_dir, seed=9527)
    return gen
