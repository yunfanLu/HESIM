import math
import os
from os.path import abspath, dirname, join
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from absl.logging import info
from absl.testing import absltest
from PIL import Image
from scipy.stats import norm
from tqdm import tqdm

from hesim.io import three_channel_to_raw_mapping

DEBUG = False


def Q(x: torch.Tensor) -> torch.Tensor:

    return 0.5 * torch.erfc(x / math.sqrt(2.0))


class EVSFrameSimulator(torch.nn.Module):
    """
    Hybrid EVS simulator for a single interval [I0, I1] -> event frame (+1/0/-1).
    Parameters are initialized from calibrated betas per CFA position.
    """

    def __init__(
        self,
        betas_cfsxcfs_6: torch.Tensor,
        cfa_size: int,
        mu_n_map: Optional[torch.Tensor],
        sensor: str,
        device: str = "cuda",
        theta_scale: float = 1.0,
    ):
        super().__init__()
        theta_hw = 0.75e-3
        assert betas_cfsxcfs_6.shape[-1] == 6
        self.register_buffer("betas", betas_cfsxcfs_6.to(device=device, dtype=torch.float32))
        self.theta_hw = float(theta_hw) * float(theta_scale)
        self.cfa_size = int(cfa_size)
        self.device = device
        self.mu_n_map = mu_n_map.to(device=device, dtype=torch.float32)
        self.sensor = sensor

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(9527))

    def _tile_betas_per_pixel(self, H: int, W: int) -> Dict[str, torch.Tensor]:
        """Broadcast β0..β5 to per-pixel maps according to 2×2 CFA position."""
        if self.cfa_size == 1:

            b = self.betas[0, 0]
            out = {f"b{k}": b[k].expand(H, W) for k in range(6)}
            return out

        bmaps = {}
        for k in range(6):

            tile = torch.zeros((H, W), device=self.device, dtype=torch.float32)
            for r_off in range(self.cfa_size):
                for c_off in range(self.cfa_size):
                    mask_r = torch.arange(H, device=self.device) % self.cfa_size == r_off
                    mask_c = torch.arange(W, device=self.device) % self.cfa_size == c_off
                    mask = mask_r[:, None] & mask_c[None, :]
                    tile[mask] = self.betas[r_off, c_off, k]
            bmaps[f"b{k}"] = tile
        return bmaps

    @torch.no_grad()
    def simulate_from_path(self, I0_path: str, I1_path: str):
        I0 = cv2.imread(I0_path)
        I0 = cv2.cvtColor(I0, cv2.COLOR_BGR2RGB)
        I0 = cv2.rotate(I0, cv2.ROTATE_90_CLOCKWISE)
        I1 = cv2.imread(I1_path)
        I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
        I1 = cv2.rotate(I1, cv2.ROTATE_90_CLOCKWISE)
        if self.sensor == "eiger":
            I0 = cv2.resize(I0, (1224, 1632))
            I1 = cv2.resize(I1, (1224, 1632))
            I0_quad = three_channel_to_raw_mapping(I0)
            I1_quad = three_channel_to_raw_mapping(I1)
        else:
            I0 = cv2.resize(I0, (612, 816))
            I1 = cv2.resize(I1, (612, 816))
            I0_quad = I0.mean(axis=2)
            I1_quad = I1.mean(axis=2)

        I0_quad = (I0_quad / 255.0) ** 2.2 / 20
        I1_quad = (I1_quad / 255.0) ** 2.2 / 20
        I0 = torch.from_numpy(I0_quad).float()
        I1 = torch.from_numpy(I1_quad).float()
        return self.simulate(I0, I1)

    simulatr_from_path = simulate_from_path

    @torch.no_grad()
    def simulate(self, I0: torch.Tensor, I1: torch.Tensor):
        """E: (H,W) in {-1,0,+1} (optional) P_on, P_off: (H,W)"""
        assert I0.shape == I1.shape and I0.ndim == 2
        I0 = I0.to(self.device, dtype=torch.float32)
        I1 = I1.to(self.device, dtype=torch.float32)
        H, W = I0.shape
        betas = self._tile_betas_per_pixel(H, W)
        b0, b1, b2, b3, b4, b5 = (betas["b0"], betas["b1"], betas["b2"], betas["b3"], betas["b4"], betas["b5"])

        Ic = 0.5 * (I0 + I1)

        V0 = torch.clamp(b1 * I0 + b2, min=1e-18)
        V1 = torch.clamp(b1 * I1 + b2, min=1e-18)
        log0 = torch.log(V0)
        log1 = torch.log(V1)
        S = log1 - log0
        info(f"S: {S.shape}, {S.min().item():.3e}..{S.max().item():.3e}")

        denom_inside = 2.0 * ((b3 * Ic) ** 2 + (b4**2) + 2.0 * b5 * b3 * Ic * b4)
        denom_inside = torch.clamp(denom_inside, min=1e-18)
        denom_inside = torch.sqrt(denom_inside)

        denom_1 = torch.clamp((b1 * Ic + b2), min=1e-18)

        Q_P_n1 = (b0 * self.theta_hw * denom_1) / denom_inside

        P_noise_event = Q(Q_P_n1)
        sigma_n = denom_inside / denom_1

        mu_n = self.mu_n_map

        theta_pos = b0 * self.theta_hw
        if DEBUG:
            info(f"theta_pos: {theta_pos.shape}, {theta_pos.min().item():.3e}..{theta_pos.max().item():.3e}")
            info(f"Signal S : {S.shape}, {S.min().item():.3e}..{S.max().item():.3e}")
            info(f"sigma_n  :  {sigma_n.shape}, {sigma_n.min().item():.3e}..{sigma_n.max().item():.3e}")
            info(f"mu_n.    :  {mu_n.shape}, {mu_n.min().item():.3e}..{mu_n.max().item():.3e}")

        z_on = (theta_pos - (S + mu_n)) / torch.clamp(sigma_n, min=1e-18)
        P_on = Q(z_on)
        z_off = (-theta_pos - (S + mu_n)) / torch.clamp(sigma_n, min=1e-18)
        P_off = 1 - Q(z_off)

        U = torch.rand((H, W), device=self.device, generator=self.rng)
        E = torch.zeros((H, W), device=self.device, dtype=torch.int8)
        th_pos = P_on
        th_neg = P_on + P_off
        E[U < th_pos] = 1
        E[(U >= th_pos) & (U < th_neg)] = -1
        return E, P_on, P_off, P_noise_event, S


def get_evs_noise_simulator(evs_betas_path: str, evs_dark_noise_npz: str, sensor: str = "eiger", theta_scale=1.0):
    betas = torch.from_numpy(np.load(evs_betas_path))

    P_dark_event_on = torch.from_numpy(np.load(evs_dark_noise_npz)["positive_rate"])
    P_dark_event_off = torch.from_numpy(np.load(evs_dark_noise_npz)["negative_rate"])
    mu_n_map = P_dark_event_on - P_dark_event_off
    cfa_size, _, _ = betas.shape
    sim = EVSFrameSimulator(
        betas_cfsxcfs_6=betas, cfa_size=cfa_size, mu_n_map=mu_n_map, sensor=sensor, theta_scale=theta_scale
    )
    return sim


def get_eiger_evs_simulator(theta_scale):
    from hesim.simulator.calibration_config import (
        Eiger_EVS_ALL_P_N_Calibration_Beta,
        Eiger_EVS_ALL_P_N_Calibration_JSON,
        Eiger_EVS_Calibration_Folder,
        Eiger_EVS_Dark_Noise_NPZ,
        Eiger_EVS_Shot_Noise_NPZ,
    )

    return get_evs_noise_simulator(
        Eiger_EVS_ALL_P_N_Calibration_Beta, Eiger_EVS_Dark_Noise_NPZ, sensor="eiger", theta_scale=theta_scale
    )


def get_gen2_evs_simulator(theta_scale):
    from hesim.simulator.calibration_config import (
        GEN2_EVS_ALL_P_N_Calibration_Beta,
        GEN2_EVS_ALL_P_N_Calibration_JSON,
        GEN2_EVS_Calibration_Folder,
        GEN2_EVS_Dark_Noise_NPZ,
        GEN2_EVS_Shot_Noise_NPZ,
    )

    return get_evs_noise_simulator(
        GEN2_EVS_ALL_P_N_Calibration_Beta, GEN2_EVS_Dark_Noise_NPZ, sensor="gen2", theta_scale=theta_scale
    )
