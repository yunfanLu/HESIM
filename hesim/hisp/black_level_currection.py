from os.path import dirname, join

import numpy as np
from absl.logging import info, warn


class BlackLevelCorrector:
    def __init__(self, calib_dir):
        """Initialize the black level corrector by loading slope and offset maps.
        calib_dir : str Directory containing 'dark_slope_map.npy' and 'dark_offset_map.npy'.
        """
        self.offset_map = np.load(join(calib_dir, "dark_offset_map.npy"))
        self.slope_map = np.load(join(calib_dir, "dark_slope_map.npy"))
        assert self.offset_map.shape == self.slope_map.shape, "Offset and slope maps must have the same shape."
        self.H, self.W = self.offset_map.shape
        info(f"Init BlackLevelCorrector: {calib_dir}")
        info(f"  offset_map: {self.offset_map.shape}, {self.offset_map.max()}, {self.offset_map.min()}")
        info(f"  slope_map : {self.slope_map.shape}, {self.slope_map.max()}, {self.slope_map.min()}")

    def __call__(self, raw, exposure_time_ms):
        return self.correct(raw, exposure_time_ms)

    def correct(self, raw, exposure_time_ms):
        """Apply black level correction to a raw frame given the exposure time.
        raw : ndarray (H, W) Input raw image (float32 within [0, 1] to be casted to float).
        exposure_time_ms : float Exposure time in milliseconds.
        Returns: raw_corrected : ndarray (H, W) Black level corrected raw image (float32).
        """
        exposure_time_ms = np.clip(exposure_time_ms, 0.001, 100.0)
        if raw.shape != (self.H, self.W) and raw.shape == (3264, 2312):
            warn(f"raw shape {raw.shape} is not equal to {self.H, self.W}, use raw directly")
            return raw
        fpn_noise = self.slope_map * exposure_time_ms + self.offset_map
        raw_corrected = raw - fpn_noise
        return raw_corrected.astype(np.float32)


def get_black_level_corrector(sensor="eiger"):
    if sensor.lower() == "eiger":
        meta_dir = join(dirname(__file__), "../../meta/ARGB_ERGB_Eiger_I_dt/")
        return BlackLevelCorrector(meta_dir)
    elif sensor.lower() == "gen2":
        meta_dir = join(dirname(__file__), "../../meta/ARGB_EW_GEN2_I_dt")
        return BlackLevelCorrector(meta_dir)
    raise ValueError(f"Unsupported sensor: {sensor}. Expected 'eiger' or 'gen2'.")
