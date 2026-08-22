from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any, Dict, Literal, Optional, Union

import numpy as np

from hesim.hisp.auto_white_balance_by_grayness_index import auto_wb_gi
from hesim.hisp.black_level_currection import get_black_level_corrector
from hesim.io import read_raw

__all__ = ["HISPConfig", "HISP", "isp"]


@dataclass(frozen=True)
class HISPConfig:
    trans_quad2bayer_type: Literal["binning", "pixel_shuffle"] = "binning"
    demosaic_algorithm: Literal["Malvar"] = "Malvar"

    black_level_currect: bool = False

    use_awb: bool = True
    awb_top_percent: float = 0.002

    is_quad_bayer: bool = False
    use_color_correction: bool = False
    cc_method: Literal["robust", "retinex"] = "retinex"

    use_gamma: bool = False
    gamma: float = 2.2

    sensor: Literal["gen2", "eiger"] = "eiger"


class HISP:
    def __init__(self, config: Optional[HISPConfig] = None):
        self.cfg = config or HISPConfig()
        self._black_level_corrector = None

    def _get_black_level_corrector(self):
        if self._black_level_corrector is None:
            self._black_level_corrector = get_black_level_corrector(sensor=self.cfg.sensor)
        return self._black_level_corrector

    def isp(
        self,
        raw: Union[str, PathLike[str], np.ndarray],
        exposure: Optional[float] = None,
        cc_kwargs: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Convert a RAW path or two-dimensional RAW array to sRGB."""
        exp = 1.0 if exposure is None else float(exposure)

        if isinstance(raw, (str, PathLike)):
            raw = read_raw(str(raw))
        if not isinstance(raw, np.ndarray) or raw.ndim != 2:
            raise ValueError("raw must be a two-dimensional NumPy array or a path to one.")
        raw = raw.astype(np.float32)

        if self.cfg.black_level_currect:
            raw = self._get_black_level_corrector()(raw, exp)

        if self.cfg.is_quad_bayer:
            from hesim.hisp.quad_bayer_demosaic_by_trans_quad2bayer import quad_bayer_demosaic_by_trans_quad2bayer

            rgb = quad_bayer_demosaic_by_trans_quad2bayer(
                raw,
                trans_quad2bayer_type=self.cfg.trans_quad2bayer_type,
                algorithm=self.cfg.demosaic_algorithm,
            ).astype(np.float32)
        else:
            rgb = _demosaic_bayer(raw, self.cfg.demosaic_algorithm)

        denom = float(rgb.max()) + 1e-8
        rgb = rgb / denom

        if self.cfg.use_awb:
            rgb, _illum = auto_wb_gi(rgb, top_percent=self.cfg.awb_top_percent)

        if self.cfg.use_color_correction:
            from hesim.hisp.auto_color_correction import ColorCorrectionFactory

            cc = ColorCorrectionFactory(method=self.cfg.cc_method)
            rgb = cc.auto_ccm(rgb) if not cc_kwargs else cc.auto_ccm(rgb, **cc_kwargs)

        if self.cfg.use_gamma:
            g = max(self.cfg.gamma, 1e-6)
            rgb = np.clip(rgb, 0.0, 1.0) ** (1.0 / g)

        return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _demosaic_bayer(raw: np.ndarray, algorithm: str) -> np.ndarray:
    from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004, demosaicing_CFA_Bayer_Menon2007

    if algorithm.lower() == "malvar":
        return demosaicing_CFA_Bayer_Malvar2004(raw, pattern="GRBG").astype(np.float32)
    return demosaicing_CFA_Bayer_Menon2007(raw, pattern="GRBG").astype(np.float32)


def isp(
    raw_path: str,
    *,
    exposure: Optional[float] = None,
    config: Optional[HISPConfig] = None,
    cc_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    engine = HISP(config=config)
    return engine.isp(raw_path, exposure=exposure, cc_kwargs=cc_kwargs)
