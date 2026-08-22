import os
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from os import makedirs
from os.path import basename, dirname, isfile, join
from typing import Optional

import cv2
import numpy as np
from absl import app
from absl.logging import info
from absl.testing import absltest
from numpy.linalg import lstsq
from scipy.stats import binned_statistic
from tqdm import tqdm

from hesim.io import evs_raw_stack, read_evs_raw, sorted_raws
from hesim.visualization import visualize_matrix_with_histogram

DEBUG = True


def _per_file_stats(path: str):
    evs = read_evs_raw(path)

    pos = (evs == 1).astype(np.uint8)
    neg = (evs == -1).astype(np.uint8)
    abs_mask = (evs != 0).astype(np.uint8)
    return pos, neg, abs_mask, np.int64(1)


def save_event_shot_noise_into_npy(
    brightness_evs_folders,
    out_dir,
    num_workers: Optional[int] = None,
    chunksize: int = 2,
):

    all_paths = list(chain.from_iterable(sorted_raws(folder) for folder in brightness_evs_folders))
    if DEBUG:
        all_paths = all_paths[:20000]
    if not all_paths:
        raise RuntimeError("No EVS files found in the given folders.")
    info(f"[EVS] Found {len(all_paths)} raw file(s)")

    p0, n0, a0, f0 = _per_file_stats(all_paths[0])
    H, W = p0.shape
    total_pos = np.zeros((H, W), dtype=np.uint64)
    total_neg = np.zeros((H, W), dtype=np.uint64)
    total_evs = np.zeros((H, W), dtype=np.uint64)
    total_frames = np.int64(0)
    print(f"H={H} W={W} total_files(frames)={len(all_paths)}")

    total_pos += p0
    total_neg += n0
    total_evs += a0
    total_frames += f0

    rest_paths = all_paths[1:]
    if rest_paths:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for (pos, neg, abs_mask, frames), path in tqdm(
                zip(ex.map(_per_file_stats, rest_paths, chunksize=chunksize), rest_paths),
                total=len(rest_paths),
                desc="EVS per-file reduction",
            ):
                if pos.shape != (H, W):
                    raise ValueError(f"Shape mismatch for {path}: got {pos.shape}, expected {(H, W)}")
                total_pos += pos
                total_neg += neg
                total_evs += abs_mask
                total_frames += frames
                print(f"Reading total_frames: {total_frames}")

    frames_scalar = int(total_frames)
    if frames_scalar <= 0:
        raise RuntimeError("total_frames == 0; check your inputs.")

    positive_rate = (total_pos / frames_scalar).astype(np.float32)
    negative_rate = (total_neg / frames_scalar).astype(np.float32)
    evs_rate = (total_evs / frames_scalar).astype(np.float32)

    info(f"[EVS] H={H} W={W} total_files(frames)={len(all_paths)} total_frames={frames_scalar}")

    np.savez(
        join(out_dir, "evs_shot_noise.npz"),
        positive_rate=positive_rate,
        negative_rate=negative_rate,
        evs_rate=evs_rate,
        total_frames=frames_scalar,
        H=H,
        W=W,
    )
    visualize_matrix_with_histogram(
        positive_rate, title="evs_shot_noise-positive_rate", filename=join(out_dir, "evs_shot_noise-positive_rate.png")
    )
    visualize_matrix_with_histogram(
        negative_rate, title="evs_shot_noise-negative_rate", filename=join(out_dir, "evs_shot_noise-negative_rate.png")
    )
    visualize_matrix_with_histogram(
        evs_rate, title="evs_shot_noise-evs_rate", filename=join(out_dir, "evs_shot_noise-evs_rate.png")
    )


def main_ARGB_ERGB_Eiger(args):
    brightness_evs_folders = [
        "calibration_data/ARGB_ERGB_Eiger/resolution_board_color_checker_exp01ms_20240516153213128/EVS/EventMode16_1632_1224_20240516153213128",
        "calibration_data/ARGB_ERGB_Eiger/resolution_board_color_checker_exp02ms_20240516153221655/EVS/EventMode16_1632_1224_20240516153221655",
        "calibration_data/ARGB_ERGB_Eiger/resolution_board_color_checker_exp10ms_20240516153244340/EVS/EventMode16_1632_1224_20240516153244340",
        "calibration_data/ARGB_ERGB_Eiger/resolution_board_color_checker_exp20ms_20240516153253743/EVS/EventMode16_1632_1224_20240516153253743",
        "calibration_data/ARGB_ERGB_Eiger/resolution_board_color_checker_exp40ms_20240516153312041/EVS/EventMode16_1632_1224_20240516153312041",
        "calibration_data/ARGB_ERGB_Eiger/resolution_board_color_checker_exp50ms_20240516153331327/EVS/EventMode16_1632_1224_20240516153331327",
        "calibration_data/ARGB_ERGB_Eiger/resolution_board_color_checker_exp80ms_20240516153343982/EVS/EventMode16_1632_1224_20240516153343982",
    ]
    out_dir = "./meta/ARGB_ERGB_Eiger_EVS_SHOT_Noise/"
    makedirs(out_dir, exist_ok=True)
    save_event_shot_noise_into_npy(brightness_evs_folders, out_dir)


def main_ARGB_EW_GEN2(args):
    sensitity7 = "calibration_data/ARGB_EW_EVB_GEN2/GEB2Sensitity7/"
    brightness_evs_folders = [
        sensitity7 + "20250922170640101sen7_1/EVS/normal_v2_816_612_20250922170640101/evs_raw",
        sensitity7 + "20250922170514149sen7_2/EVS/normal_v2_816_612_20250922170514149/evs_raw",
        sensitity7 + "20250922170346721sen7_5/EVS/normal_v2_816_612_20250922170346721/evs_raw",
        sensitity7 + "20250922170058811sen7_20/EVS/normal_v2_816_612_20250922170058811/evs_raw",
        sensitity7 + "20250922165939212sen7_40/EVS/normal_v2_816_612_20250922165939212/evs_raw",
        sensitity7 + "20250922165807108sen7_50/EVS/normal_v2_816_612_20250922165807108/evs_raw",
        sensitity7 + "20250922165651933sen7_80/EVS/normal_v2_816_612_20250922165651933/evs_raw",
    ]
    out_dir = "meta/ARGB_EW_GEN2_EVS_SHOT_Noise/"
    makedirs(out_dir, exist_ok=True)
    save_event_shot_noise_into_npy(brightness_evs_folders, out_dir)


if __name__ == "__main__":
    app.run(main_ARGB_ERGB_Eiger)
