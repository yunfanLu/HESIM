import csv
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor
from os import listdir, makedirs
from os.path import basename, dirname, getsize, isdir, isfile, join
from typing import Dict, List, Optional

import cv2
import numpy as np
from absl import app, logging
from absl.logging import debug, info
from absl.testing import absltest
from tqdm import tqdm

from hesim.hisp.black_level_currection import get_black_level_corrector

_Bayer_RE_EIGER = re.compile(r"bayer_bit8_3264_2448_\d+")
_QB_RE_EIGER = re.compile(r"quadbayer_bit8_3264_2448_\d+")
_QB_RE_GEN2 = re.compile(r"quadbayer_10bit_3264_2448_\d+")

DEBUG = False
print(f"IO DEBUG = {DEBUG}")


"""
(base) luyunfan:H-ESIM/ (main✗) $ tree -L 1 dataset/2023-04-06-APS40ms-BlurData/20230405175148869/APS                                                                                                                               [2:15:11]
dataset/2023-04-06-APS40ms-BlurData/20230405175148869/APS
├── bayer_bit8_3264_2448_20230405175148869
└── bayer_bit8_3264_2448_20230405175148869_info.txt
"""


""" Example of an expected folder structure for a calibration case:
./calibration/resolution_board_color_checker_exp02ms_20240516153221655
├── 3264_2448_8_10_20240516153221655.png
├── APS
│   ├── quadbayer_bit8_3264_2448_20240516153221655
│   └── quadbayer_bit8_3264_2448_20240516153221655_info.txt
├── ApsEvsInfo.txt
├── DeviceCfg.txt
└── EVS
    ├── EventMode16_1632_1224_20240516153221655
    └── EventMode16_1632_1224_20240516153221655_info.txt
"""


"""
/data/luyunfan/workspace/NATURE-HYIMAGING/H-ESIM/dataset/1-HESIM-Temporal-Evaluation/1-Gen2-APX-Video/a20250703093040412
├── APS
│   └── quadbayer_10bit_3264_2448_20250703093040412
│       ├── aps_png
│       └── aps_raw
├── ApsEvsInfo.txt
├── DeviceCfg.txt
└── EVS
    └── normal_v2_816_612_20250703093040412
        └── evs_raw
"""


def find_aps_folder(case_dir: str, frame_type: str, sensor: str) -> str:
    aps = join(case_dir, "APS")
    if not os.path.isdir(aps):
        raise FileNotFoundError(f"{aps} missing")

    subs = [
        d
        for d in os.listdir(aps)
        if (
            (_QB_RE_EIGER.fullmatch(d) or _QB_RE_GEN2.fullmatch(d) or _Bayer_RE_EIGER.fullmatch(d))
            and isdir(join(aps, d))
        )
    ]
    assert subs, f"No quadbayer dir in {aps}"
    if sensor.lower() == "gen2":
        return join(aps, subs[0], "aps_raw" if frame_type == "raw" else "aps_png")
    return join(aps, subs[0])


def _load_txt_info(path: str) -> List[dict]:
    """
    Parse *_info.txt* from APS folder (CSV format).
    APS
    index,sof,eof,exposure_time,offset,length
    0,35734993,35769818,1010,0,7990312
    1,35859984,35894809,1010,7990312,7990312
    EVS:
    index,timestamp,offset,length
    0,35645775,0,388632
    1,35647014,388632,358816
    2,35648253,747448,335760
    --- Sensor Gen2 ---
    APS
    index,timestamp,sof,eof,exposure_time,offset,length
    0000000000,53857575,53857674,53892697,99,0,9987992
    0000000001,53907563,53907662,53942685,99,9987992,9987992
    0000000002,53957551,53957650,53992673,99,19975984,9987992
    EVS
    index,timestamp,sof,eof,exposure_time,offset,length
    0000000000,0,221,611221,221,0,124984
    0000000001,0,221,611221,221,124984,124984
    """
    records: List[dict] = []
    if not path:
        return records
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:

            row_parsed = [
                int(row["index"]),
                int(row["sof"]),
                int(row["eof"]),
                int(row["exposure_time"]),
                int(row["offset"]),
                int(row["length"]),
            ]
            records.append(row_parsed)
    return records


def _load_evs_txt_info(evs_info_txt):
    records: List[dict] = []
    if not evs_info_txt:
        return records
    with open(evs_info_txt, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:

            row_parsed = [
                int(row["index"]),
                int(row["timestamp"]),
                int(row["offset"]),
                int(row["length"]),
            ]
            records.append(row_parsed)
    return records


def find_aps_txt_info_file(case_dir: str) -> str:
    aps = join(case_dir, "APS")
    if not os.path.isdir(aps):
        raise FileNotFoundError(f"{aps} missing")
    for f in os.listdir(aps):
        if f.endswith("_info.txt"):
            return join(aps, f)
    raise ValueError(f"No APS info file found in {aps}")


def find_aps_frames_with_txtinfo(case_dir: str, frame_type: type = "png", sensor: str = "EIGER"):
    aps_frame_folder = find_aps_folder(case_dir, frame_type, sensor)
    aps_info_txt = find_aps_txt_info_file(case_dir)
    aps_info_records = _load_txt_info(aps_info_txt)
    frames = [
        f for f in os.listdir(aps_frame_folder) if (f.lower().endswith(f".{frame_type}") and "_good_rgb" not in f)
    ]
    frames.sort(key=lambda f: int(analytic_file_name(f)[3]))
    frames_with_info = []
    for frame, aps_info in zip(frames, aps_info_records):
        frames_with_info.append([join(aps_frame_folder, frame), *aps_info])
    return frames_with_info


def _find_evs_root(case_dir):
    for name in ("EVS", "DVS"):
        evs = join(case_dir, name)
        if isdir(evs):
            return evs
    else:
        raise FileNotFoundError(f"Neither 'EVS' nor 'DVS' folder found under {case_dir}")


def find_evs_folder(case_dir: str, sensor: str = "EIGER") -> str:
    evs = _find_evs_root(case_dir)

    if sensor.lower() == "eiger":
        subs = [d for d in os.listdir(evs) if (isdir(join(evs, d)) and d.startswith("EventMode"))]
        return join(evs, subs[0])
    elif sensor.lower() == "gen2":
        subs = [d for d in os.listdir(evs) if (isdir(join(evs, d)) and d.startswith("normal_v2"))]
        return join(evs, subs[0], "evs_raw")


def find_evs_txt_info_file(case_dir):
    evs = _find_evs_root(case_dir)
    for f in os.listdir(evs):
        if f.endswith("_info.txt"):
            return join(evs, f)
    raise ValueError(f"No EVS info file found in {evs}")


def find_evs_frames_with_txtinfo(case_dir, sensor="EIGER"):
    evs_raw_folder = find_evs_folder(case_dir, sensor)
    evs_info_txt = find_evs_txt_info_file(case_dir)
    evs_info_records = _load_evs_txt_info(evs_info_txt)
    raws = [f for f in os.listdir(evs_raw_folder) if f.lower().endswith(".raw")]
    raws.sort(key=lambda f: int(analytic_file_name(f)[3]))
    evs_with_info = []
    for evs_frame, evs_info in zip(raws, evs_info_records):
        evs_with_info.append([join(evs_raw_folder, evs_frame), *evs_info])
    return evs_with_info


def sorted_raws(folder: str) -> List[str]:
    """Return absolute RAW paths sorted by numeric index N (…_<N>_YYYY.raw)."""
    files = [f for f in os.listdir(folder) if f.lower().endswith(".raw")]
    files.sort(key=lambda f: int(analytic_file_name(f)[3]))
    return [os.path.join(folder, f) for f in files]


def aps_raw_stack(folder: str, exposure_time: float, with_blc: bool) -> np.ndarray:
    """Load every RAW in *folder* –> (N,H,W) float32 stack."""

    def process_with_blc(p):
        return __black_level_corrector(read_raw(p), exposure_time_ms=exposure_time)

    __black_level_corrector = get_black_level_corrector()
    paths = sorted_raws(folder)
    print(f"Found {len(paths)} .raw in {folder}")
    if DEBUG:
        paths = paths[:1000]
        print("IO DEBUG: only use first 1000 raws")
    if not paths:
        raise FileNotFoundError(f"No .raw in {folder}")

    worker = process_with_blc if with_blc else read_raw

    with ProcessPoolExecutor() as executor:
        raws = list(tqdm(executor.map(worker, paths), total=len(paths), desc=f"APS {basename(folder)}"))
    return raws


def evs_raw_stack(folder: str, max_frames: Optional[int] = None) -> np.ndarray:
    """Load EVS frames in *folder* → (N,H,W) int8 / int32."""
    paths = sorted_raws(folder)
    if (max_frames is not None) and (max_frames < len(paths)):
        paths = paths[:max_frames]

    with ProcessPoolExecutor() as executor:
        frames = list(tqdm(executor.map(read_evs_raw, paths), total=len(paths), desc=f"EVS {basename(folder)}"))
    return np.stack(frames, 0)


def raw_map_to_3_channel(quad):
    debug(f"quad range: {quad.min()} to {quad.max()}")
    h, w = quad.shape
    raw = np.zeros([h, w, 3], dtype=np.float32)

    raw[0::4, 0::4, 2] = quad[0::4, 0::4]
    raw[0::4, 1::4, 2] = quad[0::4, 1::4]
    raw[1::4, 0::4, 2] = quad[1::4, 0::4]
    raw[1::4, 1::4, 2] = quad[1::4, 1::4]

    raw[0::4, 2::4, 1] = quad[0::4, 2::4]
    raw[0::4, 3::4, 1] = quad[0::4, 3::4]
    raw[1::4, 2::4, 1] = quad[1::4, 2::4]
    raw[1::4, 3::4, 1] = quad[1::4, 3::4]

    raw[2::4, 0::4, 1] = quad[2::4, 0::4]
    raw[2::4, 1::4, 1] = quad[2::4, 1::4]
    raw[3::4, 0::4, 1] = quad[3::4, 0::4]
    raw[3::4, 1::4, 1] = quad[3::4, 1::4]

    raw[2::4, 2::4, 0] = quad[2::4, 2::4]
    raw[2::4, 3::4, 0] = quad[2::4, 3::4]
    raw[3::4, 2::4, 0] = quad[3::4, 2::4]
    raw[3::4, 3::4, 0] = quad[3::4, 3::4]
    return raw


def three_channel_to_raw_mapping(img):

    H, W, C = img.shape
    assert C == 3
    raw = np.zeros([H, W], dtype=np.float32)

    raw[0::4, 0::4] = img[0::4, 0::4, 1]
    raw[0::4, 1::4] = img[0::4, 1::4, 1]
    raw[1::4, 0::4] = img[1::4, 0::4, 1]
    raw[1::4, 1::4] = img[1::4, 1::4, 1]

    raw[0::4, 2::4] = img[0::4, 2::4, 2]
    raw[0::4, 3::4] = img[0::4, 3::4, 2]
    raw[1::4, 2::4] = img[1::4, 2::4, 2]
    raw[1::4, 3::4] = img[1::4, 3::4, 2]

    raw[2::4, 0::4] = img[2::4, 0::4, 0]
    raw[2::4, 1::4] = img[2::4, 1::4, 0]
    raw[3::4, 0::4] = img[3::4, 0::4, 0]
    raw[3::4, 1::4] = img[3::4, 1::4, 0]

    raw[2::4, 2::4] = img[2::4, 2::4, 1]
    raw[2::4, 3::4] = img[2::4, 3::4, 1]
    raw[3::4, 2::4] = img[3::4, 2::4, 1]
    raw[3::4, 3::4] = img[3::4, 3::4, 1]
    return raw


def analytic_file_name(file):
    """Returns the meta information from the file name."""
    file_name, file_type = file.split(".")
    file_list = file_name.split("_")

    w, h, b, n = file_list[0], file_list[1], file_list[2], file_list[3]
    return int(w), int(h), int(b), int(n)


def _unpack_10bit_packed(buf: np.ndarray, total_px: int) -> np.ndarray:
    """Unpack "10-bit packed" LE stream (5 bytes → 4 pixels).
    Layout per 5 bytes:  B0 B1 B2 B3 B4
        P0 =  B0               + ((B4 & 0x03) << 8)
        P1 =  B1               + ((B4 & 0x0C) << 6)
        P2 =  B2               + ((B4 & 0x30) << 4)
        P3 =  B3               + ((B4 & 0xC0) << 2)
    """
    buf = buf.reshape(-1, 5)
    p0 = buf[:, 0].astype(np.uint16) | ((buf[:, 4] & 0x03) << 8)
    p1 = buf[:, 1].astype(np.uint16) | ((buf[:, 4] & 0x0C) << 6)
    p2 = buf[:, 2].astype(np.uint16) | ((buf[:, 4] & 0x30) << 4)
    p3 = buf[:, 3].astype(np.uint16) | ((buf[:, 4] & 0xC0) << 2)
    out = np.concatenate([p0, p1, p2, p3]).astype(np.uint16)[:total_px]
    return out


def _read_raw_file(path, normalize):
    """Reads a raw file from the APLEX camera."""
    file_name = path.split("/")[-1]
    W, H, B, N = analytic_file_name(file_name)
    total_px = W * H
    size = getsize(path)
    debug(f"      File Size: {size}, N = {N}, b = {B}, W * H = {W} * {H} = {W * H}")
    if B == 8:

        raw = np.fromfile(path, dtype=np.uint8, count=total_px)
        if raw.size != total_px:
            raise IOError("File truncated!")

    elif B == 10:
        size_16bit = total_px * 2
        size_pack = (total_px * 10 + 7) // 8
        if size == size_16bit:
            raw16 = np.fromfile(path, dtype="<u2", count=total_px)
        elif size == size_pack:
            buf = np.fromfile(path, dtype=np.uint8)
            raw16 = _unpack_10bit_packed(buf, total_px)
        else:
            raise ValueError(f"Unexpected 10-bit file size ({size} B).")
        raw = raw16.reshape((W, H), order="F").astype(np.int32)
        if normalize:
            raw = raw / (2**B - 1)
        return raw
    else:
        raise ValueError(f"Unsupported bit depth: {B} bits.")
    raw = raw.reshape((W, H), order="F").astype(np.float32)
    if normalize:
        raw = raw.astype(np.float32) / (2**B - 1)
    return raw


def read_raw(path, normalize: bool = True) -> np.ndarray:
    if path.endswith(".raw"):
        return _read_raw_file(path, normalize)
    if path.endswith(".npy"):
        return np.load(path)


def vis_raw(path, raw: np.ndarray, bit: int = 8):
    """Visualizate the raw data as an image
    :param path: path to save the image
    :param raw: raw data, should be a 2D array with (H W) resolution
    :param bit: bit depth of the image
    """
    raw_3_channel = raw_map_to_3_channel(raw)
    max_value = 2**bit - 1
    raw_3_channel = raw_3_channel / max_value * 255
    cv2.imwrite(path, raw_3_channel.astype(np.uint8))


def read_evs_raw(event_raw_file_path):
    file_name = event_raw_file_path.split("/")[-1]
    W, H, B, N = analytic_file_name(file_name)
    size = getsize(event_raw_file_path)
    debug(f"  Reading DVS raw file: {event_raw_file_path}")
    debug(f"    File Size: {size}, w * h: {W * H}")
    expected_size = W * H
    if size != expected_size:
        raise IOError(f"File size {size} does not match expected {expected_size} for {file_name}!")
    raw = np.fromfile(event_raw_file_path, dtype=np.uint8, count=size)
    raw = raw.reshape((W, H), order="F").astype(np.int32)

    raw = np.where(raw == 2, 1, np.where(raw == 1, -1, 0))
    return raw


def vis_evs_raw(raw):
    rw, rh = raw.shape
    vis = np.zeros((rw, rh, 3), dtype=np.uint8)
    vis[raw == 0] = [255, 255, 255]
    vis[raw == 1] = [255, 0, 0]
    vis[raw == -1] = [0, 0, 255]
    return vis
