import csv
import json
from os import makedirs
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from absl import app
from absl.logging import info
from scipy.stats import norm
from tqdm import tqdm

from hesim.calibration.evs_noise_visualization import plot_count_eveny_frame, positive_vs_negative_rate_scatter
from hesim.calibration.evs_params_fitting import evs_calibrate
from hesim.io import evs_raw_stack, find_evs_folder, read_evs_raw, sorted_raws
from hesim.visualization import visualize_matrix_with_histogram

DARK_VISUALIZATION = False


def eiger_average_pool(quad):
    H, W = quad.shape
    raw = np.zeros([H // 2, W // 2], dtype=np.float32)

    raw[0::2, 0::2] = quad[0::4, 0::4] + quad[0::4, 1::4] + quad[1::4, 0::4] + quad[1::4, 1::4]
    raw[0::2, 1::2] = quad[0::4, 2::4] + quad[0::4, 3::4] + quad[1::4, 2::4] + quad[1::4, 3::4]
    raw[1::2, 0::2] = quad[2::4, 0::4] + quad[2::4, 1::4] + quad[3::4, 0::4] + quad[3::4, 1::4]
    raw[1::2, 1::2] = quad[2::4, 2::4] + quad[2::4, 3::4] + quad[3::4, 2::4] + quad[3::4, 3::4]
    raw /= 4.0
    return raw


def gen2_average_pool(quad):
    H, W = quad.shape

    wr: float = 0.299
    wg: float = 0.587
    wb: float = 0.114

    raw = np.zeros([H // 2, W // 2], dtype=np.float32)
    raw[0::2, 0::2] = quad[0::4, 0::4] + quad[0::4, 1::4] + quad[1::4, 0::4] + quad[1::4, 1::4]
    raw[0::2, 1::2] = quad[0::4, 2::4] + quad[0::4, 3::4] + quad[1::4, 2::4] + quad[1::4, 3::4]
    raw[1::2, 0::2] = quad[2::4, 0::4] + quad[2::4, 1::4] + quad[3::4, 0::4] + quad[3::4, 1::4]
    raw[1::2, 1::2] = quad[2::4, 2::4] + quad[2::4, 3::4] + quad[3::4, 2::4] + quad[3::4, 3::4]
    raw /= 4.0
    raw[0::2, 0::2] = raw[0::2, 0::2] / wg
    raw[0::2, 1::2] = raw[0::2, 1::2] / wr
    raw[1::2, 0::2] = raw[1::2, 0::2] / wb
    raw[1::2, 1::2] = raw[1::2, 1::2] / wg
    raw_w = (raw[0::2, 0::2] + raw[1::2, 0::2] + raw[0::2, 1::2] + raw[1::2, 1::2]) / 4.0
    return raw_w


def dark_event_noise_calibration(Eiger_EVS_Dark_Event_Folder_List, outdir):
    dark_stk = []
    for dark_folder in Eiger_EVS_Dark_Event_Folder_List:
        stk = evs_raw_stack(dark_folder, 2000)
        dark_stk.append(stk)
    dark_stk = np.concatenate(dark_stk, axis=0)

    N, H, W = dark_stk.shape
    event_generate_rate = np.abs(dark_stk).sum(axis=0) / N
    positive_rate = (dark_stk > 0).sum(axis=0) / N
    negative_rate = (dark_stk < 0).sum(axis=0) / N
    no_event_rate = 1 - event_generate_rate

    savename = join(outdir, "dark_event_noise_calibration.npz")
    np.savez(
        savename, event_generate_rate=event_generate_rate, positive_rate=positive_rate, negative_rate=negative_rate
    )

    visualize_matrix_with_histogram(
        event_generate_rate, title="dark_event_generate_rate", filename=join(outdir, "dark_event_generate_rate.png")
    )
    visualize_matrix_with_histogram(
        positive_rate, title="dark_positive_rate", filename=join(outdir, "dark_positive_rate.png")
    )
    visualize_matrix_with_histogram(
        negative_rate, title="dark_negative_rate", filename=join(outdir, "dark_negative_rate.png")
    )
    return event_generate_rate, positive_rate, negative_rate


def EVSCalibration(APS_Calibrated_Folder, EVS_calibrated_Folder, EVS_Dark_Event_Folder_List, sensor):
    aps_intensity_file = join(APS_Calibrated_Folder, "aps_slope.npy")
    aps_slope = np.load(aps_intensity_file)
    if sensor == "eiger":
        aps_slope = eiger_average_pool(aps_slope)
    elif sensor == "gen2":
        aps_slope = gen2_average_pool(aps_slope)
    print(f"Sensor: {sensor}, slope {aps_slope.shape}")
    evs_posibility_file = join(EVS_calibrated_Folder, "evs_shot_noise.npz")
    evs_posibility = np.load(evs_posibility_file)
    positive_rate = evs_posibility["positive_rate"]
    evs_rate = evs_posibility["evs_rate"]
    negative_rate = evs_posibility["negative_rate"]
    print(f"positive_rate:{positive_rate.shape}, negative_rate:{negative_rate.shape}, evs_rate:{evs_rate.shape}")
    total_frames = evs_posibility["total_frames"]

    visualize_matrix_with_histogram(aps_slope, title="I_c", filename=join(EVS_calibrated_Folder, "I_c.png"))
    visualize_matrix_with_histogram(
        evs_rate,
        title="P_e",
        filename=join(EVS_calibrated_Folder, "P_e.png"),
        clip_percentile_low=1,
        clip_percentile_high=90,
    )

    all_folder = join(EVS_calibrated_Folder, "all-calibration")
    evs_betas = evs_calibrate(aps_slope, evs_rate, all_folder, N=total_frames, sensor=sensor)
    print(f"Finish EVS beta calibration")

    dark_event_generate_rate, dark_positive_rate, dark_negative_rate = dark_event_noise_calibration(
        EVS_Dark_Event_Folder_List, EVS_calibrated_Folder
    )
    light_positive = positive_rate - dark_positive_rate
    light_negative = negative_rate - dark_negative_rate
    light_evs = evs_rate - dark_event_generate_rate
    visualize_matrix_with_histogram(
        light_positive, title="light_positive", filename=join(EVS_calibrated_Folder, "light_positive.png")
    )
    visualize_matrix_with_histogram(
        light_negative, title="light_negative", filename=join(EVS_calibrated_Folder, "light_negative.png")
    )
    visualize_matrix_with_histogram(light_evs, title="light_evs", filename=join(EVS_calibrated_Folder, "light_evs.png"))

    light_positive_e = np.power(10, light_positive)
    light_negative_e = np.power(10, light_negative)
    light_evs_e = np.power(10, light_evs)
    visualize_matrix_with_histogram(
        light_positive_e, title="light_positive_e", filename=join(EVS_calibrated_Folder, "light_positive_e.png")
    )
    visualize_matrix_with_histogram(
        light_negative_e, title="light_negative_e", filename=join(EVS_calibrated_Folder, "light_negative_e.png")
    )
    visualize_matrix_with_histogram(
        light_evs_e, title="light_evs_e", filename=join(EVS_calibrated_Folder, "light_evs_e.png")
    )


def main(args):

    from hesim.calibration.evs_calibrator_data_config import (
        Eiger_APS_Calibrated_Folder,
        Eiger_EVS_Calibrated_Folder,
        Eiger_EVS_Dark_Event_Folder_List,
    )

    EVSCalibration(Eiger_APS_Calibrated_Folder, Eiger_EVS_Calibrated_Folder, Eiger_EVS_Dark_Event_Folder_List, "eiger")

    from hesim.calibration.evs_calibrator_data_config import (
        Gen2_APS_Calibrated_Folder,
        Gen2_EVS_Calibrated_Folder,
        Gen2_EVS_Dark_Event_Folder_List,
    )

    EVSCalibration(Gen2_APS_Calibrated_Folder, Gen2_EVS_Calibrated_Folder, Gen2_EVS_Dark_Event_Folder_List, "gen2")


if __name__ == "__main__":
    app.run(main)
