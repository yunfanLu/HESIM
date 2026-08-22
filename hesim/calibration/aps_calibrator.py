import json
import os
from math import ceil
from os import makedirs
from os.path import basename, dirname, join

import matplotlib.pyplot as plt
import numpy as np
from absl import app, logging
from absl.logging import error, info, warning
from absl.testing import absltest
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift

from hesim.calibration.aps_black_level_calibration import (
    calibrate_black_level,
    calibrate_black_level_with_multiple_exposures,
    fixed_noise_split_to_row_noise_and_black_level_noise,
)
from hesim.calibration.aps_noise_calibration_with_multiple_groups import _exposure_time_relation_with_intensity
from hesim.calibration.aps_noise_calibration_with_multiple_groups_fitting import (
    fit_noise_poly_gpu_per_cfa,
    fit_noise_second_order_gpu,
    fit_noise_second_order_gpu_without_negative_remove,
    fit_without_row_interaction,
    fit_without_row_interaction_not_negative,
    plot_var_mean_scatter_by_channel,
    var_prediction_by_fit_noise_poly_gpu_per_cfa,
)
from hesim.calibration.color_checker import _quad_bayer_map
from hesim.io import aps_raw_stack, read_raw
from hesim.visualization import visualize_matrix_with_histogram


def aps_black_noise_calibration(exp_dirs, out_dir):
    a_map, b_map, T, stk = calibrate_black_level_with_multiple_exposures(exp_dirs, out_dir)
    row_noise, blc_noise = fixed_noise_split_to_row_noise_and_black_level_noise(b_map)

    E, H, W = stk.shape
    for i in range(E):
        t = T[i]
        intensity = stk[i, :, :]
        visualize_matrix_with_histogram(
            matrix=intensity,
            title=f"Intensity w {t} ms",
            filename=join(out_dir, f"black_level_intensity_w_{t}.png"),
        )

    visualize_matrix_with_histogram(
        matrix=a_map,
        title="N_DP",
        filename=join(out_dir, "black_level_a_map"),
    )
    np.save(join(out_dir, "black_level_a_map.npy"), a_map)
    visualize_matrix_with_histogram(
        matrix=b_map,
        title="N_FP",
        filename=join(out_dir, "black_level_b_map"),
    )
    np.save(join(out_dir, "black_level_b_map.npy"), b_map)

    row_noise = np.repeat(row_noise[:, np.newaxis], a_map.shape[1], axis=1)
    visualize_matrix_with_histogram(
        matrix=row_noise,
        title="N_Row",
        filename=join(out_dir, "black_level_row_noise"),
    )
    np.save(join(out_dir, "black_level_row_noise.npy"), row_noise)
    visualize_matrix_with_histogram(
        matrix=blc_noise,
        title="N_BL",
        filename=join(out_dir, "black_level_blc_noise"),
    )
    np.save(join(out_dir, "black_level_blc_noise.npy"), blc_noise)


def aps_dynamic_noise_calibration_shlop_to_vars(slope, vars_, times, out_dir):
    means_wo_bais = slope * times[:, None, None]
    G = means_wo_bais.shape[0]
    makedirs(out_dir, exist_ok=True)

    for g in range(G):
        exposure_time = int(times[g])
        mean_wi_bais = means_wo_bais[g]
        var_g = vars_[g]
        visualize_matrix_with_histogram(
            mean_wi_bais,
            title=rf"$\Delta t$={exposure_time}, Clean Intensity",
            filename=join(out_dir, f"{exposure_time}_clearn_intensity.png"),
            clip_percentile_low=1,
            clip_percentile_high=99,
        )
        visualize_matrix_with_histogram(
            var_g,
            title=rf"$\Delta t$={exposure_time}, Measured Var",
            filename=join(out_dir, f"{exposure_time}_measure_var.png"),
            clip_percentile_low=1,
            clip_percentile_high=99,
        )

    betas, terms = fit_noise_poly_gpu_per_cfa(means_wo_bais, vars_, times, cfa_size=4, poly_order=2)
    info(betas.shape)
    info("Feature order (μ^i * t^j), {terms}")

    items_length = len(terms)
    for i in range(4):
        for j in range(4):
            info(f"Var(CFA: [{i}, {j}])")
            for k in range(items_length):
                P, O = terms[k]
                info(f"      +  Mu^{P} * T^{O} * {betas[i, j, k]:.8f}")

    betas_to_save = {
        "description": "Polynomial fit of pixel variance as a function of illumination (Mu) and exposure time (T)",
        "cfa_size": [4, 4],
        "poly_order": 2,
        "terms": [{"O": O, "P": P} for (O, P) in terms],
        "betas": {},
    }
    items_length = len(terms)
    for i in range(4):
        for j in range(4):
            key = f"CFA_{i}_{j}"
            pixel_terms = []
            for k in range(items_length):
                P, O = terms[k]
                coeff = float(betas[i, j, k])
                pixel_terms.append({"term": f"Mu^{P} * T^{O}", "O": int(O), "P": int(P), "coefficient": coeff})
            betas_to_save["betas"][key] = pixel_terms
    with open(join(out_dir, "noise_fit_results.json"), "w") as f:
        json.dump(betas_to_save, f, indent=4)
    info("Saved polynomial fit results to noise_fit_results.json")

    np.savez_compressed(
        join(out_dir, "betas_calibration_with_fit_noise_poly_gpu_per_cfa.npz"),
        betas=betas,
        terms=terms,
    )

    for g in range(G):
        mu = means_wo_bais[g]
        t = times[g]
        var_hat = var_prediction_by_fit_noise_poly_gpu_per_cfa(betas, terms, mu, t)
        visualize_matrix_with_histogram(
            var_hat,
            title=f"Var Prediction by Fit Noise Poly GPU Per CFA Group {g}",
            filename=join(out_dir, f"var_group_{g}_hat.png"),
            clip_percentile_low=1,
            clip_percentile_high=99,
        )
        visualize_matrix_with_histogram(
            vars_[g],
            title=f"Var Group {g}",
            filename=join(out_dir, f"var_group_{g}.png"),
            clip_percentile_low=1,
            clip_percentile_high=99,
        )

    plot_var_mean_scatter_by_channel(
        means=means_wo_bais,
        vars_=vars_,
        times=times,
        betas4x4=betas,
        terms=terms,
        save_dir=out_dir,
        num_samples=100_000,
    )


def aps_dynamic_noise_calibration(exp_dirs, out_dir):
    means, vars_, times = [], [], []
    for t, folder in exp_dirs.items():
        t = int(t)
        stk = aps_raw_stack(folder, t, with_blc=False)
        stk = np.stack(stk, 0)
        means.append(stk.mean(0))
        var = stk.var(0, ddof=0)
        vars_.append(var)
        times.append(t)
        info(f"Processed {folder} with exposure time {t} ms, shape: {stk.shape}")
    means = np.stack(means)
    G, H, W = means.shape
    vars_ = np.stack(vars_)
    times = np.asarray(times, np.float32)
    np.savez_compressed(
        join(out_dir, f"means_var_times.npz"),
        means=means,
        vars_=vars_,
        times=times,
    )
    T = np.asarray(times, np.float32)
    slope, intercept = _exposure_time_relation_with_intensity(means, times, out_dir, with_blc=False, plot_vis=True)

    np.save(join(out_dir, "aps_slope.npy"), slope)
    visualize_matrix_with_histogram(slope, title=f"aps_slope", filename=join(out_dir, "aps_slope.png"))
    np.save(join(out_dir, "aps_intercept.npy"), intercept)
    visualize_matrix_with_histogram(intercept, title=f"aps_intercept", filename=join(out_dir, "aps_intercept.png"))
    aps_dynamic_noise_calibration_shlop_to_vars(slope, vars_, T, out_dir)


def main(calibration_data, out_dir):
    DARK_NOISE, DYNAMIC_NOISE = True, True
    if DARK_NOISE:
        dark_exp_dirs = calibration_data["dark_exp_dirs"]
        aps_black_noise_calibration(dark_exp_dirs, out_dir)
    if DYNAMIC_NOISE:
        dynamic_exp_dirs = calibration_data["dynamic_exp_dirs"]
        aps_dynamic_noise_calibration(dynamic_exp_dirs, out_dir)


def run(args):
    from hesim.calibration.aps_calibrator_data_config import (
        eiger_dark_exp_dirs,
        eiger_dynamic_exp_dirs,
        gen2_dark_exp_dirs,
        gen2_dynamic_exp_dirs,
    )

    EIGER, GEN2 = True, False

    if EIGER:
        calibration_data = {
            "dark_exp_dirs": eiger_dark_exp_dirs,
            "dynamic_exp_dirs": eiger_dynamic_exp_dirs,
        }
        out_dir = "meta/ARGB_ERGB_Eiger_I_dt"
        main(calibration_data, out_dir)

    if GEN2:
        gen2_calibration_data = {
            "dark_exp_dirs": gen2_dark_exp_dirs,
            "dynamic_exp_dirs": gen2_dynamic_exp_dirs,
        }
        out_dir = "meta/ARGB_EW_GEN2_I_dt"
        makedirs(out_dir, exist_ok=True)
        main(gen2_calibration_data, out_dir)


if __name__ == "__main__":
    app.run(run)
