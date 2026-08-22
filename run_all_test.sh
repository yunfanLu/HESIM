#!/bin/bash
set -e
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$(mktemp -d)"
python - <<'PY'
import importlib

CORE = [
    "hesim", "hesim.io", "hesim.meta", "hesim.visualization",
    "hesim.calibration.aps_calibrator",
    "hesim.calibration.aps_black_level_calibration",
    "hesim.calibration.aps_noise_calibration_with_multiple_groups",
    "hesim.calibration.aps_noise_calibration_with_multiple_groups_fitting",
    "hesim.calibration.aps_calibrator_data_config",
    "hesim.calibration.aps_calibrator_json_to_tex",
    "hesim.calibration.evs_calibrator",
    "hesim.calibration.evs_params_fitting",
    "hesim.calibration.evs_shot_noise_calibration",
    "hesim.calibration.evs_dark_noise_calibration",
    "hesim.calibration.evs_dark_noise_params_estimation",
    "hesim.calibration.evs_shot_noise_preprocessing",
    "hesim.calibration.evs_noise_visualization",
    "hesim.calibration.evs_shot_noise_visualization",
    "hesim.calibration.evs_calibrator_data_config",
    "hesim.calibration.color_checker",
    "hesim.calibration.white_balance_calibration",
    "hesim.hisp.main_hisp",
    "hesim.hisp.black_level_currection",
    "hesim.hisp.white_balance",
    "hesim.hisp.denoise",
    "hesim.hisp.auto_ccm_color_correction",
    "hesim.hisp.auto_color_correction",
    "hesim.hisp.color_correction",
    "hesim.hisp.auto_white_balance_by_grayness_index",
    "hesim.hisp.white_balance_calibration",
    "hesim.hisp.color_checker",
    "hesim.simulator.application",
    "hesim.simulator.sRGB_to_RAW",
    "hesim.simulator.evs_noise_generate",
    "hesim.simulator.rolling_shutter_blur_simulation",
    "hesim.simulator.hesim_for_hkust_3200fps_video_dataset",
    "hesim.simulator.calibration_config",
]
OPTIONAL = [
    "hesim.nr_iqa_video_evaluator",
]

failed = []
for m in CORE:
    try:
        importlib.import_module(m)
        print(f"OK   {m}")
    except Exception as e:
        failed.append((m, e))
        print(f"FAIL {m}: {e}")

for m in OPTIONAL:
    try:
        importlib.import_module(m)
        print(f"OK   {m}")
    except (ImportError, AttributeError) as e:
        print(f"SKIP {m}: unavailable optional dependency ({e})")
    except Exception as e:
        failed.append((m, e))
        print(f"FAIL {m}: {e}")

if failed:
    print(f"\n{len(failed)} module(s) FAILED (real errors, not missing data):")
    for m, e in failed:
        print(f"  {m}: {e}")
    raise SystemExit(1)
print("\nAll CORE modules imported OK; OPTIONAL modules OK or SKIP (missing heavy deps).")
PY
