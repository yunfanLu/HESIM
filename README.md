# H-ESIM: Hybrid Event–Frame Sensor Modeling, Calibration, and Simulation

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#installation)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

Official implementation accompanying **“Hybrid Event–Frame Sensors: Modeling, Calibration, and Simulation”** (ECCV 2026).

Hybrid event–frame sensors integrate an Active Pixel Sensor (APS) and an Event Vision Sensor (EVS) on one chip. H-ESIM provides the code underpinning the paper's unified statistical noise model, APS/EVS calibration pipeline, and simulator for jointly producing noisy RAW frames and event data.

Project page: <https://yunfanlu.github.io/HESIM>

## Paper and scope

The paper makes three connected contributions:

1. **Unified hybrid-sensor noise model.** APS and EVS noise are described with photon shot noise, dark-current noise, fixed-pattern noise, quantization noise, and layout-dependent effects. The EVS model uses the statistical Q-function to connect event probability, illumination, dark current, and the event threshold.
2. **Quantitative calibration.** Multi-exposure APS calibration estimates dark-current and fixed-pattern terms plus a CFA-position-dependent variance model. EVS calibration uses inverse-Q regression to estimate the six-parameter event-noise model and a dark-event map.
3. **H-ESIM simulator.** Given a high-frame-rate input sequence and calibrated parameters, the simulator generates APS RAW frames through an sRGB-to-RAW inverse pipeline and generates EVS events through a calibrated log-difference trigger model.

The evaluated devices are two Quad-Bayer hybrid sensors:

| Sensor | APS / EVS layout in the paper | EVS resolution |
| --- | --- | --- |
| GEN2 | One white EVS pixel per Quad-Bayer block | 816 × 612 |
| Eiger | Four color-filtered EVS pixels per Quad-Bayer block | 1632 × 1224 |


## Repository layout

```text
H-ESIM-Release/
├── hesim/                         # Library code; see hesim/ARCHITECTURE.md
│   ├── calibration/                # Contribution 2: APS and EVS parameter estimation
│   ├── hisp/                       # NumPy-oriented RAW-to-sRGB open ISP
│   ├── simulator/                  # Contribution 3: calibrated APS and EVS simulation
│   ├── io.py                       # RAW/EVS readers and acquisition-layout helpers
│   ├── meta.py                     # CFA masks and sRGB transfer helpers
│   ├── visualization.py            # Shared calibration visualizations
│   └── nr_iqa_video_evaluator.py   # Downstream no-reference video-IQA evaluation
├── scripts/                        # Dataset-dependent calibration/simulation entry points
├── requirements.txt                # Python dependencies
├── run_all_test.sh                 # Import-level smoke-test entry point
└── LICENSE
```

## Code architecture

The package follows the paper's calibration-to-simulation flow:

```text
captured APS RAW + EVS data
              │
              ▼
     hesim.calibration
     APS: dark/FPN + variance coefficients (β_a)
     EVS: inverse-Q parameters + dark-event map (β_e)
              │
              ▼
       calibrated meta artifacts
              │
       ┌──────┴───────────────────────┐
       ▼                              ▼
hesim.simulator                 hesim.hisp
sRGB → noisy APS RAW            RAW → sRGB reference ISP
frames → calibrated EVS events
```

| Area | Main modules | Responsibility | Paper connection |
| --- | --- | --- | --- |
| Shared I/O | `io.py`, `meta.py`, `visualization.py` | Read APS/EVS payloads, interpret sensor layouts, provide CFA/gamma helpers and plots. | Shared support for calibration and simulation. |
| APS calibration | `calibration/aps_black_level_calibration.py`, `aps_noise_calibration_with_multiple_groups.py`, `aps_noise_calibration_with_multiple_groups_fitting.py`, `aps_calibrator.py` | Estimate dark-current/fixed-pattern components and the six-term, CFA-dependent APS variance model `β_a`. | APS calibration in the paper. |
| EVS calibration | `calibration/evs_shot_noise_preprocessing.py`, `evs_shot_noise_calibration.py`, `evs_dark_noise_calibration.py`, `evs_params_fitting.py`, `evs_calibrator.py` | Compute event statistics, estimate dark-event behavior, and fit the inverse-Q event model `β_e`. | EVS calibration in the paper. |
| Open ISP | `hisp/main_hisp.py`, `black_level_currection.py`, `white_balance.py`, `quad_bayer_demosaic_by_trans_quad2bayer.py`, color-correction modules | Provide a NumPy-oriented reference path from calibrated RAW data to sRGB. The historical filename `black_level_currection.py` is retained for import compatibility. | Companion RAW-to-sRGB ISP described with the simulator. |
| APS simulation | `simulator/sRGB_to_RAW.py` | Invert sRGB processing, mosaic to CFA/Quad-Bayer RAW, add calibrated fixed/dynamic noise, then quantize. | APS branch of H-ESIM. |
| EVS simulation | `simulator/evs_noise_generate.py` | Apply calibrated thresholds and noise parameters to the log-difference event-trigger model. | EVS branch of H-ESIM and the Q-function model. |
| Joint orchestration | `simulator/application.py`, `hesim_for_hkust_3200fps_video_dataset.py`, `rolling_shutter_blur_simulation.py` | Coordinate APS and EVS generation over high-frame-rate videos and prepare rolling-shutter data. | Joint H-ESIM generation. |
| Evaluation | `nr_iqa_video_evaluator.py` | Compute no-reference metrics used for downstream visual-quality evaluation. | Deblurring evaluation in the paper. |

For a maintainer-oriented file-level map and dependency direction, see [`hesim/ARCHITECTURE.md`](hesim/ARCHITECTURE.md). Package-level docstrings in `hesim`, `hesim.calibration`, `hesim.hisp`, and `hesim.simulator` state each package's boundary.

## Installation

Python 3.9 or newer is required. A virtual environment is recommended:

```bash
git clone <YOUR-REPOSITORY-URL>
cd H-ESIM-Release
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch installation may need to be adapted to your CUDA platform; follow the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) when using GPU acceleration.

This release requires NumPy 1.x because the current `colour-demosaicing` dependency is not compatible with NumPy 2.x. The constraint is recorded in `requirements.txt`.

## Data and artifact contract

The repository intentionally excludes captured sensor data, calibration outputs, and evaluation datasets. The code can be installed and smoke-tested without them, but a full calibration or simulation run requires the following user-provided inputs.

| Workflow | Required input | Required location or argument | Produced artifact |
| --- | --- | --- | --- |
| APS calibration | Dark and colour-checker RAW captures organised by exposure | Paths in `hesim/calibration/aps_calibrator_data_config.py` | `meta/<sensor>_I_dt/` with `black_level_a_map.npy`, `black_level_b_map.npy`, and `betas_calibration_with_fit_noise_poly_gpu_per_cfa.npz` (`betas`, `terms`) |
| EVS calibration | Paired EVS captures and APS calibration output | Paths in `hesim/calibration/evs_calibrator_data_config.py` | `meta/<sensor>_EVS_SHOT_Noise/` with `all-calibration/evs_betas.npy` and `dark_event_noise_calibration.npz` (`positive_rate`, `negative_rate`) |
| Joint simulation | High-speed RGB frames plus both sets of calibrated artifacts | `--INPUT_DIR` and `--OUT_DIR` | Per video group: `APS/*_raw.npz` (`raw`) and `EVS/*.npz` (`events`) |

The high-speed input directory must have the layout `INPUT_DIR/<video_group>/<part>/<image files>`. Frame names are sorted lexicographically, so use zero-padded frame filenames. The simulator supports `EIGER` and `GEN2` only. It expects the calibration folders at the repository-relative paths declared in `hesim/simulator/calibration_config.py`; either create those folders from the calibration workflow or edit that configuration for your own artifact location.

## Using the code

### 1. Prepare calibrated artifacts

The calibration modules operate on real APS/EVS captures and write fitted artifacts such as dark-current maps, APS noise coefficients, EVS coefficients, and dark-event statistics. Configure the data locations in the `*_data_config.py` modules for your own captures.

### 2. Simulate paired APS RAW frames and EVS events

`hesim.simulator.HybridSensorSimulator` combines the APS (`sRGB_to_RAW.py`) and EVS (`evs_noise_generate.py`) branches. After placing calibration artifacts according to the table above, run:

```bash
python -m hesim.simulator.hesim_for_hkust_3200fps_video_dataset \
  --SENSOR=EIGER \
  --INPUT_DIR=/absolute/path/to/high_speed_frames \
  --OUT_DIR=/absolute/path/to/output \
  --EXPOSURE_TIME_MS=20
```

The output NPZ fields are named explicitly: APS files contain `raw`; EVS files contain `events`. The historical misspelled class name remains as a compatibility alias but should not be used in new code.

### 3. Process RAW with the open ISP

The `hesim.hisp` modules expose black-level correction, white balance, Quad-Bayer conversion/demosaicing, and color correction. They are designed as a reference implementation and expect compatible calibration artifacts when black-level correction is enabled.

## Verification

The release includes an import-level smoke test that never invokes calibration, simulation, or dataset processing:

```bash
bash run_all_test.sh
```

You can also validate Python syntax without executing data-dependent code:

```bash
python -m py_compile $(find hesim -name '*.py')
```

The smoke test checks import safety only. It does not establish numerical equivalence to the paper because the private captures and fitted artifacts are not part of this release. To reproduce reported results, obtain the exact captures, calibration artifacts, and downstream datasets described by the paper, then record the commit hash and hardware/CUDA versions used for the run.

## Limitations and reproducibility

The model is statistically calibrated for the sensors and capture regimes studied in the paper. As discussed in the paper, very low illumination, extreme temperatures, and bandwidth constraints are not explicitly modeled and may depart from the Gaussian approximation. Reproducing the data-dependent calibration or downstream experiments requires the corresponding sensor captures and evaluation datasets.

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{lu2026hesim,
  title     = {Hybrid Event--Frame Sensors: Modeling, Calibration, and Simulation},
  author    = {Lu, Yunfan and Messikommer, Nico and Xu, Xiaogang and Chen, Liming and Chen, Yuhan and Zubi{\'c}, Nikola and Scaramuzza, Davide and Xiong, Hui},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE).
