# `hesim` architecture

`hesim` is organized by responsibility, not by experiment. The dependencies
should flow from shared utilities and calibrated artifacts toward the runtime
simulation/ISP paths; data-dependent entry points remain at the outer edge.

```text
io.py + meta.py + visualization.py
              │
              ├── calibration/ ──► fitted meta artifacts
              │                         │
              ├── hisp/ ◄───────────────┤
              │                         │
              └── simulator/ ◄──────────┘
                         │
                         └── application.py / dataset drivers
```

## Boundaries

| Unit | Owns | Does not own |
| --- | --- | --- |
| `io.py` | File decoding, metadata parsing, APS/EVS folder discovery, batch loading. | Noise estimation or image processing policy. |
| `meta.py` | Sensor color masks and transfer helpers. | Dataset paths or fitted parameters. |
| `calibration/` | Estimation of APS `beta_a`, EVS `beta_e`, dark-event statistics, and diagnostic visualizations. | Bundled calibration captures; path constants are only user-editable presets. |
| `hisp/` | Reference RAW-to-sRGB processing primitives and their composition. | Event generation or calibration fitting. |
| `simulator/` | Calibrated sRGB-to-RAW conversion, EVS event triggering, and synchronized sequence generation. | Acquiring data or estimating new calibration parameters. |
| `nr_iqa_video_evaluator.py` | No-reference video-IQA measurement for downstream outputs. | Sensor simulation or calibration. |

## Reading guide

1. Start with `meta.py` and `io.py` to understand layouts and data formats.
2. Read `calibration/aps_calibrator.py` and `calibration/evs_calibrator.py`
   for the two parameter-estimation pipelines.
3. Read `simulator/sRGB_to_RAW.py` and
   `simulator/evs_noise_generate.py` for the two simulation branches.
4. Use `simulator/application.py` to see how both branches are coordinated;
   use `hisp/main_hisp.py` for the complementary RAW-to-sRGB reference path.