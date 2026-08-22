"""H-ESIM library package.

The package mirrors the pipeline in *Hybrid Event--Frame Sensors: Modeling,
Calibration, and Simulation*: shared I/O and sensor-layout utilities feed the
calibration package; its fitted artifacts parameterize the simulator; and the
``hisp`` package provides a reference RAW-to-sRGB ISP.

Subpackages:
    calibration: APS and EVS noise-parameter estimation from captured data.
    hisp: Reference image-signal-processing operators for calibrated RAW data.
    simulator: Joint APS-RAW and EVS-event synthesis from calibrated models.
"""
