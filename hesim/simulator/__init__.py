"""Calibrated hybrid APS and EVS simulation components.

``sRGB_to_RAW`` implements the APS inverse pipeline: inverse display mapping,
sensor-color conversion, CFA mosaicing, calibrated noise injection, and ADC
quantization. ``evs_noise_generate`` implements the calibrated EVS
log-difference trigger model. ``application`` coordinates both branches over
high-frame-rate input sequences.
"""
