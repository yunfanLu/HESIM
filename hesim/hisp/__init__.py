"""Reference open ISP for H-ESIM RAW data.

This package contains composable NumPy-oriented operators for black-level
correction, white balance, Quad-Bayer conversion and demosaicing, denoising,
and color correction. ``main_hisp`` combines these operators into a reference
RAW-to-sRGB path; calibration artifacts are required only by the stages that
explicitly load them.
"""
