"""Calibration routines for the paper's APS and EVS statistical models.

APS modules estimate dark-current and fixed-pattern components together with
the CFA-position-dependent variance coefficients ``beta_a``. EVS modules
derive event statistics from paired captures and fit the inverse-Q model
coefficients ``beta_e`` plus dark-event maps. Configuration modules contain
dataset-specific paths only; callers should replace them for their own data.
"""
