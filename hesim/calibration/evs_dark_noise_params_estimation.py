import math

from scipy.stats import norm

invQ = norm.isf
Qfunc = norm.sf


def estimate_sigma_delta(P_on, P_off, theta=0.75e-3):
    """
    Parameters
    ----------
    P_on, P_off : float
        Per-sample ON / OFF event probabilities.
    theta : float, optional
        Nominal symmetric threshold in volts (default 0.75 mV).

    Returns
    -------
    sigma : float
        Estimated σ (standard deviation of the differential noise) in volts.
    delta : float
        Estimated threshold bias δ in volts.
    """

    S = 0.5 * (P_on + P_off)
    z = invQ(S)
    sigma = theta / (math.sqrt(2) * z)

    D = P_on - P_off
    phi = math.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    delta = (D * math.sqrt(2) * sigma) / (2 * phi)

    return sigma, delta


if __name__ == "__main__":

    P_on = 0.00046367493983795406
    P_off = 0.00036714119793047114
    sigma_est, delta_est = estimate_sigma_delta(P_on, P_off)
    print(f"σ ≈ {sigma_est*1e6:7.1f} µV")
    print(f"δ ≈ {delta_est*1e6:7.2f} µV")
    print(f"θ_ON ≈ {(0.75e-3-delta_est)*1e3:6.3f} mV")
    print(f"θ_OFF ≈ {(0.75e-3+delta_est)*1e3:6.3f} mV")
