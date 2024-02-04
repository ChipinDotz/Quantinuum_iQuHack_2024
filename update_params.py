from scipy.special import ive, i0e, i1e
import numpy as np

kappa_max = 500

def bessel_ratio(kappa):
    return np.where(
        kappa < kappa_max, i1e(kappa) / i0e(kappa), 1 - 0.5 / kappa
    )

def _inverse_bessel_ratio_small(
    r
):
    return (
        2 * r
        - r**3
        - 1 / 6 * r**5
        - 1 / 24 * r**7
        + 1 / 360 * r**9
        + 53 / 2160 * r**11
    ) / (1 - r**2)


def _inverse_bessel_ratio_large(
    r
):
    return 1 / (2 * (1 - r) - (1 - r) ** 2 - (1 - r) ** 3)


def inverse_bessel_ratio(r):
    # https://dl.acm.org/doi/pdf/10.1145/355945.355949
    ibr = np.where(
        r < 0.8, _inverse_bessel_ratio_small(r), _inverse_bessel_ratio_large(r)
    )
    return np.where(r < bessel_ratio(kappa_max), ibr, 0.5 / (1 - r))

def _update_single_von_mises(
    mu, kappa, m, k, beta, error_rate
):
    gamma_shift = beta + k * mu + m * np.pi
    cos_gamma_shift = np.cos(gamma_shift) * (1 - error_rate)
    sin_gamma_shift = np.sin(gamma_shift) * (1 - error_rate)

    I0 = i0e(kappa)

    k = np.array(k, dtype=float)

    norm_const = (1 + cos_gamma_shift * ive(k, kappa) / I0) / 2

    A_km1 = ive(k - 1, kappa) / I0
    A_kp1 = ive(k + 1, kappa) / I0

    m1 = (
        bessel_ratio(kappa)
        + 0.5 * cos_gamma_shift * (A_km1 + A_kp1)
        + 1.0j * 0.5 * sin_gamma_shift * (A_kp1 - A_km1)
    ) / (norm_const * 2)

    posterior_mu_shift, posterior_kappa = _circular_mean_to_von_mises(m1)

    posterior_kappa = np.where(posterior_kappa < 0, kappa, posterior_kappa)
    return posterior_mu_shift + mu, posterior_kappa

def _circular_mean_to_von_mises(circular_m1):
    circular_mean = np.angle(circular_m1)
    one_minus_circular_var = np.abs(circular_m1)
    approximate_kappa = inverse_bessel_ratio(one_minus_circular_var)
    return circular_mean, approximate_kappa

def update_params(mu, sigma, meas, M, theta, error_rate=0):
    kappa = 1/sigma**2
    new_mu, new_kappa = _update_single_von_mises(mu, kappa, meas, M, M*theta, error_rate)
    #print(new_mu, new_kappa)
    new_sigma = 1/np.sqrt(new_kappa)
    return new_mu, new_sigma