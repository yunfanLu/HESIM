import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace

__all__ = ["auto_wb_gi", "auto_wb_gi_multi"]


def auto_wb_gi(
    rgb: np.ndarray, top_percent: float = 0.001, sigma: float = 1.0, eps: float = 1e-4
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    rgb : (H,W,3) float32/float64, **linear** RGB in [0, 1].
    top_percent, sigma, eps :  Grayness-Index hyper-parameters.

    Returns
    -------
    rgb_balanced : same shape/order as input, float32 ∈ [0, 1].
    illum        : (3,) unit-norm illumination estimate that was applied.
    """
    illum, _, _ = _estimate_illumination(
        rgb.astype(np.float64),
        top_percent=top_percent,
        sigma=sigma,
        eps=eps,
    )
    balanced = _white_balance(rgb, illum)
    return balanced.astype(np.float32), illum.astype(np.float32)


def auto_wb_gi_multi(
    rgb: np.ndarray,
    clusters: int = 3,
    top_percent: float = 0.1,
    sigma: float = 1.0,
    eps: float = 1e-4,
    smooth_sigma: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    rgb : (H,W,3) float, linear RGB in [0, 1].
    clusters        : number of K-means centres (illuminants) to use.
    top_percent …   : hyper-parameters – see `grayness_index.py`.

    Returns
    -------
    rgb_balanced : (H,W,3) float32
    illum_map    : (H,W,3) float32 unit vectors for every pixel
    """
    illum_map, _ = _estimate_multi_illumination(
        rgb.astype(np.float64),
        clusters=clusters,
        top_percent=top_percent,
        sigma=sigma,
        eps=eps,
        smooth_sigma=smooth_sigma,
    )
    balanced = _white_balance(rgb, illum_map)
    return balanced.astype(np.float32), illum_map.astype(np.float32)


def _top_percent_mask(arr, percent):
    """Mask of elements belonging to the lowest `percent` values of arr."""
    flat = arr.ravel()
    k = max(1, int(np.ceil(percent * flat.size)))
    thresh = np.partition(flat, k - 1)[k - 1]
    return arr <= thresh


def _kmeans(data, k=2, max_iter=50, seed=0):
    """Minimal k‑means (Lloyd) on NumPy (Euclidean)."""
    n, dim = data.shape
    rng = np.random.default_rng(seed)
    centers = data[rng.choice(n, k, replace=False)]
    for _ in range(max_iter):
        dists = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        labels = dists.argmin(1)
        new_centers = np.vstack([data[labels == i].mean(0) if np.any(labels == i) else centers[i] for i in range(k)])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers, labels


def compute_gi(rgb, sigma=1.0, eps=1e-4):
    """Compute Grayness Index (GI) map and valid‑pixel mask.
    Parameters
    ----------
    rgb : ndarray(H,W,3)
        Linear RGB in [0,1].
    sigma : float
        LoG sigma.
    eps : float
        Contrast threshold for valid pixels.
    Returns
    -------
    GI : ndarray(H,W)
    valid_mask : ndarray(bool)
    """
    rgb = np.clip(rgb.astype(np.float64), 1e-6, 1.0)
    log_rgb = np.log(rgb)
    luminance = rgb.sum(2) + 1e-6
    log_lum = np.log(luminance)
    delta_r = gaussian_laplace(log_rgb[..., 0] - log_lum, sigma)
    delta_b = gaussian_laplace(log_rgb[..., 2] - log_lum, sigma)
    GI = np.sqrt(delta_r**2 + delta_b**2)

    contrast_r = gaussian_laplace(rgb[..., 0], sigma)
    contrast_g = gaussian_laplace(rgb[..., 1], sigma)
    contrast_b = gaussian_laplace(rgb[..., 2], sigma)
    valid = (np.abs(contrast_r) > eps) & (np.abs(contrast_g) > eps) & (np.abs(contrast_b) > eps)
    GI = np.where(valid, GI, np.inf)
    return GI, valid


def _estimate_illumination(rgb, top_percent=0.001, sigma=1.0, eps=1e-4):
    """Estimate global illumination color (unit vector)."""
    GI, valid = compute_gi(rgb, sigma, eps)
    mask = _top_percent_mask(GI, top_percent) & valid
    if not np.any(mask):
        raise RuntimeError("No gray pixels detected; adjust parameters.")
    illum = rgb[mask].mean(0)
    illum /= np.linalg.norm(illum) + 1e-12
    return illum, mask, GI


def _estimate_multi_illumination(rgb, clusters=3, top_percent=0.1, sigma=1.0, eps=1e-4, smooth_sigma=5.0):
    """Pixel‑wise illumination estimation for multi‑illuminant scenes."""
    GI, valid = compute_gi(rgb, sigma, eps)
    graymask = _top_percent_mask(GI, top_percent) & valid
    ys, xs = np.nonzero(graymask)
    if len(xs) < clusters:
        raise ValueError("Insufficient gray pixels; reduce clusters or increase top_percent.")
    samples = rgb[ys, xs]
    centers, labels = _kmeans(samples, clusters)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12

    H, W = rgb.shape[:2]
    illum_map = np.zeros((H, W, 3), float)
    for m in range(clusters):
        mask_m = labels == m
        if not np.any(mask_m):
            continue
        binary = np.zeros((H, W), float)
        binary[ys[mask_m], xs[mask_m]] = 1.0
        weights = gaussian_filter(binary, smooth_sigma)
        if weights.max() > 0:
            weights /= weights.max()
        illum_map += weights[..., None] * centers[m]
    illum_map /= np.linalg.norm(illum_map, axis=2, keepdims=True) + 1e-12
    return illum_map, centers


def _white_balance(rgb, illumin):
    """Apply diagonal white‑balance given illumination (global or per‑pixel)."""
    gains = illumin.mean(-1, keepdims=True) / (illumin + 1e-12)
    balanced = rgb * gains
    balanced /= balanced.max()
    return np.clip(balanced, 0, 1)
