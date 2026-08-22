import json
import os
from math import ceil
from os import makedirs
from os.path import basename, dirname, join

import matplotlib.pyplot as plt
import numpy as np
from absl import logging
from absl.logging import error, info, warning
from absl.testing import absltest
from matplotlib.colors import PowerNorm
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift


def visualize_matrix_with_histogram(
    matrix,
    title="",
    cmap="bwr",
    hist_bins=256,
    clip_percentile_low=0.01,
    clip_percentile_high=99.99,
    filename=None,
    islog=False,
):
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Input 'matrix' must be a 2D NumPy array.")

    true_min = matrix.min()
    true_max = matrix.max()

    hist_range_min = np.percentile(matrix.ravel(), clip_percentile_low)
    hist_range_max = np.percentile(matrix.ravel(), clip_percentile_high)

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    im_vmin, im_vmax = np.percentile(matrix.ravel(), (clip_percentile_low, clip_percentile_high))
    im = axs[0].imshow(np.clip(matrix, im_vmin, im_vmax), cmap=cmap)
    cbar = fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
    cbar.set_label(f"Intensity", rotation=270, labelpad=15)
    axs[0].set_title(title, fontsize=14)
    axs[0].axis("on")
    axs[1].hist(
        matrix.ravel(),
        bins=hist_bins,
        color="skyblue",
        edgecolor="black",
        range=(hist_range_min, hist_range_max),
        density=True,
    )

    axs[1].set_title(f"Intensity Histogram\n(Total Range: [{true_min:.2e}, {true_max:.2e}])", fontsize=14)
    axs[1].set_xlabel(f"Visual Range [{hist_range_min:.2e}, {hist_range_max:.2e}]", fontsize=10)

    axs[1].grid(axis="y", alpha=0.75)
    if islog:
        axs[1].set_yscale("log")
    fig.suptitle(title, fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95] if title else [0, 0, 1, 1])

    if filename:
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {filename}")
    else:
        plt.show()
