import numpy as np
import matplotlib.pyplot as plt
import requests
import argparse

from PIL import Image
from scipy.ndimage import gaussian_filter

from typing import Optional, Tuple


def update(phi: np.ndarray, r0: float, r1: float) -> np.ndarray:
    """
    Helper function to update the image to create the turing pattern
    Parameters
    ----------
    phi : np.ndarray,
        array of the image to use
    r0 : float,
        radius for the gaussian filter
    r1 : float,
        radius for the gaussian filter

    Returns
    -------
    phi: np.ndarray,
        updated image
    """
    dt = 0.1
    p = gaussian_filter(phi, sigma=r0, mode="wrap")
    q = gaussian_filter(phi, sigma=r1, mode="wrap")
    u = dt * (q > p) - dt * (p > q)
    phi += u

    # Normalise phi in range [-1, 1]
    phi = 2.0 * (phi - phi.min()) / phi.ptp() - 1.0

    return phi


def arg_parse():
    """function for argument parsing"""
    p = argparse.ArgumentParser()
    p.add_argument("--url", type=str, metavar="FILENAME", help="File name or URL")
    p.add_argument("-r", nargs=2, metavar=("r0", "r1"),
                   type=float, help="Gaussian radius")
    args = p.parse_args()
    print(args.r, args.url)
    return args


def run(r: Optional[Tuple[float, float]], url: Optional[str]) -> None:
    """
    Generating turing pattern from an image.

    Parameters
    ----------
    r : tuple of floats, optional,
        tuple of the radii for the gaussian filter

    url : str, optional,
        url or path of the image
        if None, random image is used
    """
    if r:
        r0, r1 = r
    else:
        r0, r1 = 5, 4

    if url:
        phi = Image.open(requests.get(url, stream=True).raw)
        w, h = phi.size
        phi = phi.resize((w * 4, h * 4), Image.ANTIALIAS)
    else:
        phi = np.random.rand(400, 400)

    phi = np.array(phi, dtype=np.float32)
    if len(phi.shape) == 3:
        phi = np.sum(phi, axis=2)

    plt.imshow(phi, cmap="gray")
    plt.show()
    print(phi.shape)

    # Run for a few steps
    for i in range(15):
        phi = update(phi, r0, r1)

    # Smooth the result a little
    phi = gaussian_filter(phi, sigma=2.0, mode="wrap")

    plt.imshow(phi, cmap="magma")
    plt.show()


if __name__ == "__main__":
    args = arg_parse()
    run(args.r, args.url)
