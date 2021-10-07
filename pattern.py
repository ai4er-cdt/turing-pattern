import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import numpy
import requests
import argparse


def update(phi, r0, r1):
    dt = 0.1
    p = gaussian_filter(phi, sigma=r0, mode='wrap')
    q = gaussian_filter(phi, sigma=r1, mode='wrap')
    u = dt*(q > p) - dt*(p > q)
#     u = dt*numpy.tanh(30*(q - p))
    phi += u

    # Normalise phi in range [-1, 1]
    phi = 2.0*(phi - phi.min())/phi.ptp() - 1.0

    return phi


def run():
    p = argparse.ArgumentParser()
    p.add_argument("--url", type=str, metavar="FILENAME", help="File name or URL")
    p.add_argument("-r", nargs=2, metavar=("r0", "r1"),
                   type=float, help="Gaussian radius")
    p.add_argument("-n", nargs="?", type=int, help="Iteration Num", default=10)
    args = p.parse_args()
    print(args.r, args.url, args.n)
    if args.r:
        r0, r1 = args.r
    else:
        r0, r1 = 5, 4
    if args.url:
        url = args.url
        phi = Image.open(requests.get(url, stream=True).raw)
        w, h = phi.size
        phi = phi.resize((w*4, h*4), Image.ANTIALIAS)
    else:
        phi = numpy.random.rand(400,400)

    phi = numpy.array(phi, dtype=numpy.float32)
    if len(phi.shape) == 3:
        phi = numpy.sum(phi, axis=2)

    plt.imshow(phi, cmap='gray')
    plt.show()
    print(phi.shape)

    # Run for a few steps
    for i in range(args.n):
        phi = update(phi, r0, r1)

    # Smooth the result a little
    phi = gaussian_filter(phi, sigma=2.0, mode='wrap')

    # plt.figure(figsize=(16,16))
    plt.imshow(phi, cmap='magma')
    plt.show()
