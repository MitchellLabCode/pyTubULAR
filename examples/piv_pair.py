"""Known periodic particle displacement with explicit PIV validity flags."""

import numpy as np
from scipy.ndimage import gaussian_filter

from tubular import TubULAR


def main():
    texture = gaussian_filter(np.random.default_rng(781).random((32,32)),1,mode="wrap")
    a = np.tile(texture,(3,3))
    b = np.roll(a,(-2,3),axis=(0,1))
    result = TubULAR().measurePIV2d(a,b,window_size=32,step=16,dt=.5)
    print(f"Valid vectors: {result.valid.sum()}/{result.valid.size}")
    print(f"Median displacement: ({np.nanmedian(result.dx):.6g}, {np.nanmedian(result.dy):.6g}) pixels")
    print(f"Median velocity: ({np.nanmedian(result.u):.6g}, {np.nanmedian(result.v):.6g}) pixels/time unit")
    print("Positive y points down in the image; no physical calibration is inferred.")


if __name__ == "__main__":
    main()
