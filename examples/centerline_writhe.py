"""Measure existing synthetic centerlines; no output files or GUI required."""

import numpy as np

from tubular import TubULAR


def run():
    tubi = TubULAR(time_points=[2, 5, 9], opts={"timeInterval": 2, "timeUnits": "min", "t0": 2})
    theta = np.linspace(0, 2.3*np.pi, 41)
    for time, pitch in ((2, 0.5), (5, 0.6), (9, 0.8)):
        curve = np.column_stack((pitch*theta, 1.2*np.sin(theta), 0.7*np.cos(theta)))
        tubi.register_centerline(time, curve)
    wr, density, rate, length, curves = tubi.measureWrithe({"filter_curve": 0, "omit_endpts": 0})
    print("Time (min):", length["times"])
    print("Full centerline length (um):", length["lengths"])
    print("Length rate (um/min):", length["dl"])
    print("Polygonal Levitt writhe:", wr["Levitt"])
    print("Sampled Gauss writhe:", wr["Gauss"])
    print("Gauss convention:", length["gauss_convention"])
    print("Polar writhe and curve/temporal smoothing are not yet implemented.")


if __name__ == "__main__":
    run()
