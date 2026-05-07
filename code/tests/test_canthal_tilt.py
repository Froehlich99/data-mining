import math
import unittest

import numpy as np

from scripts.process import compute_canthal_tilt


def point(x, y):
    return np.array([float(x), float(y)])


def offset(dx, angle_degrees):
    return point(dx, math.tan(math.radians(angle_degrees)) * dx)


class CanthalTiltTests(unittest.TestCase):
    def test_positive_when_outer_corners_are_higher(self):
        tilt = compute_canthal_tilt(
            l_outer=point(0, 0),
            l_inner=point(100, 10),
            r_inner=point(200, 10),
            r_outer=point(300, 0),
        )

        self.assertGreater(tilt, 0)

    def test_negative_when_outer_corners_are_lower(self):
        tilt = compute_canthal_tilt(
            l_outer=point(0, 10),
            l_inner=point(100, 0),
            r_inner=point(200, 0),
            r_outer=point(300, 10),
        )

        self.assertLess(tilt, 0)

    def test_shared_head_roll_does_not_change_signed_tilt(self):
        roll = 12.0
        expected_tilt = 4.5
        l_outer = point(0, 0)
        l_inner = l_outer + offset(100, roll + expected_tilt)
        r_inner = point(200, 0)
        r_outer = r_inner + offset(100, roll - expected_tilt)

        tilt = compute_canthal_tilt(l_outer, l_inner, r_inner, r_outer)

        self.assertAlmostEqual(tilt, expected_tilt, places=6)


if __name__ == "__main__":
    unittest.main()
