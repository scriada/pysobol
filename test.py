import sobol
import unittest

class Tests(unittest.TestCase):

    def test_init(self):
        s = sobol.isobol(dims=3)
        self.assertEqual(s.dims(), 3)
        self.assertEqual(s.size(), None)

    def test_init_finite(self):
        s = sobol.isobol(dims=3, size=4)
        self.assertEqual(s.dims(), 3)
        self.assertEqual(s.size(), 4)

    def test_isobol(self):
        expected_s = [[0.5, 0.5],
                      [0.75, 0.25],
                      [0.25, 0.75],
                      [0.375, 0.375],
                      [0.875, 0.875]]
        s = sobol.isobol(dims=2, size=5)

        for s1, s2 in zip(s, expected_s):
            self.assertEqual(s1[0], s2[0])
            self.assertEqual(s1[1], s2[1])

    def test_sobol(self):
        expected_s = [[0.5, 0.5],
                      [0.75, 0.25],
                      [0.25, 0.75],
                      [0.375, 0.375],
                      [0.875, 0.875]]
        s = sobol.sobol(dims=2, size=5)

        for s1, s2 in zip(s, expected_s):
            self.assertEqual(s1[0], s2[0])
            self.assertEqual(s1[1], s2[1])
