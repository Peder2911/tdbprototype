
import numpy as np
import unittest

from geometry import containedby

class TestGeometry(unittest.TestCase):
    """
    Basic geometric actions used to handle n-dimensional tensors
    """

    def test_contains(self):
        a = np.ones((3,3,3))
        b = np.ones((6,6,6))
        c = np.ones((9,9,9))
        self.assertTrue(containedby(a,b))
        self.assertFalse(containedby(c,b))
