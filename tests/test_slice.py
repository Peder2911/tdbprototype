
from slicing import shift_slice 
import numpy as np
import unittest

class SliceTests(unittest.TestCase):

    def test_slice_shift(self):
        orig = slice(0,1,1)
        shifted = shift_slice(orig,1)
        self.assertEqual(shifted.start,1)
        self.assertEqual(shifted.stop,2)

    def test_int_slice(self):
        shifted = shift_slice(0,1)
        self.assertEqual(shifted,1)

    def test_on_tensor(self):
        cube = np.linspace(1,27,27).reshape((3,3,3))
        np.testing.assert_array_equal(
                cube[1,:,:],
                cube[shift_slice(0,1),:,:]
            )

    def test_on_list(self):
        ls = np.linspace(1,10,10)
        for i in range(1,4):
            np.testing.assert_array_equal(
                    ls[0+i:4+i:2],
                    ls[shift_slice(slice(0,4,2),i)]
                )
