
import time
import os
import sys

import unittest
import numpy as np
from tensor import Tensor,compileTensors

class TestTensor(unittest.TestCase):
    def setUp(self):
        self.oldenv = os.getenv("TDB_TEST")
        os.environ["TDB_TEST"] = "1"
        self.files = []
    def tearDown(self):
        for f in self.files:
            os.unlink(f)

        if self.oldenv:
            os.environ["TDB_TEST"] = self.oldenv
        else:
            os.unsetenv("TDB_TEST")

    """
    OFFSET moved to compiler

    def test_anchor(self):
        cube = np.linspace(1,27,27).reshape((3,3,3))
        t = Tensor.new(cube,x=1,y=1,z=1)
        self.files += [t.path()]

        self.assertEqual(cube[1,1,1],t[2,2,2])
        self.assertEqual(cube[0:1,1,1],t[1:2,2,2])

    def test_offset(self):
        cube = np.linspace(1,27,27).reshape((3,3,3))
        t = Tensor.new(cube,z=1,y=1,z=1)
        c = t.get(x=(1,4),y=(1,2),z=(1,2))
        np.testing.assert_array_equal(cube[:,0,0],c)
        """

    def test_tensor(self):
        cube = np.linspace(1,27,27).reshape((3,3,3))
        t = Tensor.new(cube)
        self.files += [t.path()]

        self.assertEqual(cube[1,1,1],t[1,1,1])

    def test_id(self):
        cube = np.linspace(1,27,27).reshape((3,3,3))
        t = Tensor.new(cube)
        c = t.get()
        np.testing.assert_array_equal(cube,c)

    def test_getter(self):
        cube = np.linspace(1,27,27).reshape((3,3,3))
        t = Tensor.new(cube)
        c = t.get(x=(0,3),y=(0,2),z=(0,1))
        np.testing.assert_array_equal(cube[:,0:2,0],c)

    def test_compile(self):
        m = time.time()
        cubes = [np.ones((720,360,12)) for _ in range(750)]
        print(f"Init: {time.time()-m}")

        m = time.time()
        cubes = [Tensor.new(t) for t in cubes]
        print(f"Create: {time.time()-m}")

        m = time.time()
        hcube = compileTensors(cubes,(720,360,12),(0,720),(0,360),(0,100))
        print(f"Read: {time.time()-m}")
        print(hcube.flatten().shape)

        self.assertTrue(True)

