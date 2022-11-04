# import library
import unittest
import numpy as np
import opencl_matmul as mm

N = 10000
FP32 = True

# create a class


class Test_test(unittest.TestCase):

    # define a function
    def test_xxxxxxx(self):

        data = [1000, 2000, 3000]
        result = sum(data)
        self.assertEqual(result, 6000)


class Test_molt(unittest.TestCase):

    # define a function
    def test_molt(self):

        data = [1000, 2000, 3000]
        A = np.random.rand(N//3, N).astype(np.float32)
        B = np.random.rand(N, N//2).astype(np.float32)
        C = A@B
        m1 = mm.matmul(A, B, N//3, N, N//2, FP32)
        print(m1)
        errore = np.sum(np.subtract(m1, C))/(N*N)

        self.assertLess(errore, 1)


# driver code
if __name__ == '__main__':

    unittest.main()
