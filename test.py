# import library
import unittest
import numpy as np
import opencl_matmul as mm
import opencl_matsum as ms
import opencl_matdif as md

N = 1000
FP32 = True
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

# Test classes


class Test_molt(unittest.TestCase):

    # test the moltiplication function
    def test_molt(self):

        numpyMatrix_mult = A@B
        m1 = mm.matmul(A, B, N, N, N, FP32)
        errore = np.sum(np.subtract(m1, numpyMatrix_mult))/(N*N)

        self.assertLess(errore, 1)

    def test_sum(self):

        numpyMatrix_sum = np.add(A, B)
        sum = ms.matsum(A, B, N, N, FP32)
        errore = np.sum(np.subtract(numpyMatrix_sum, sum))/(N*N)

        self.assertLess(errore, 1)

    def test_diff(self):

        numpyMatrix_sum = np.add(A, -B)
        diff = md.matdiff(A, B, N, N, FP32)
        errore = np.sum(np.subtract(numpyMatrix_sum, diff))/(N*N)

        self.assertLess(errore, 0.0001)


# driver code
if __name__ == '__main__':

    unittest.main()
