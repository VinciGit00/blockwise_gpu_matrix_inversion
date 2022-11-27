import pyopencl as cl
import numpy as np
import os
import warnings

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '0'
warnings.filterwarnings("ignore")


# https://cnugteren.github.io/tutorial/pages/page4.html

def matdiff(matrix1, matrix2, M, N, fp32):

    # OpenCL Setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Buffers
    if fp32:
        out_matrix = np.random.rand(M, N).astype(np.float32)
    else:
        out_matrix = np.random.rand(M, N)

    mf = cl.mem_flags
    A = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=matrix1)
    B = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=matrix2)
    C = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=out_matrix)

    # Kernels Creation
    # NB: max local mem size: 65536 byte (for each workgroup) == 90 FP64 values
    # M = larghezza, N = altezza
    if fp32:
        prog = cl.Program(ctx,  """
                                __kernel void matdiff(__global float* A, __global float* B, __global float* C, int M, int N){
                                    size_t row = get_global_id(0);
                                    size_t col = get_global_id(1);
                                    
                                    int size = get_global_size(1);

                                    C[row*size + col] = A[row*size + col]  - B[row*size + col];
                                }
                                """).build()

    # Kernel Execution
    pp = prog.matdiff

    pp.set_args(A, B, C, np.int32(M), np.int32(N))
    # queue, kernel, global dims, local dims, offset
    res = cl.enqueue_nd_range_kernel(queue, pp, [M, N], None, None)
    queue.finish()

    # Lettura risultato finale
    cl.enqueue_copy(queue, out_matrix, C)
    queue.finish()

    return out_matrix
