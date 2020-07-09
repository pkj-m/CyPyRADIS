import cupy as cp
import numpy as np

## tell Cython what the struct looks like:
cdef struct s_intPair:
    int    a
    int    b

ctypedef s_intPair intPair
    
def gpu_add(x1,x2,int a,int b):

    cuda_code = r'''
    extern "C"{
        
        //also tell the CUDA compiler what the struct looks like:
        struct intPair{
            int    a;
            int    b;
            };
        
        //declare the constant memory variables:
        __device__ __constant__ intPair const_struct_d;
        __device__ __constant__ int const_float_d;
        __device__ __constant__ float const_float_arr_d[10];
        
        __global__ void my_add(const float* x1, const float* x2, float* y) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            y[tid] = x1[tid] + x2[tid];
            y[tid] *= const_float_arr_d[0];
            y[tid] += const_float_arr_d[1];
        }
    }
    '''
        
    module = cp.RawModule(code = cuda_code)
    add_kernel = module.get_function('my_add')

    x1_d = cp.asarray(x1)
    x2_d = cp.asarray(x2)
    y_d = cp.zeros((5, 5), dtype=cp.float32)

    ## Transfer float array h2d using constant memory
    memptr = module.get_global("const_float_arr_d")
    const_float_arr_h = cp.ndarray((10,), cp.float32, memptr)
    const_float_arr_h[0] = a
    const_float_arr_h[1] = b

    ## Transfer struct h2d using constant memory
    cdef intPair const_struct_h
    const_struct_h.a = a
    const_struct_h.b = b


    add_kernel((5,), (5,), (x1_d, x2_d, y_d)) 

    return cp.asnumpy(y_d)