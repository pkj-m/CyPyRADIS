#distutils: language=c++
from __future__ import print_function
import cupy as cp
import numpy as np
cimport numpy as np
import ctypes
from libcpp.vector cimport vector
import sys

              
def gpu_add():
    global array_struct
    # length of array = 16

    dummy_np_array = np.arange(16, dtype=np.float32)
    test_array_h = np.zeros(16, dtype=np.float32)
    test_array = cp.array(test_array_h)
    
    # THIS WORKS DO NOT USE 'CDEF VECTOR[FLOAT] DIRECTLY'
    cdef np.ndarray[dtype=np.float32_t, ndim=1] vec = dummy_np_array

    vec_d = cp.array(vec)

    print("test array before kernel: ")
    for i in range(16):
        print(test_array[i], end = ", ")
    print()
    print("vec looks as follows: ")
    for i in range(16):
        print(vec[i], end = ", ")
    print()
    print("vec_d looks as follows: ")
    for i in range(16):
        vec_d[i] = 10*i
        print(vec_d[i], end = ", ")
    print()

    print("\n KERNEL LAUNCHED \n")
    cnt = cp.int32(0)

    cuda_code = r'''
    extern "C"{
        
        __global__ void my_add(float* dummy, float *vec_d, int N) {
            for (int i = 0; i < N; i++){
                dummy[i] = vec_d[i];
            }
        }
    }
    '''
        
    module = cp.RawModule(code = cuda_code)
    add_kernel = module.get_function('my_add')

    add_kernel((1,), (1,), (test_array, vec_d, 16))
    test_array_h = test_array.get()
    print("array after kernel: ")
    
    for i in range(16):
        print(test_array_h[i], end = ", ")
    print()

    return