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

    dummy_array = np.arange(16, dtype=np.float32)
    
    cdef vector[float] vec


    
    test_arr = cp.zeros(16, dtype=cp.float32)

    print("test array before kernel: ")
    for i in range(16):
        print(test_arr[i], end = ", ")
    print()
    print("array struct looks as follows: ")
    print("arr1: ")
    for i in range(16):
        print(array_struct.arr1[i], end = ", ")
    print("\narr2: ")
    for i in range(16):
        print(array_struct.arr2[i], end = ", ")
    print()

    print("\n KERNEL LAUNCHED \n")
    
    cuda_code = r'''
    #include<cupy/complex.cuh>
    extern "C"{
        
        struct arrayStruct{
            float *arr1;
            float *arr2;
        };
        
        __device__ __constant__ arrayStruct array_struct;

        __global__ void my_add(float* dummy, int N) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < N){
                dummy[tid] = array_struct.arr1[tid] + array_struct.arr2[tid];
            }
        }
    }
    '''
        
    module = cp.RawModule(code = cuda_code)
    add_kernel = module.get_function('my_add')

    memptr = module.get_global("array_struct")
    struct_ptr = ctypes.cast(ctypes.pointer(array_struct),ctypes.c_void_p)
    struct_size = ctypes.sizeof(array_struct)
    print('sizeof p:', struct_size)
    memptr.copy_from_host(struct_ptr,struct_size)
    print('Done!')

    add_kernel((4,), (4,), (test_arr, 16))
    
    print("array after kernel: ")
    
    for i in range(16):
        print(test_arr[i], end = ", ")
    print()

    return