from __future__ import print_function
import cupy as cp
import numpy as np
cimport numpy as np
import ctypes
import sys

# Python declaration of the struct:
class floatPair(ctypes.Structure):
    _fields_=[("a",ctypes.c_float),
              ("b",ctypes.c_float)]

class arrayStruct(ctypes.Structure):
    _fields_=[
        ("arr1", ctypes.POINTER(ctypes.c_float)),
        ("arr2", ctypes.POINTER(ctypes.c_float))
    ]

array_struct = arrayStruct()

              
def gpu_add():
    global array_struct
    # length of array = 16

    dummy_array = np.arange(16, dtype=np.float32)
    dummy_array2 = np.ones(16, dtype=np.float32)

    array_struct.arr1 = (ctypes.c_float * len(dummy_array))(*dummy_array)
    array_struct.arr2 = (ctypes.c_float * len(dummy_array2))(*dummy_array2)

    arr1_d = cp.array(array_struct.arr1)
    arr2_d = cp.array(array_struct.arr2)

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
        
        __global__ void my_add(float* dummy, float *arr1, float *arr2, int N) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < N){
                dummy[tid] = arr1[tid] + arr2[tid];
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

    add_kernel((4,), (4,), (test_arr, arr1_d, arr2_d, 16))
    
    print("array after kernel: ")
    
    for i in range(16):
        print(test_arr[i], end = ", ")
    print()

    return