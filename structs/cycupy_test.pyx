from __future__ import print_function
import cupy as cp
import numpy as np
cimport numpy as np
import ctypes
import sys
#from libcpp.vector cimport vector

# Python declaration of the struct:
class floatPair(ctypes.Structure):    
    _fields_=[("a",ctypes.c_float),
              ("b",ctypes.c_float)]

class floatPairPair(ctypes.Structure):
    _fields_=[
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("p", floatPair)
    ]

class cArray(ctypes.Structure):
    _fields_=[
        ("arr", floatPair * 25)
    ]

class spectraldata(ctypes.Structure):
    _fields_=[
        ("arr1", ctypes.POINTER(ctypes.c_int)),
        ("arr2", ctypes.POINTER(ctypes.c_int))
    ]
              
def gpu_add(x1,x2,int a,int b, int x, int y, l):

    cuda_code = r'''
    extern "C"{
        
        //also tell the CUDA compiler what the struct looks like:
        typedef struct floatPair{
            float   a;
            float   b;
            };
        
        typedef struct floatPairPair{
            float   x;
            float   y;
            floatPair  p;
        };

        typedef struct cArray{
            int N;
            int arr[25];
        };
        
        //declare the constant memory variables:
        __device__ __constant__ floatPair const_struct_d;
        __device__ __constant__ floatPairPair const_struct_struct_d;
        __device__ __constant__ cArray const_array_d;
        
        __global__ void my_add(const float* x1, const float* x2, float* y) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            y[tid] = x1[tid] + x2[tid];
            
            //y[tid] *= const_struct_d.a;
            //y[tid] += const_struct_d.b;

            //y[tid] = const_array_d.arr[tid];

            y[tid] += const_struct_struct_d.x;
            y[tid] += const_struct_struct_d.y;
            y[tid] += const_struct_struct_d.p.a;
            y[tid] *= const_struct_struct_d.p.b;
        }
    }
    '''
        
    module = cp.RawModule(code = cuda_code)
    add_kernel = module.get_function('my_add')

    x1_d = cp.asarray(x1)
    x2_d = cp.asarray(x2)
    y_d = cp.zeros((5, 5), dtype=cp.float32)

    

    # testArray = cArray()
    
    # for i in range(25):
    #     testArray.arr[i] = floatPair(i, 2*i)

    # for i in range(25):
    #     print(testArray.arr[i].a, testArray.arr[i].b)

    # memptr = module.get_global("const_array_d")

    # struct_ptr = ctypes.cast(ctypes.pointer(testArray),ctypes.c_void_p)
    # struct_size = ctypes.sizeof(testArray)
    # print('sizeof p:', struct_size)

    # #memptr.copy_from_host(struct_ptr,struct_size)
    # print('Done!')

    #--------------------------------------------------------------#


    ## Transfer struct h2d using constant memory
    # params = floatPair(a, b)
    # params.a = a
    # params.b = b

    # params = floatPairPair()
    # params.x = x
    # params.y = y
    # params.p = floatPair(a, b)

    # memptr = module.get_global("const_struct_struct_d")

    # struct_ptr = ctypes.cast(ctypes.pointer(params),ctypes.c_void_p)
    # struct_size = ctypes.sizeof(params)
    # print('sizeof p:', struct_size)

    # memptr.copy_from_host(struct_ptr,struct_size)
    # print('Done!')

    #-----------------------------------------------------------------#

    
    # Transfer struct h2d using constant memory
    arrs = spectraldata()
    # cdef vector[int] vec
    # vec.push_back(1)
    # vec.push_back(2)
    # vec.push_back(3)

    vec = np.arange(3, dtype=np.int32)

    #cdef np.ndarray[np.int32_t, ndim=1] S = np.arange(10,dtype=np.int32)
    #S = [0 for i in range(10)]
    arrs.arr1 = (ctypes.c_int * len(vec))(*vec)

    for i in range(3):
        print(arrs.arr1[i], end = " ")


    memptr = module.get_global("const_struct_struct_d")

    struct_ptr = ctypes.cast(ctypes.pointer(arrs),ctypes.c_void_p)
    struct_size = ctypes.sizeof(arrs)
    struct_size1 = ctypes.sizeof(arrs.arr1)
    print('sizeof p:', struct_size)
    print('sizeof p.arr: ', struct_size1)

    #memptr.copy_from_host(struct_ptr,struct_size)
    print('Done!')



    #add_kernel((5,), (5,), (x1_d, x2_d, y_d)) 

    return cp.asnumpy(y_d)