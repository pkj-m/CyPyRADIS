#distutils: language=c++

from __future__ import print_function
import numpy as np
import cupy as cp
import scipy
import sys
from libcpp.vector cimport vector

loaded_from_source = r'''
#include <cupy/complex.cuh>
extern "C"{
__global__ void test_sum(complex<float>* y, complex<float>* rx, complex<float>* ix, float A, int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N){
        y[tid] = rx[tid] + A * ix[tid];
    }
}


}'''

################## GPU PART ######################

# module = cp.RawModule(code=loaded_from_source)
# ker_sum = module.get_function('test_sum')


# rx = cp.arange(25, dtype=cp.complex64)
# ix = 1j* cp.arange(25, dtype=cp.complex64)
# y = None
# y = cp.cuda.runtime.malloc(sys.getsizeof(cp.complex64() * 25))
# cp.cuda.runtime.memset(y, 0, 25 * sys.getsizeof(cp.complex64)) 
# cdef float a = 5
# a_d = cp.float64(a)

# ker_sum((5,), (5,), (y, rx, ix, a_d, 25))

# y_h = y.get()

# cdef vector[float] real_device_part

# real_device_part = y.get().real

# print("output = ")
# print(y_h)

# print("real device part = ")
# print(real_device_part)
# print()

#################################################

################ Cython Part ####################


cdef float add(float a, float b):
    #print(a+b)
    return  a+b


cdef vector[float] host_params_h_v0_dec

# cdef int prepare_blocks():

#     # ----------- setup global variables -----------------
#     global host_params_h_v0_dec
#     global host_params_h_da_dec
#     global host_params_h_dec_size

#     global iter_params_h_p
#     global iter_params_h_blocks_iv_offset
#     global iter_params_h_blocks_line_offset

#     global init_params_h_block_preparation_step_size
#     global init_params_h_N_points_per_block
#     global init_params_h_N_threads_per_block
#     global init_params_h_dv
#     global init_params_h_Max_iterations_per_thread
#     global init_params_h_v_min
#     global init_params_h_dv
#     #------------------------------------------------------

#     cdef float* v0 = host_params_h_v0_dec
#     cdef float* da = host_params_h_da_dec

#     cdef float v_prev
#     cdef float dvdi
#     cdef int i = 0
#     cdef int n = 0
#     cdef int step = host_params_h_block_preparation_step_size

#     # in lieu of blockData struct, create new arrays
#     cdef int new_block_line_offset
#     cdef int iv_offset

#     cdef float v_cur = v0[0] + iter_params_h_p * da[0]
#     cdef float v_max = v_cur + init_params_h_N_points_per_block * init_params_h_dv
#     cdef int i_max = init_params_h_Max_iterations_per_thread
    
#     new_block_line_offset = 0
#     new_block_iv_offset = int(((v_cur - init_params_h_v_min) / init_params_h_dv))


#     while 1:
#         i += step
#         if i > host_params_h_dec_size:
#             iter_params_h_blocks_line_offset[n] = new_block_line_offset
#             iter_params_h_blocks_iv_offset[n] = new_block_iv_offset
#             n+=1
#             new_block_line_offset = i * init_params_h_N_threads_per_block
#             iter_params_h_blocks_line_offset[n] = new_block_line_offset
#             iter_params_h_blocks_iv_offset[n] = new_block_iv_offset
#             break
        
#         v_prev = v_cur
#         v_cur = v0[i] + iter_params_h_p * da[i]
        
#         if ((v_cur > v_max) or (i >= i_max)) : 
#             if (v_cur > v_max) : 
#                 dvdi = (v_cur - v_prev) / float(step)
#                 i -= int(((v_cur - v_max) / dvdi)) + 1
#                 v_cur = v0[i] + iter_params_h_p * da[i]
            
#             iter_params_h_blocks_line_offset[n] = new_block_line_offset
#             iter_params_h_blocks_iv_offset[n] = new_block_iv_offset
#             n+=1
#             new_block_iv_offset = int(((v_cur - init_params_h_v_min) / init_params_h_dv))
#             new_block_line_offset = i * init_params_h_N_threads_per_block
#             v_max = v_cur + (init_params_h_N_points_per_block) * init_params_h_dv
#             i_max = i + init_params_h_Max_iterations_per_thread
    
#     return n


def start():
    a = 10
    b = 20

    # check if we can return a value and store it in a cdef var: CONFIRMED
    cdef float x
    x = add(a, b)
    print('result = {0}'.format(x))

    # check if a pointer to float can be set to a numpy array: DOES NOT WORK
    # if we try to set a vector equal to a nupy array, it works
    sz = 20
    test_array = np.arange(sz, dtype=np.float32)
    global host_params_h_v0_dec
    host_params_h_v0_dec = test_array
    print("original array: ", end="")
    print(test_array, end="\n")
    print("pointer to array: ", end="")
    for i in range(sz):
        print(host_params_h_v0_dec[i], end=" ")
    print(end="\n")
    print("size of vector = {0}".format(host_params_h_v0_dec.size()), end="\n")




# cdef vector[float] c_pointer

# np_array = np.arange(10, dtype=np.float32)

# print("pure numpy array = ", np_array)

# c_pointer = np_array

# print("printing array as float*: ")
# for i in c_pointer:
#     print(i, end=" ")
# print(end="\n")





code = '''
__global__ void test_sum(float* out_real, float* out_img, float* rx, float* ix,int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N){
        cufftComplex* cp = make_cuComplex(rx[tid], ix[tid]);
        // do processing on this element
        out_real[tid] = cp.x;
        out_img[tid] = cp.y;
    }
}
'''

