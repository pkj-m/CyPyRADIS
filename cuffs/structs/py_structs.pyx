# distutils: language=c++

from cpython cimport array
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

#-----------------------------------
#    hostData: host_params_h       #
#-----------------------------------

cdef int host_params_h_block_preparation_step_size
cdef int host_params_h_Min_threads_per_block
cdef int host_params_h_Max_threads_per_block

cdef float host_params_h_log_2vMm_min
cdef float host_params_h_log_2vMm_max

cdef vector[float] host_params_h_top_x
cdef vector[float] host_params_h_top_a
cdef vector[float] host_params_h_top_b
cdef vector[float] host_params_h_bottom_x
cdef vector[float] host_params_h_bottom_a
cdef vector[float] host_params_h_bottom_b

cdef int host_params_h_dec_size
cdef float* host_params_h_v0_size
cdef float* host_params_h_da_size

cdef int host_params_h_shared_size
#cdef cudaEvent_t host_params_h_start
#cdef cudaEvent_t host_params_h_stop
#cdef cudaEvent_t host_params_h_start_DLM
#cdef cudaEvent_t host_params_h_stop_DLM
cdef float host_params_h_elapsedTime
cdef float host_params_h_elapsedTimeDLM

#cdef cufftHandle host_params_h_plan_DLM
#cdef cufftHandle host_params_h_plan_spectrum

# device pointers
cdef float* host_params_h_v0_d
cdef float* host_params_h_da_d
cdef float* host_params_h_S0_d
cdef float* host_params_h_El_d
cdef float* host_params_h_log_2gs_d
cdef float* host_params_h_na_d
cdef float* host_params_h_log_2vMm_d
cdef float* host_params_h_DLM_d
cdef float* host_params_h_spectrum_d

#cdef cufftReal* host_params_h_DLM_d_in
#cdef cufftComplex* host_params_h_DLM_d_out

#cdef cufftComplex* host_params_h_spectrum_d_in
#cdef cufftReal* host_params_h_spectrum_d_out

#-----------------------------------

#-----------------------------------
#    initData: init_params_h       #
# ----------------------------------

# DLM spectral parameters
cdef float init_params_h_v_min
cdef float init_params_h_v_max
cdef float init_params_h_dv

# DLM sizes:
cdef int init_params_h_N_v
cdef int init_params_h_N_wG
cdef int init_params_h_N_wL
cdef int init_params_h_N_wG_x_N_wL
cdef int init_params_h_N_total

# work parameters:
cdef int init_params_h_Max_lines
cdef int init_params_h_N_lines
cdef int init_params_h_N_points_per_block
cdef int init_params_h_N_threads_per_block
cdef int init_params_h_N_blocks_per_grid
cdef int init_params_h_N_points_per_thread
cdef int init_params_h_Max_iterations_per_thread

cdef int init_params_h_shared_size_floats

# ---------------------------------


#-----------------------------------------
#       iterData: iter_params_h          #
#-----------------------------------------

# pressure and temperature
cdef float iter_params_h_p
cdef float iter_params_h_log_p
cdef float iter_params_h_hlog_T
cdef float iter_params_h_log_rT
cdef float iter_params_h_c2T
cdef float iter_params_h_rQ

# spectral parameters
cdef float iter_params_h_log_wG_min
cdef float iter_params_h_log_wL_min
cdef float iter_params_h_log_dwG
cdef float iter_params_h_log_dwL

cdef int iter_params_h_blocks_line_offset[4096]
cdef int iter_params_h_blocks_iv_offset[4096]

#------------------------------------------


#--------------------------------------------
#       spectralData: spec_h                #
#--------------------------------------------

cdef vector[float] spec_h_v0
cdef vector[float] spec_h_da
cdef vector[float] spec_h_S0
cdef vector[float] spec_h_El
cdef vector[float] spec_h_log_2vMm
cdef vector[float] spec_h_na
cdef vector[float] spec_h_log_2gs

#--------------------------------------------



def decimate(np.ndarray[np.float32_t, ndim=1] v_in, np.ndarray[np.float32_t, ndim=1] v_out, int step):
    print("Decimating ({0})...".format(step))
    idx_limit = int(len(v_in)/step)
    v_out.resize(idx_limit, refcheck=False)
    #v_out = [v_in[i * step] for i in range(v_in_len/step)]
    for i in range(idx_limit):
        v_out[i] = v_in[i*step]
    print("Done!")



















# cdef extern from "c_struct.c":
#     ctypedef struct spectralData:
#         float* v0
#         float* da
#         float* S0
#         float* El
#         float* log_2vMm
#         float* na
#         float* log_2gs

#     ctypedef struct initData:
#         float v_min
#         float v_max	# Host only
#         float dv

#         # DLM sizes:
#         int N_v
#         int N_wG
#         int N_wL
#         int N_wG_x_N_wL
#         int N_total

#         # Work parameters :
#         int Max_lines
#         int N_lines
#         int N_points_per_block
#         int N_threads_per_block
#         int N_blocks_per_grid
#         int N_points_per_thread
#         int	Max_iterations_per_thread

#         int shared_size_floats
    
#     ctypedef struct blockData:
#         int line_offset
#         # int N_iterations
#         int iv_offset
    
#     ctypedef struct iterData:
#         # Any info needed for the Kernel that does not change during 
#         # kernel execution but MAY CHANGE during spectral iteration step

#         # Pressure & temperature:
#         float p
#         # float T
#         float log_p
#         float hlog_T
#         float log_rT
#         float c2T
#         float rQ

#         # Spectral parameters:
#         float log_wG_min
#         float log_wL_min
#         float log_dwG
#         float log_dwL

#         # Block data:
#         blockData blocks[4096] 
    
#     spectralData init_spectralData()


# # # Then we describe a class that has a Point member "pt"
# cdef class py_spectralData:
#     cdef spectralData sd

#     def __init__(self):
#         self.sd = init_spectralData()
#         self.sd.v0 = NULL
#         self.sd.da = NULL
#         self.sd.S0 = NULL
#         self.sd.El = NULL
#         self.sd.log_2vMm = NULL
#         self.sd.na = NULL
#         self.sd.log_2gs = NULL

#     @v0.setter
#     def v0(self,val):
#     #    cdef float* v0_ptr = <float*> val.data
#         self.sd.v0 = <float *> val.data
    
#     # @property
#     # def v0(self):
#     #     cdef float* v0_ptr = <float*> self.sd.v0.data
#     #     return v0_ptr

# cdef extern from "point.c":
#     ctypedef struct Point:
#         int x
#         int y
#     Point make_and_send_point(int x, int y)

# cdef class PyPoint:
#     cdef Point p

#     def __init__(self, x, y):
#         self.p = make_and_send_point(x, y)

#      # define properties in the normal Python way
#     @property
#     def x(self):
#         return self.p.x

#     @x.setter
#     def x(self,val):
#         self.p.x = val

#     # @property
#     # def y(self):
#     #     return self.pt.y

#     # @y.setter
#     # def y(self,val):
#     #     self.pt.y = val
