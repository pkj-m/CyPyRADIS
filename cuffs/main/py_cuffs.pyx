# distutils: language=c++

from cpython cimport array
import numpy as np
cimport numpy as np
import sys
from libcpp.vector cimport vector

cdef  float epsilon = 0.0001

#-----------------------------------
#    hostData: host_params_h       #
#-----------------------------------

cdef  int host_params_h_block_preparation_step_size
cdef  int host_params_h_Min_threads_per_block
cdef  int host_params_h_Max_threads_per_block

cdef  float host_params_h_log_2vMm_min
cdef  float host_params_h_log_2vMm_max

cdef  vector[float] host_params_h_top_x
cdef  vector[float] host_params_h_top_a
cdef  vector[float] host_params_h_top_b
cdef  vector[float] host_params_h_bottom_x
cdef  vector[float] host_params_h_bottom_a
cdef  vector[float] host_params_h_bottom_b

cdef  int host_params_h_dec_size
cdef  float* host_params_h_v0_size
cdef  float* host_params_h_da_size

cdef  int host_params_h_shared_size
#cdef  cudaEvent_t host_params_h_start
#cdef  cudaEvent_t host_params_h_stop
#cdef  cudaEvent_t host_params_h_start_DLM
#cdef  cudaEvent_t host_params_h_stop_DLM
cdef  float host_params_h_elapsedTime
cdef  float host_params_h_elapsedTimeDLM

#cdef  cufftHandle host_params_h_plan_DLM
#cdef  cufftHandle host_params_h_plan_spectrum

# device pointers
cdef  float* host_params_h_v0_d
cdef  float* host_params_h_da_d
cdef  float* host_params_h_S0_d
cdef  float* host_params_h_El_d
cdef  float* host_params_h_log_2gs_d
cdef  float* host_params_h_na_d
cdef  float* host_params_h_log_2vMm_d
cdef  float* host_params_h_DLM_d
cdef  float* host_params_h_spectrum_d

#cdef  cufftReal* host_params_h_DLM_d_in
#cdef  cufftComplex* host_params_h_DLM_d_out

#cdef  cufftComplex* host_params_h_spectrum_d_in
#cdef  cufftReal* host_params_h_spectrum_d_out

#-----------------------------------

#-----------------------------------
#    initData: init_params_h       #
# ----------------------------------

# DLM spectral parameters
cdef  float init_params_h_v_min
cdef  float init_params_h_v_max
cdef  float init_params_h_dv

# DLM sizes:
cdef  int init_params_h_N_v
cdef  int init_params_h_N_wG
cdef  int init_params_h_N_wL
cdef  int init_params_h_N_wG_x_N_wL
cdef  int init_params_h_N_total

# work parameters:
cdef  int init_params_h_Max_lines
cdef  int init_params_h_N_lines
cdef  int init_params_h_N_points_per_block
cdef  int init_params_h_N_threads_per_block
cdef  int init_params_h_N_blocks_per_grid
cdef  int init_params_h_N_points_per_thread
cdef  int init_params_h_Max_iterations_per_thread

cdef  int init_params_h_shared_size_floats

# ---------------------------------


#-----------------------------------------
#       iterData: iter_params_h          #
#-----------------------------------------

# pressure and temperature
cdef  float iter_params_h_p
cdef  float iter_params_h_log_p
cdef  float iter_params_h_hlog_T
cdef  float iter_params_h_log_rT
cdef  float iter_params_h_c2T
cdef  float iter_params_h_rQ

# spectral parameters
cdef  float iter_params_h_log_wG_min
cdef  float iter_params_h_log_wL_min
cdef  float iter_params_h_log_dwG
cdef  float iter_params_h_log_dwL

cdef  int iter_params_h_blocks_line_offset[4096]
cdef  int iter_params_h_blocks_iv_offset[4096]

#------------------------------------------


#--------------------------------------------
#       spectralData: spec_h                #
#--------------------------------------------

cdef  vector[float] spec_h_v0
cdef  vector[float] spec_h_da
cdef  vector[float] spec_h_S0
cdef  vector[float] spec_h_El
cdef  vector[float] spec_h_log_2vMm
cdef  vector[float] spec_h_na
cdef  vector[float] spec_h_log_2gs

#--------------------------------------------



####################################
######## KERNELS COME HERE #########
####################################











####################################


cdef void set_pT(float p, float T):
    
    # ----------- setup global variables -----------------
    global iter_params_h_p
    global iter_params_h_log_p
    global iter_params_h_hlog_T
    global iter_params_h_log_rT
    global iter_params_h_c2T
    global iter_params_h_rQ
    #------------------------------------------------------

    cdef float c2 = 1.4387773538277204
    iter_params_h_p = p
    iter_params_h_log_p = log(p)
    iter_params_h_hlog_T = 0.5 * log(T)
    iter_params_h_log_rT = log(296.0/T)
    iter_params_h_c2T = -c2/T

    cdef float B = 0.39
    cdef float w1 = 1388.0
    cdef float w2 = 667.0
    cdef float w3 = 1349.0

    cdef int d1 = 1
    cdef int d2 = 2
    cdef int d3 = 1

    cdef float Q2 = T(c2 * B)
    cdef float Qv1 = 1 / pow(1 - exp(-c2 * w1 / T), d1);
	cdef float Qv2 = 1 / pow(1 - exp(-c2 * w2 / T), d2);
	cdef float Qv3 = 1 / pow(1 - exp(-c2 * w3 / T), d3);
	cdef float Q = Qr * Qv1 * Qv2 * Qv3;

    iter_params_h_rQ = 1 / Q
    iter_params_h_rQ = iter_params_h_rQ / T



cdef void init_lorentzian_params(void):

    # ----------- setup global variables -----------------
    global host_params_h_top_x
    global host_params_h_bottom_x
    global host_params_h_top_a
    global host_params_h_bottom_a
    global host_params_h_top_b
    global host_params_h_bottom_b
    global iter_params_h_log_rT
    global iter_params_h_log_p
    global iter_params_h_log_wL_min
    global iter_params_h_log_wL_max
    global epsilon
    #------------------------------------------------------
    
    cdef float log_wL_min
    cdef float log_wL_max

    for i in range(host_params_h_bottom_x.size()):
        if iter_params_h_log_rT < host_params_h_bottom_x[i]:
            log_wL_min = iter_params_h_log_rT * host_params_h_bottom_a[i] + host_params_h_bottom_b[i]  + iter_params_h_log_p
            break
    
    for i in range(host_params_h_top_x.size()):
        if iter_params_h_log_rT < host_params_h_top_x[i]:
            log_wL_max = iter_params_h_log_rT * host_params_h_top_a[i] + host_params_h_top_b[i]  + iter_params_h_log_p + epsilon
            break
        
    cdef  float log_dwL = (log_wL_max - log_wL_min) / (init_params_h_N_wL - 1)

    iter_params_h_log_wL_min = log_wL_min
    iter_params_h_log_wL_max = log_wL_max
    return 



def read_npy(fname, arr):
    print("Loading {0}...".format(fname))
    arr = np.load(fname)
    print("Done!")

def start():
    dir_path = '/home/pankaj/radis-lab/'

    v0 = np.array(0,dtype="float")
    da = np.array(0,dtype="float")
    S0 = np.array(0,dtype="float")
    El = np.array(0,dtype="float")
    log_2vMm = np.array(0,dtype="float")
    na = np.array(0,dtype="float")
    log_2gs = np.array(0,dtype="float")
    v0_dec = np.array(0,dtype="float")
    da_dec = np.array(0,dtype="float")

    spectrum_h = np.array(0,dtype="float")
    v_arr = np.array(0,dtype="float")

    init_params_h_v_min = 1750.0
    init_params_h_v_max = 2400.0
    init_params_h_dv = 0.002
    init_params_h_N_v = int((init_params_h_v_max - init_params_h_v_max)/init_params_h_dv)

    init_params_h_N_wG = 4
    init_params_h_N_wL = 8 
    np.resize(spectrum_h, init_params_h_N_v)
    v_arr = [init_params_h_v_min + i * init_params_h_dv for i in range(init_params_h_N_v)]

    init_params_h_Max_iterations_per_thread = 1024
    init_params_h_block_preparation_step_size = 128

    host_params_h_shared_size = 0x8000          # Bytes - Size of the shared memory
    host_params_h_Min_threads_per_block = 128   # Ensures a full warp from each of the 4 processors
    host_params_h_Max_threads_per_block = 1024  # Maximum determined by device parameters
    init_params_h_shared_size_floats = host_params_h_shared_size / sys.getsizeof(float());

    init_params_h_N_wG_x_N_wL = init_params_h_N_wG * init_params_h_N_wL;
    init_params_h_N_total = init_params_h_N_wG_x_N_wL * init_params_h_N_v;
    init_params_h_N_points_per_block = init_params_h_shared_size_floats / init_params_h_N_wG_x_N_wL;
    init_params_h_N_threads_per_block = 1024;
    init_params_h_N_blocks_per_grid = 4 * 256 * 256;
    init_params_h_N_points_per_thread = init_params_h_N_points_per_block / init_params_h_N_threads_per_block;

    print()
    print("Spectral points per block  : {0}".format(init_params_h_N_points_per_block))
    print("Threads per block          : {0}".format(init_params_h_N_threads_per_block))
    print("Spectral points per thread : {0}".format(init_params_h_N_points_per_thread))
    print()

    # init v:
    print("Init v : ")
    init_params_h_Max_lines = int(2.4E8)
    read_npy(dir_path+'v0.npy', v0)
    spec_h_v0 = v0
    read_npy(dir_path+'da.npy', da)
    spec_h_da = da

    #decimate (v0, v0_dec, init_params_h_N_threads_per_block)
    v0_dec = np.minimum.reduceat(v0, np.arange(0, len(v0), init_params_h_N_threads_per_block))
    host_params_h_v0_dec = v0_dec
    host_params_h_dec_size = len(v0_dec)
    #decimate (da, da_dec, init_params_h_N_threads_per_block)
    da_dec = np.minimum.reduceat(da, np.arange(0, len(da), init_params_h_N_threads_per_block))
    host_params_h_da_dec = da_dec
    print()

    # wL inits
    print("Init wL: ")
    read_npy(dir_path + 'log_2gs.npy', log_2gs)
    spec_h_log_2gs = log_2gs
    read_npy(dir_path + 'na.npy', na)
    spec_h_na = na
    #init_lorentzian_params(log_2gs, na)
    print()

    # wG inits:
    print("Init wG: ")
    read_npy(dir_path + 'log_2vMm.npy', log_2vMm)
    spec_h_log_2vMm = log_2vMm
    #init_gaussian_params(log_2vMm)
    print()

    # I inits:
    print("Init I: ")
    read_npy(dir_path + 'S0.npy', S0)
    spec_h_S0 = S0
    read_npy(dir_path + 'El.npy', El)
    spec_h_El = El
    print()

    init_params_h_N_lines = int(len(v0))
    print("Number of lines loaded: {0}".format(init_params_h_N_lines))
    print()

    print("Allocating device memory...")

    # start the CUDA work
    # 
    # LINE 1103
    # .
    # .
    # .
    # LINE 1164    


    # START ITERATIONS

    p = 0.1
    T = 2000.0

    T_min = 500.0
    T_max = 5000.0
    dT = 500.0

    for T in range(T_min, T_max, dT):
        #iterate(p, T, spectrum_h)
        1
    return






########## JUNK: INGORE ALL THIS #################3
# P.S. kept for syntax/reference


#cdef  void c_init_lorentzian_params(vector[float] log_2gS, vector[float] na):

# def set_val():
#     global iter_params_h_log_wL_min 
#     iter_params_h_log_wL_min = -69
#     return

# def get_val():
#     global iter_params_h_log_wL_min 
#     print("the value you are looking for = {0}".format(iter_params_h_log_wL_min))
#     return iter_params_h_log_wL_min




# cdef   from "c_struct.c":
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
# cdef  class py_spectralData:
#     cdef  spectralData sd

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
#     #    cdef  float* v0_ptr = <float*> val.data
#         self.sd.v0 = <float *> val.data
    
#     # @property
#     # def v0(self):
#     #     cdef  float* v0_ptr = <float*> self.sd.v0.data
#     #     return v0_ptr

# cdef   from "point.c":
#     ctypedef struct Point:
#         int x
#         int y
#     Point make_and_send_point(int x, int y)

# cdef  class PyPoint:
#     cdef  Point p

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