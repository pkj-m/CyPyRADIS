from cpython cimport array
import numpy as np
cimport numpy as np

cdef extern from "c_struct.c":
    ctypedef struct spectralData:
        float* v0
        float* da
        float* S0
        float* El
        float* log_2vMm
        float* na
        float* log_2gs

    ctypedef struct initData:
        float v_min
        float v_max	# Host only
        float dv

        # DLM sizes:
        int N_v
        int N_wG
        int N_wL
        int N_wG_x_N_wL
        int N_total

        # Work parameters :
        int Max_lines
        int N_lines
        int N_points_per_block
        int N_threads_per_block
        int N_blocks_per_grid
        int N_points_per_thread
        int	Max_iterations_per_thread

        int shared_size_floats
    
    ctypedef struct blockData:
        int line_offset
        # int N_iterations
        int iv_offset
    
    ctypedef struct iterData:
        # Any info needed for the Kernel that does not change during 
        # kernel execution but MAY CHANGE during spectral iteration step

        # Pressure & temperature:
        float p
        # float T
        float log_p
        float hlog_T
        float log_rT
        float c2T
        float rQ

        # Spectral parameters:
        float log_wG_min
        float log_wL_min
        float log_dwG
        float log_dwL

        # Block data:
        blockData blocks[4096] 
    
    spectralData init_spectralData()


# # Then we describe a class that has a Point member "pt"
cdef class py_spectralData:
    cdef spectralData sd

    def __init__(self):
        self.sd = init_spectralData()
        self.sd.v0 = NULL
        self.sd.da = NULL
        self.sd.S0 = NULL
        self.sd.El = NULL
        self.sd.log_2vMm = NULL
        self.sd.na = NULL
        self.sd.log_2gs = NULL

    @v0.setter
    def v0(self,val):
    #    cdef float* v0_ptr = <float*> val.data
        self.sd.v0 = <float *> val.data
    
    # @property
    # def v0(self):
    #     cdef float* v0_ptr = <float*> self.sd.v0.data
    #     return v0_ptr

cdef extern from "point.c":
    ctypedef struct Point:
        int x
        int y
    Point make_and_send_point(int x, int y)

cdef class PyPoint:
    cdef Point p

    def __init__(self, x, y):
        self.p = make_and_send_point(x, y)

     # define properties in the normal Python way
    @property
    def x(self):
        return self.p.x

    @x.setter
    def x(self,val):
        self.p.x = val

    # @property
    # def y(self):
    #     return self.pt.y

    # @y.setter
    # def y(self,val):
    #     self.pt.y = val
