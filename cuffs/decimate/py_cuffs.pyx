# distutils: language=c++

from libcpp.vector cimport vector

cdef extern from "decimate.cpp":
    cdef vector[float] decimate(vector[float], int)

def py_decimate(vector[float] v_in, int step):
    return decimate(v_in,step)