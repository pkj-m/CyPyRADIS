# distutils: language=c++

cdef extern from "struct_def.c":
    ctypedef struct spectralData
    ctypedef struct initData
    ctypedef struct blockData
    ctypedef struct iterData

