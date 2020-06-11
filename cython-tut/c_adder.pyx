cdef extern from "c_adder.c":
    cdef int add(int, int)

def Add(a, b):
    return add(a, b)