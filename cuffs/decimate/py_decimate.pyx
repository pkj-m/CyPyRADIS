# distutils: language=c++

from libcpp.vector cimport vector

# cdef extern from "decimate.cpp":
#    cdef void decimate(vector[float]&, vector[float]&,  int)

cdef extern from "decimate.cpp":
    cdef vector[float] decimate(vector[float], int)

# cdef void mega_dec(vector[float]& v_in, vector[float]& v_out, int step):
#     print("Decimating (", step,")... ")
#     v_out.clear()
#     for i in range(<int>v_in.size() / step):
#         v_out.push_back(v_in[i * step])
#     print("Done!")

def Dec(vector[float] v_in, int step):
    return decimate(v_in, step)