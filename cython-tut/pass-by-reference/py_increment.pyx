# distutils: language=c++
cimport numpy as np
import numpy as np
from libcpp.vector cimport vector

cdef extern from *:
    """
    #include<iostream>
    #include<vector>
    using namespace std;
    void add_one(vector<double> &v_in, vector<double> &v_out){
        int n = v_in.size();
        //v_out = (double *)malloc(n * sizeof(double));
        for (int i = 0; i < n; i++){
            v_out[i] = v_in[i] + 1;
        }
        return;
    }
    """
    #void add_one(double* v_in, double* v_out)
    void add_one(vector[double] &v_in, vector[double] &v_out)


# input is vin and vout, numpy array
def inc(np.ndarray[double, ndim=1, mode="c"] v_in, 
        np.ndarray[double, ndim=1, mode="c"] v_out):
    add_one(v_in, v_out)
    # cdef int n = len(v_in)
    
    # cdef double *ptr_vin = &v_in[0]
    # cdef double *ptr_vout = &v_out[0]
    # add_one(ptr_vin, ptr_vout)
