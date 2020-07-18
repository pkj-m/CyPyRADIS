import sys
import numpy as np
cimport numpy as np
import cupy as cp

#cdef np.ndarray[np.float32_t, ndim=1] offs

cdef test(int n):
    cdef np.ndarray[np.float32_t, ndim=1] arr = np.zeros(n, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] offs = np.zeros(0, dtype=np.float32)
    arr_d = cp.array(arr, dtype=cp.float32)
    offs_d = cp.array(offs, dtype=cp.float32)
    #arr = np.zeros(n ,dtype=np.float32)
    #offs = np.zeros(0 ,dtype=np.float32)

    for i in range(n):
        arr[i] = i
        arr_d[i] = i

    print("array = {0}".format(arr))
    print("array_d = {0}".format(arr_d))
    print("offs = {0}".format(offs))
    print("offs_d = {0}".format(offs_d))
    print("size of array  = {0}".format(sys.getsizeof(arr)))
    print("size of array_d  = {0}".format(sys.getsizeof(arr_d)))
    print("size of offs  = {0}".format(sys.getsizeof(offs)))
    print("size of offs_d  = {0}".format(sys.getsizeof(offs_d)))
    #print("size of offset = {0}".format(sys.getsizeof(offs)))

def start():
    n = int(input())
    test(n)