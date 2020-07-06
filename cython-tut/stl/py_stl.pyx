#distutils: language=c++

from __future__ import print_function
from libcpp.map cimport map as mapcpp
from libcpp.unordered_set cimport unordered_set
from libcpp.set cimport set
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from cython.operator import dereference, postincrement

cdef vector[int] vactor

cdef extern from *:
    """
    struct greater {
        bool operator () (const float x, const float y) const {return x > y;}       
    };
    """
    ctypedef struct greater:
        float a
        float b

cdef void test_func():
    global vactor
    cdef set[pair[int,int]] unique_set
    cdef mapcpp[int, int, greater] benv
    cdef mapcpp[int, int] tenv
    cdef vector[pair[int,int]] vac
    cdef mapcpp[int, int].iterator it = benv.begin()

    # print("inserting in set : ", end = "")
    # for i in range(10):
    #     print("({0} {1}) ({2} {3}) ".format(i, i+1, i, i+1), end = "")
    #     unique_set.insert({i,i+1})
    #     unique_set.insert({i, i+1})
    
    # print("\nprinting the set: ", end = "")
    # for i,j in unique_set:
    #     print("( {0} {1} ) ".format(i, j), end="")
    
    print("\ninserting in map: ", end="")
    benv.insert([10, 20])
    benv.insert([11, 22])
    benv.insert([9, 18])

    print("size of map = {0}".format(benv.size()), end="\n")
    
    for i, j in benv:
        print("{0} -> {1}".format(i, j))

    # vactor.push_back(10)
    # vactor.push_back(11)
    # vactor.push_back(12)
    # vactor.push_back(13)

    # print("Size of vactor = {0}".format(vactor.size()))
    # vactor.resize(2)
    
    # print("new size of vactor = {0}".format(vactor.size()))
    # vactor = [69, 42]
    # for i in vactor:
    #     print(i)

    # print("finished printing vactor")
    # vac.assign(benv.begin(), benv.end())
    # print("printing the vector pair: ")
    # for i,j in vac:
    #     print("{0} {1}".format(i,j), end = "\n")

    # print("...........................")

    # cdef unordered_set[int] s
    # for i in range(10):
    #     s.insert(i)
    
    # print("checking i = 5")
    # if s.count(5):
    #     print("found 5!")
    # else:
    #     print("not found 5!!")

    # print("checking i = 19...")
    # if s.count(19):
    #     print("found 19!")
    # else:
    #     print("not found 19!!!")


def py_def():
    test_func()