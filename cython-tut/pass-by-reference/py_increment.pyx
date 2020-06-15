# distutils: language=c++
from libcpp.vector cimport vector


cdef extern from *:
    """
    #include<iostream>
    #include<vector>
    using namespace std;
    void add_one(vector<int> &vec){
        for (int i = 0; i < (int)vec.size(); i++){
            vec[i] = vec[i] + 1;
        }
        return;
    }
    """
    void add_one(vector[int] &vec)

def inc(vector[int] &a):
    add_one(a)
    return a

