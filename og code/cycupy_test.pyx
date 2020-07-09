import cupy as cp
import numpy as np
cimport numpy as np
import ctypes
import sys

# Python declaration of the struct:
class floatPair(ctypes.Structure):
    _fields_=[("a",ctypes.c_float),
              ("b",ctypes.c_float)]

class initData(ctypes.Structure):
    _fields_= [
        ("v_min", ctypes.c_float),
        ("v_max", ctypes.c_float),
        ("dv", ctypes.c_float),

        ("N_v", ctypes.c_int),
        ("N_wG", ctypes.c_int),
        ("N_wL", ctypes.c_int),
        ("N_wG_x_N_wL", ctypes.c_int),
        ("N_total", ctypes.c_int),

        ("Max_lines", ctypes.c_int),
        ("N_lines", ctypes.c_int),
        ("N_points_per_block", ctypes.c_int),
        ("N_threads_per_block", ctypes.c_int),
        ("N_blocks_per_grid", ctypes.c_int),
        ("N_points_per_thread", ctypes.c_int),
        ("Max_iterations_per_thread", ctypes.c_int),

        ("shared_size_floats", ctypes.c_int)
    ]

init_params_h = initData()

              
def gpu_add(x1,x2,int a,int b):
    global init_params_h

    init_params_h.v_min=1.0
    init_params_h.v_max	=2.0
    init_params_h.dv=3.0
    init_params_h.N_v=4
    init_params_h.N_wG=5
    init_params_h.N_wL=6
    init_params_h.N_wG_x_N_wL=7
    init_params_h.N_total=8
    init_params_h.Max_lines=9
    init_params_h.N_lines=10
    init_params_h.N_points_per_block=11
    init_params_h.N_threads_per_block=12
    init_params_h.N_blocks_per_grid=13
    init_params_h.N_points_per_thread=14
    init_params_h.Max_iterations_per_thread=15
    init_params_h.shared_size_floats=16

    # length of array = 16

    dummy_array = cp.zeros(16, dtype=cp.float32)

    print("og array before kernel processing: ")
    for i in dummy_array:
        print(i)

    print("\n KERNEL LAUNCHED \n")
    cuda_code = r'''
    extern "C"{
        
        //also tell the CUDA compiler what the struct looks like:
        struct floatPair{
            float   a;
            float   b;
            };

        struct initData {
            float v_min;
            float v_max;	//Host only
            float dv;

            // DLM sizes:
            int N_v;
            int N_wG;
            int N_wL;
            int N_wG_x_N_wL;
            int N_total;

            //Work parameters :
            int Max_lines;
            int N_lines;
            int N_points_per_block;
            int N_threads_per_block;
            int N_blocks_per_grid;
            int N_points_per_thread;
            int	Max_iterations_per_thread;

            int shared_size_floats;
        };
        
        //declare the constant memory variables:
        __device__ __constant__ floatPair const_struct_d;
        __device__ __constant__ initData init_params_d;

        
        __global__ void my_add(float* dummy, int N) {
            dummy[0] = init_params_d.v_min;
            dummy[1] = init_params_d.v_max;	
            dummy[2] = init_params_d.dv;
            dummy[3] = init_params_d.N_v;
            dummy[4] = init_params_d.N_wG;
            dummy[5] = init_params_d.N_wL;
            dummy[6] = init_params_d.N_wG_x_N_wL;
            dummy[7] = init_params_d.N_total;
            dummy[8] = init_params_d.Max_lines;
            dummy[9] = init_params_d.N_lines;
            dummy[10] = init_params_d.N_points_per_block;
            dummy[11] = init_params_d.N_threads_per_block;
            dummy[12] = init_params_d.N_blocks_per_grid;
            dummy[13] = init_params_d.N_points_per_thread;
            dummy[14] = init_params_d.Max_iterations_per_thread;
            dummy[15] = init_params_d.shared_size_floats;
        }
    }
    '''
        
    module = cp.RawModule(code = cuda_code)
    add_kernel = module.get_function('my_add')

    x1_d = cp.asarray(x1)
    x2_d = cp.asarray(x2)
    y_d = cp.zeros((5, 5), dtype=cp.float32)

    ## Transfer struct h2d using constant memory
    params = floatPair()
    params.a = a
    params.b = b

    memptr = module.get_global("init_params_d")
    struct_ptr = ctypes.cast(ctypes.pointer(init_params_h),ctypes.c_void_p)
    struct_size = ctypes.sizeof(init_params_h)
    print('sizeof p:', struct_size)
    memptr.copy_from_host(struct_ptr,struct_size)
    print('Done!')

    add_kernel((1,), (1,), (dummy_array, 16))
    
    print("array after kernel: ")
    for i in dummy_array:
        print(i)

    print("end...\n")

    return np.arange(25).resize(5,5)