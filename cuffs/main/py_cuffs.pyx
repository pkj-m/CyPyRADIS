# distutils: language=c++

from __future__ import print_function
import pickle
from cpython cimport array
import numpy as np
import cupy as cp
cimport numpy as np
import sys
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.utility cimport pair
from libcpp.map cimport map as mapcpp
from cython.operator import dereference, postincrement


cdef float epsilon = 0.0001
cdef float FLOAT_MAX =  1e30
cdef float FLOAT_MIN = -1e30

#-----------------------------------
#    hostData: host_params_h       #
#-----------------------------------

cdef  int host_params_h_block_preparation_step_size
cdef  int host_params_h_Min_threads_per_block
cdef  int host_params_h_Max_threads_per_block

cdef  float host_params_h_log_2vMm_min
cdef  float host_params_h_log_2vMm_max

cdef  vector[float] host_params_h_top_x
cdef  vector[float] host_params_h_top_a
cdef  vector[float] host_params_h_top_b
cdef  vector[float] host_params_h_bottom_x
cdef  vector[float] host_params_h_bottom_a
cdef  vector[float] host_params_h_bottom_b

cdef  int host_params_h_dec_size
cdef  float* host_params_h_v0_dec
cdef  float* host_params_h_da_dec

cdef  int host_params_h_shared_size

host_params_h_start_ptr = cp.cuda.runtime.eventCreate()
host_params_h_stop_ptr = cp.cuda.runtime.eventCreate()
host_params_h_start_DLM_ptr = cp.cuda.runtime.eventCreate()
host_params_h_stop_DLM_ptr = cp.cuda.runtime.eventCreate()

host_params_h_start = cp.cuda.Event(host_params_h_start_ptr)
host_params_h_stop = cp.cuda.Event(host_params_h_stop_ptr)
host_params_h_start_DLM = cp.cuda.Event(host_params_h_start_DLM_ptr)
host_params_h_stop_DLM = cp.cuda.Event(host_params_h_stop_DLM_ptr)

cdef  float host_params_h_elapsedTime
cdef  float host_params_h_elapsedTimeDLM

#cdef  cufftHandle host_params_h_plan_DLM
#cdef  cufftHandle host_params_h_plan_spectrum

# device pointers
cdef  float* host_params_h_v0_d
cdef  float* host_params_h_da_d
cdef  float* host_params_h_S0_d
cdef  float* host_params_h_El_d
cdef  float* host_params_h_log_2gs_d
cdef  float* host_params_h_na_d
cdef  float* host_params_h_log_2vMm_d
cdef  float* host_params_h_DLM_d
cdef  float* host_params_h_spectrum_d

#cdef  cufftReal* host_params_h_DLM_d_in
#cdef  cufftComplex* host_params_h_DLM_d_out

#cdef  cufftComplex* host_params_h_spectrum_d_in
#cdef  cufftReal* host_params_h_spectrum_d_out

#-----------------------------------

#-----------------------------------
#    initData: init_params_h       #
# ----------------------------------

# DLM spectral parameters
cdef  float init_params_h_v_min
cdef  float init_params_h_v_max
cdef  float init_params_h_dv

# DLM sizes:
cdef  int init_params_h_N_v
cdef  int init_params_h_N_wG
cdef  int init_params_h_N_wL
cdef  int init_params_h_N_wG_x_N_wL
cdef  int init_params_h_N_total

# work parameters:
cdef  int init_params_h_Max_lines
cdef  int init_params_h_N_lines
cdef  int init_params_h_N_points_per_block
cdef  int init_params_h_N_threads_per_block
cdef  int init_params_h_N_blocks_per_grid
cdef  int init_params_h_N_points_per_thread
cdef  int init_params_h_Max_iterations_per_thread

cdef  int init_params_h_shared_size_floats

# ---------------------------------


#-----------------------------------------
#       iterData: iter_params_h          #
#-----------------------------------------

# pressure and temperature
cdef  float iter_params_h_p
cdef  float iter_params_h_log_p
cdef  float iter_params_h_hlog_T
cdef  float iter_params_h_log_rT
cdef  float iter_params_h_c2T
cdef  float iter_params_h_rQ

# spectral parameters
cdef  float iter_params_h_log_wG_min
cdef  float iter_params_h_log_wL_min
cdef  float iter_params_h_log_dwG
cdef  float iter_params_h_log_dwL

cdef  int iter_params_h_blocks_line_offset[4096]
cdef  int iter_params_h_blocks_iv_offset[4096]

#------------------------------------------


#--------------------------------------------
#       spectralData: spec_h                #
#--------------------------------------------

cdef  vector[float] spec_h_v0
cdef  vector[float] spec_h_da
cdef  vector[float] spec_h_S0
cdef  vector[float] spec_h_El
cdef  vector[float] spec_h_log_2vMm
cdef  vector[float] spec_h_na
cdef  vector[float] spec_h_log_2gs

#--------------------------------------------


#--------------------------------------------
#               gpu data                    #
#--------------------------------------------

#-----------------------------------
#    initData: init_params_d       #
# ----------------------------------

# DLM spectral parameters
cdef  float init_params_d_v_min
cdef  float init_params_d_v_max
cdef  float init_params_d_dv

# DLM sizes:
cdef  int init_params_d_N_v
cdef  int init_params_d_N_wG
cdef  int init_params_d_N_wL
cdef  int init_params_d_N_wG_x_N_wL
cdef  int init_params_d_N_total

# work parameters:
cdef  int init_params_d_Max_lines
cdef  int init_params_d_N_lines
cdef  int init_params_d_N_points_per_block
cdef  int init_params_d_N_threads_per_block
cdef  int init_params_d_N_blocks_per_grid
cdef  int init_params_d_N_points_per_thread
cdef  int init_params_d_Max_iterations_per_thread

cdef  int init_params_d_shared_size_floats


####################################
######## KERNELS COME HERE #########
####################################


# fillDLM

fillDLM_c_code = r'''
extern "C"{
__global__ void fillDLM(
	float* v0,
	float* da,
	float* S0,
	float* El,
	float* log_2gs,
	float* na,
	float* log_2vMm,
	float* global_DLM
    
    double iter_params_d_p,
    double iter_params_d_log_p,
    double iter_params_d_dlog_T,
    double iter_params_d_log_rT,
    double iter_params_d_c2T,
    double iter_params_d_rQ,
    double iter_params_d_log_wG_min,
    double iter_params_d_log_wL_min,
    double iter_params_d_log_dwG,
    double iter_params_d_log_dwL,
    int* iter_params_d_blocks_line_offset,
    int* iter_params_d_blocks_iv_offset,

    double init_params_d_v_min,
    double init_params_d_v_max,
    double init_params_d_dv,
    long long init_params_d_N_v,
    long long init_params_d_N_wG,
    long long init_params_d_N_wL,
    long long init_params_d_N_wG_x_N_wL,
    long long init_params_d_N_total,
    long long init_params_d_Max_lines,
    long long init_params_d_N_lines,
    long long init_params_d_N_points_per_block,
    long long init_params_d_N_threads_per_block,
    long long init_params_d_N_blocks_per_grid,
    long long init_params_d_N_points_per_thread,
    long long init_params_d_Max_iterations_per_thread,
    long long init_params_d_shared_size_floats
    ) {

	
    int block_line_offset = iter_params_d_blocks_line_offset[blockIdx.x + gridDim.x * blockIdx.y];
    int block_iv_offset = iter_params_d_blocks_iv_offset[blockIdx.x + gridDim.x * blockIdx.y];

	int block_id = blockIdx.x + gridDim.x * blockIdx.y;
	int N_iterations = (iter_params_d_blocks_line_offset[block_id + 1] - iter_params_d_blocks_line_offset[block_id]) / init_params_d_N_threads_per_block;
	int DLM_offset = iter_params_d_blocks_iv_offset[block_id] * init_params_d_N_wG_x_N_wL;
	int iv_offset = iter_params_d_blocks_iv_offset[block_id];

	int NwG = init_params_d_N_wG;
	int NwGxNwL = init_params_d_N_wG_x_N_wL;

	////Allocate and zero the Shared memory
	extern __shared__ float shared_DLM[];

	float* DLM = global_DLM;

	for (int n = 0; n < N_iterations; n++) { 

		// >>: Process from left to right edge:
		int i = iter_params_d_blocks_line_offset[block_id] + threadIdx.x + n * blockDim.x;
		
		if (i < init_params_d_N_lines) {
			//Calc v
			float v_dat = v0[i] + iter_params_d_p * da[i];  // <----- PRESSURE SHIFT
			float iv = (v_dat - init_params_d_v_min) / init_params_d_dv; //- iv_offset;
			int iv0 = (int)iv;
			int iv1 = iv0 + 1  ;

			if ((iv0 >= 0) && (iv1 < init_params_d_N_v)) {

				//Calc wG
				float log_wG_dat = log_2vMm[i] + iter_params_d_hlog_T; // <---- POPULATION
				float iwG = (log_wG_dat - iter_params_d_log_wG_min) / iter_params_d_log_dwG;
				int iwG0 = (int)iwG;
				int iwG1 = iwG0 + 1;
				//^8

				//Calc wL
				float log_wL_dat = log_2gs[i] + iter_params_d_log_p + na[i] * iter_params_d.log_rT;
				float iwL = (log_wL_dat - iter_params_d.log_wL_min) / iter_params_d.log_dwL;
				int iwL0 = (int)iwL;	
				int iwL1 = iwL0 + 1;
				//^12

				//Calc I  	Line intensity
				float I_add = iter_params_d_rQ * S0[i] * (expf(iter_params_d_c2T * El[i]) - expf(iter_params_d_c2T * (El[i] + v0[i])));
				
				//  reducing the weak line code would come here

				float av = iv - iv0;
				float awG = (iwG - iwG0) * expf((iwG1 - iwG) * iter_params_d_log_dwG);
				float awL = (iwL - iwL0) * expf((iwL1 - iwL) * iter_params_d_log_dwL);

				float aV00 = (1 - awG) * (1 - awL);
				float aV01 = (1 - awG) * awL;
				float aV10 = awG * (1 - awL);
				float aV11 = awG * awL;

				float Iv0 = I_add * (1 - av);
				float Iv1 = I_add * av;

				atomicAdd(&DLM[iwG0 + iwL0 * NwG + iv0 * NwGxNwL], aV00 * Iv0);
				atomicAdd(&DLM[iwG0 + iwL0 * NwG + iv1 * NwGxNwL], aV00 * Iv1);
				atomicAdd(&DLM[iwG0 + iwL1 * NwG + iv0 * NwGxNwL], aV01 * Iv0);
				atomicAdd(&DLM[iwG0 + iwL1 * NwG + iv1 * NwGxNwL], aV01 * Iv1); 
				atomicAdd(&DLM[iwG1 + iwL0 * NwG + iv0 * NwGxNwL], aV10 * Iv0);
				atomicAdd(&DLM[iwG1 + iwL0 * NwG + iv1 * NwGxNwL], aV10 * Iv1);
				atomicAdd(&DLM[iwG1 + iwL1 * NwG + iv0 * NwGxNwL], aV11 * Iv0);
				atomicAdd(&DLM[iwG1 + iwL1 * NwG + iv1 * NwGxNwL], aV11 * Iv1);
			}
		}
	} 
}

}'''

fillDLM_module = cp.RawModule(code=fillDLM_c_code)
fillDLM = fillDLM_module.get_function('fillDLM')


# applyLineshapes

applyLineshapes_c_code = r'''
extern "C"{
__global__ void applyLineshapes(cufftComplex* DLM, 
                                cufftComplex* spectrum
                                // include iter_params_d
                                // include init_params_d
                                ) {

	const float pi = 3.141592653589793f;
	const float r4log2 = 0.36067376022224085f; // = 1 / (4 * ln(2))
	int iv = threadIdx.x + blockDim.x * blockIdx.x;

	if (iv < init_params_d_N_v + 1) {

		float x = iv / (2 * init_params_d_N_v * init_params_d_dv);
		float mul = 0.0;
		float out_re = 0.0;
		float out_im = 0.0;
		float wG, wL;
		int index;

		for (int iwG = 0; iwG < init_params_d_N_wG; iwG++) {
			wG = expf(iter_params_d_log_wG_min + iwG * iter_params_d_log_dwG);
			for (int iwL = 0; iwL < init_params_d_N_wL; iwL++) {
				index = iwG + iwL * init_params_d_N_wG + iv * init_params_d_N_wG_x_N_wL;
				wL = expf(iter_params_d_log_wL_min + iwL * iter_params_d_log_dwL);
				mul = expf(-r4log2 * powf(pi * x * wG, 2) - pi * x * wL);
				out_re += mul * DLM[index].x;   // check this
				out_im += mul * DLM[index].y;
			}
		}
		spectrum[iv].x = out_re;    // check this as well
		spectrum[iv].y = out_im;
	}
}
}'''

apply_lineshapes_module = cp.RawModule(code=applyLineshapes_c_code)
apply_lineshapes = apply_lineshapes_module.get_function('applyLineshapes')


####################################


cdef void set_pT(float p, float T):
    
    # ----------- setup global variables -----------------
    global iter_params_h_p
    global iter_params_h_log_p
    global iter_params_h_hlog_T
    global iter_params_h_log_rT
    global iter_params_h_c2T
    global iter_params_h_rQ
    #------------------------------------------------------

    cdef float c2 = 1.4387773538277204
    iter_params_h_p = p
    iter_params_h_log_p = np.log(p)
    iter_params_h_hlog_T = 0.5 * np.log(T)
    iter_params_h_log_rT = np.log(296.0/T)
    iter_params_h_c2T = -c2/T

    cdef float B = 0.39
    cdef float w1 = 1388.0
    cdef float w2 = 667.0
    cdef float w3 = 1349.0

    cdef int d1 = 1
    cdef int d2 = 2
    cdef int d3 = 1

    cdef float Qr = T/(c2 * B)
    cdef float Qv1 = 1 / np.power(1 - np.exp(-c2 * w1 / T), d1)
    cdef float Qv2 = 1 / np.power(1 - np.exp(-c2 * w2 / T), d2)
    cdef float Qv3 = 1 / np.power(1 - np.exp(-c2 * w3 / T), d3)
    cdef float Q = Qr * Qv1 * Qv2 * Qv3;

    iter_params_h_rQ = 1 / Q
    iter_params_h_rQ = iter_params_h_rQ / T


def read_npy(fname, arr):
    print("Loading {0}...".format(fname))
    arr = np.load(fname)
    print("Done!")

cdef void init_lorentzian_params(vector[float] log_2gs, vector[float] na):

    # ----------- setup global variables -----------------
    global host_params_h_top_a
    global host_params_h_top_b
    global host_params_h_top_x
    global host_params_h_bottom_a
    global host_params_h_bottom_b
    global host_params_h_bottom_x
    #------------------------------------------------------

    cdef set[pair[float,float]] unique_set
    cdef vector[pair[float,float]] duplicates_removed
    cdef vector[float] na_short
    cdef vector[float] log_2gs_short
    cdef mapcpp[float, float] bottom_envelope_map
    cdef mapcpp[float, float] top_envelope_map

    cdef vector[float] top_a 
    cdef vector[float] top_b 
    cdef vector[float] top_x

    cdef vector[float] bottom_a 
    cdef vector[float] bottom_b 
    cdef vector[float] bottom_x

    print("Initializing Lorentzian parameters")

    cdef int top_size = 0
    cdef int bottom_size = 0

    fname = "Lorenzian_minmax_" + str(len(log_2gs)) + ".dat"

    try:
        with open(fname, 'rb') as f:
            print(" (from cache)... ")
            lt = pickle.load(f)

            top_size = lt[0]
            host_params_h_top_a.resize(top_size)
            host_params_h_top_b.resize(top_size)
            host_params_h_top_x.resize(top_size)

            # now read top_size bits 3 times to fill the above 3 vectors
            host_params_h_top_a = lt[1]
            host_params_h_top_b = lt[2]
            host_params_h_top_x = lt[3]

            bottom_size = lt[4]
            host_params_h_bottom_a.resize(bottom_size)
            host_params_h_bottom_b.resize(bottom_size)
            host_params_h_bottom_x.resize(bottom_size)

            host_params_h_bottom_a = lt[5]
            host_params_h_bottom_b = lt[6]
            host_params_h_bottom_x = lt[7]

    except:
        print(" ... ")

        for i in range(len(na)):
            unique_set.insert({na[i], log_2gs[i]})
        
        duplicates_removed.assign(unique_set.begin(), unique_set.end())

        for i, j in duplicates_removed:
            na_short.push_back(i)
            log_2gs_short.push_back(j)

        for i in range(len(na_short)):
            na_i = na_short[i]
            log_2gs_i = log_2gs_short[i]

            if bottom_envelope_map.count(na_i):
                if log_2gs_i < bottom_envelope_map.at(na_i):
                    bottom_envelope_map[na_i] = log_2gs_i
            else:
                bottom_envelope_map.insert({na_i, log_2gs_i})

            if top_envelope_map.count(na_i):
                if log_2gs_i > top_envelope_map.at(na_i):
                    top_envelope_map[na_i] = log_2gs_i
            else:
                top_envelope_map.insert({na_i, log_2gs_i})
        
        top_a = { dereference(top_envelope_map.begin()).first }
        top_b = { dereference(top_envelope_map.begin()).second }
        top_x = { FLOAT_MIN }

        idx = 0
        for first_el, second_el in top_envelope_map:
            if idx != 0:
                for i in range(len(top_x)):
                    x_ij = (second_el - top_b[i]) / (top_a[i] - first_el)
                    if x_ij >= top_x[i]:
                        if i < top_x.size() - 1:
                            if x_ij < top_x[i+1]:
                                break;
                        else:
                            break
                
                top_a.resize(i+1)
                top_b.resize(i+1)
                top_x.resize(i+1)

                top_a.push_back(first_el)
                top_b.push_back(second_el)
                top_x.push_back(x_ij)

            idx+=1

        top_x.erase(top_x.begin())
        top_x.push_back(FLOAT_MAX)

        host_params_h_top_a = top_a
        host_params_h_top_b = top_b
        host_params_h_top_x = top_x
        top_size = top_x.size()

        bottom_a = { dereference(bottom_envelope_map.begin()).first }
        bottom_b = { dereference(bottom_envelope_map.begin()).second }
        bottom_x = { FLOAT_MIN }

        idx = 0

        for first_el, second_el in bottom_envelope_map:
            if idx != 0:
                for i in range(len(bottom_x)):
                    x_ij = (second_el - bottom_b[i]) / (bottom_a[i] - first_el)
                    if x_ij >= bottom_x[i]:
                        if i < bottom_x.size() - 1:
                            if x_ij < bottom_x[i+1]:
                                break
                        else:
                            break
                
                bottom_a.resize(i + 1)
                bottom_b.resize(i + 1)
                bottom_x.resize(i + 1)
                
                bottom_a.push_back(first_el)
                bottom_b.push_back(second_el)
                bottom_x.push_back(x_ij)

            idx+=1
        
        bottom_x.erase(bottom_x.begin())
        bottom_x.push_back(FLOAT_MAX)

        host_params_h_bottom_a = bottom_a
        host_params_h_bottom_b = bottom_b
        host_params_h_bottom_x = bottom_x
        bottom_size = bottom_x.size()

        lt = [top_size,
            host_params_h_top_a,
            host_params_h_top_b,
            host_params_h_top_x,
            bottom_size,
            host_params_h_bottom_a,
            host_params_h_bottom_b,
            host_params_h_bottom_x]
        
        with open(fname, 'wb') as f:
            pickle.dump(lt, f)
    
    print("Done!")
    return

cdef void calc_lorentzian_params():

    # ----------- setup global variables -----------------
    global host_params_h_top_x
    global host_params_h_bottom_x
    global host_params_h_top_a
    global host_params_h_bottom_a
    global host_params_h_top_b
    global host_params_h_bottom_b
    global iter_params_h_log_rT
    global iter_params_h_log_p
    global iter_params_h_log_wL_min
    global iter_params_h_log_wL_max
    global epsilon
    #------------------------------------------------------
    
    cdef float log_wL_min
    cdef float log_wL_max

    for i in range(host_params_h_bottom_x.size()):
        if iter_params_h_log_rT < host_params_h_bottom_x[i]:
            log_wL_min = iter_params_h_log_rT * host_params_h_bottom_a[i] + host_params_h_bottom_b[i]  + iter_params_h_log_p
            break
    
    for i in range(host_params_h_top_x.size()):
        if iter_params_h_log_rT < host_params_h_top_x[i]:
            log_wL_max = iter_params_h_log_rT * host_params_h_top_a[i] + host_params_h_top_b[i]  + iter_params_h_log_p + epsilon
            break
        
    cdef  float log_dwL = (log_wL_max - log_wL_min) / (init_params_h_N_wL - 1)

    iter_params_h_log_wL_min = log_wL_min
    iter_params_h_log_wL_max = log_wL_max
    return 


cdef void init_gaussian_params(vector[float] log_2vMm):

    # ----------- setup global variables -----------------
    global host_params_h_log_2vMm_min
    global host_params_h_log_2vMm_max
    #------------------------------------------------------

    cdef float log_2vMm_min
    cdef float log_2vMm_max
    print("Initializing Gaussian parameters")

    fname = "Gaussian_minmax_" + str(len(log_2vMm)) + ".dat"
    try:
        with open(fname, 'rb') as f:
            print(" (from cache)... ")
            lt = pickle.load(f)
            log_2vMm_min = lt[0]
            log_2vMm_max = lt[1]
    except:
        print("... ")
        log_2vMm_min = np.minimum(log_2vMm)
        log_2vMm_max = np.maximum(log_2vMm)
        lt = [log_2vMm_min, log_2vMm_max]
        with open(fname, 'wb') as f:
            pickle.dump(lt, f)
    
    host_params_h_log_2vMm_min = log_2vMm_min
    host_params_h_log_2vMm_max = log_2vMm_max

    print("Done!")

    return


cdef void calc_gaussian_params():

    # ----------- setup global variables -----------------
    global host_params_h_log_2vMm_min
    global host_params_h_log_2vMm_max
    global init_params_h_N_wG
    global iter_params_h_hlog_T
    global iter_params_h_log_wG_min
    global iter_params_h_log_dwG
    global epsilon
    #------------------------------------------------------

    cdef float log_wG_min = host_params_h_log_2vMm_min + iter_params_h_hlog_T
    cdef float log_wG_max = host_params_h_log_2vMm_max + iter_params_h_hlog_T + epsilon
    cdef float log_dwG = (log_wG_max - log_wG_min) / (init_params_h_N_wG - 1)

    iter_params_h_log_wG_min = log_wG_min
    iter_params_h_log_dwG = log_dwG

    return

cdef int prepare_blocks():

    # ----------- setup global variables -----------------
    global host_params_h_v0_dec
    global host_params_h_da_dec
    global host_params_h_dec_size

    global iter_params_h_p
    global iter_params_h_blocks_iv_offset
    global iter_params_h_blocks_line_offset

    global init_params_h_block_preparation_step_size
    global init_params_h_N_points_per_block
    global init_params_h_N_threads_per_block
    global init_params_h_dv
    global init_params_h_Max_iterations_per_thread
    global init_params_h_v_min
    global init_params_h_dv

    global epsilon
    #------------------------------------------------------

    cdef float* v0 = host_params_h_v0_dec
    cdef float* da = host_params_h_da_dec

    cdef float v_prev
    cdef float dvdi
    cdef int i = 0
    cdef int n = 0
    cdef int step = host_params_h_block_preparation_step_size

    # in lieu of blockData struct, create new arrays
    cdef int new_block_line_offset
    cdef int iv_offset

    cdef float v_cur = v0[0] + iter_params_h_p * da[0]
    cdef float v_max = v_cur + init_params_h_N_points_per_block * init_params_h_dv
    cdef int i_max = init_params_h_Max_iterations_per_thread
    
    new_block_line_offset = 0
    new_block_iv_offset = int(((v_cur - init_params_h_v_min) / init_params_h_dv))


    while 1:
        i += step
        if i > host_params_h_dec_size:
            iter_params_h_blocks_line_offset[n] = new_block_line_offset
            iter_params_h_blocks_iv_offset[n] = new_block_iv_offset
            n+=1
            new_block_line_offset = i * init_params_h_N_threads_per_block
            iter_params_h_blocks_line_offset[n] = new_block_line_offset
            iter_params_h_blocks_iv_offset[n] = new_block_iv_offset
            break
        
        v_prev = v_cur
        v_cur = v0[i] + iter_params_h_p * da[i]
        
        if ((v_cur > v_max) or (i >= i_max)) : 
            if (v_cur > v_max) : 
                dvdi = (v_cur - v_prev) / float(step)
                i -= int(((v_cur - v_max) / dvdi)) + 1
                v_cur = v0[i] + iter_params_h_p * da[i]
            
            iter_params_h_blocks_line_offset[n] = new_block_line_offset
            iter_params_h_blocks_iv_offset[n] = new_block_iv_offset
            n+=1
            new_block_iv_offset = int(((v_cur - init_params_h_v_min) / init_params_h_dv))
            new_block_line_offset = i * init_params_h_N_threads_per_block
            v_max = v_cur + (init_params_h_N_points_per_block) * init_params_h_dv
            i_max = i + init_params_h_Max_iterations_per_thread
    
    return n

#cdef void check_block_spillage(int n_blocks, vector[float] v0, vector[float] da ...):
#    return


cdef void iterate(float p, float T, vector[float] spectrum_h):
    
    # ----------- setup global variables -----------------
    global host_params_h_start
    global host_params_h_DLM_d
    global init_params_h_N_v
    global init_params_h_N_wG_x_N_wL
    global host_params_h_start_DLM
    global init_params_h_N_threads_per_block
    global host_params_h_shared_size
    global host_params_h_v0_d
    global host_params_h_da_d
    global host_params_h_S0_d
    global host_params_h_El_d
    global host_params_h_log_2gs_d
    global host_params_h_na_d
    global host_params_h_log_2vMm_d
    global host_params_h_DLM_d
    global host_params_h_stop_DLM
    global host_params_h_elapsedTimeDLM
    global host_params_h_plan_DLM
    global host_params_h_DLM_d_in
    global host_params_h_DLM_d_out
    global host_params_h_spectrum_d_in
    global host_params_h_plan_spectrum
    global host_params_h_spectrum_d_out
    global spectrum_h
    global host_params_h_spectrum_d
    global host_params_h_stop
    global host_params_h_elapsedTime
    global iter_params_h_log_dwG
    global iter_params_h_log_dwL
    global epsilon
    #------------------------------------------------------

    host_params_h_start.record()
    
    cdef int n_blocks
    set_pT(p, T)
    calc_gaussian_params()
    calc_lorentzian_params()
    n_blocks = prepare_blocks()

	# Copy iter_params to device
	#gpuHandleError(cudaMemcpyToSymbol(iter_params_d, iter_params_h, sizeof(iterData)))
    # --> should we copy all variables under iterparams or just the arrays

    iter_params_d_blocks_line_offset = cp.array(iter_params_h_blocks_line_offset)
    iter_params_d_blocks_iv_offset = cp.array(iter_params_h_blocks_iv_offset)

	# Zero DLM:
	#gpuHandleError(cudaMemset(host_params_h_DLM_d, 0, 2 * (init_params_h_N_v + 1) * init_params_h_N_wG_x_N_wL * sizeof(float)))
    # cp.cuda.runtime.memset(host_params_h_DLM_d, 0, 2*(init_params_h_N_v+1)*init_params_h_N_wG_x_N_wL * getsizeof(float()))
    host_params_h_DLM_d = cp.zeros(2*(init_params_h_N_v+1)*init_params_h_N_wG_x_N_wL, dtype=cp.float32)

    print("Getting ready...")
	# Launch Kernel:
    host_params_h_start_DLM.record()

	# from population calculation to calculating the line set
	fillDLM ((n_blocks,), (init_params_h_N_threads_per_block,), host_params_h_shared_size (
		host_params_h_v0_d,
		host_params_h_da_d,
		host_params_h_S0_d,
		host_params_h_El_d,
		host_params_h_log_2gs_d,
		host_params_h_na_d,
		host_params_h_log_2vMm_d,
		host_params_h_DLM_d))

    host_params_h_stop_DLM.record()
    cp.cuda.runtime.eventSynchronize(host_params_h_stop_DLM_ptr)
    host_params_h_elapsedTimeDLM = cp.cuda.get_elapsed_time(host_params_h_start_DLM, host_params_h_stop_DLM)
    print("<<<LAUNCHED>>> ")

    cp.cuda.runtime.deviceSynchronize()

	# FFT
	#cufftExecR2C(host_params_h_plan_DLM, host_params_h_DLM_d_in, host_params_h_DLM_d_out)
	cp.cuda.runtime.deviceSynchronize()

    cdef int n_threads = 1024
    n_blocks = (init_params_h_N_v + 1) / n_threads + 1

    applyLineshapes (( n_blocks,), (n_threads,), (host_params_h_DLM_d_out, host_params_h_spectrum_d_in))
    cp.cuda.runtime.deviceSynchronize()

	# inverse FFT
	#cufftExecC2R(host_params_h_plan_spectrum, host_params_h_spectrum_d_in, host_params_h_spectrum_d_out)
    cp.cuda.runtime.deviceSynchronize()

	#gpuHandleError(cudaMemcpy(spectrum_h, host_params_h_spectrum_d, init_params_h_N_v * sizeof(float), cudaMemcpyDeviceToHost))
    spectrum_h = host_params_h_spectrum_d.get()
	# end of voigt broadening
	# spectrum_h is the k nu
	
    host_params_h_stop.record()
    cp.cuda.runtime.eventSynchronize(host_params_h_stop_ptr)
    host_params_h_elapsedTime = cp.cuda.get_elapsed_time(host_params_h_start, host_params_h_stop)

	#cout << "(" << elapsedTime << " ms)" << endl;
    print("[rG = {0}%".format(np.exp(iter_params_h_log_dwG) - 1) * 100, end = " ")
    print("rL = {0}%]".format(np.exp(iter_params_h_log_dwL) - 1) * 100 )
    print("Runtime: {0}".format(host_params_h_elapsedTimeDLM))
    print(" + {0}".format(host_params_h_elapsedTime - host_params_h_elapsedTimeDLM), end = " ")
    print(" = {0} ms".format(host_params_h_elapsedTime))


    return

def start():
    dir_path = '/home/pankaj/radis-lab/'

    v0 = np.array(0,dtype="float")
    da = np.array(0,dtype="float")
    S0 = np.array(0,dtype="float")
    El = np.array(0,dtype="float")
    log_2vMm = np.array(0,dtype="float")
    na = np.array(0,dtype="float")
    log_2gs = np.array(0,dtype="float")
    v0_dec = np.array(0,dtype="float")
    da_dec = np.array(0,dtype="float")

    spectrum_h = np.array(0,dtype="float")
    v_arr = np.array(0,dtype="float")

    init_params_h_v_min = 1750.0
    init_params_h_v_max = 2400.0
    init_params_h_dv = 0.002
    init_params_h_N_v = int((init_params_h_v_max - init_params_h_v_max)/init_params_h_dv)

    init_params_h_N_wG = 4
    init_params_h_N_wL = 8 
    np.resize(spectrum_h, init_params_h_N_v)
    v_arr = [init_params_h_v_min + i * init_params_h_dv for i in range(init_params_h_N_v)]

    init_params_h_Max_iterations_per_thread = 1024
    init_params_h_block_preparation_step_size = 128

    host_params_h_shared_size = 0x8000          # Bytes - Size of the shared memory
    host_params_h_Min_threads_per_block = 128   # Ensures a full warp from each of the 4 processors
    host_params_h_Max_threads_per_block = 1024  # Maximum determined by device parameters
    init_params_h_shared_size_floats = host_params_h_shared_size / sys.getsizeof(float());

    init_params_h_N_wG_x_N_wL = init_params_h_N_wG * init_params_h_N_wL;
    init_params_h_N_total = init_params_h_N_wG_x_N_wL * init_params_h_N_v;
    init_params_h_N_points_per_block = init_params_h_shared_size_floats / init_params_h_N_wG_x_N_wL;
    init_params_h_N_threads_per_block = 1024;
    init_params_h_N_blocks_per_grid = 4 * 256 * 256;
    init_params_h_N_points_per_thread = init_params_h_N_points_per_block / init_params_h_N_threads_per_block;

    print()
    print("Spectral points per block  : {0}".format(init_params_h_N_points_per_block))
    print("Threads per block          : {0}".format(init_params_h_N_threads_per_block))
    print("Spectral points per thread : {0}".format(init_params_h_N_points_per_thread))
    print()

    # init v:
    print("Init v : ")
    init_params_h_Max_lines = int(2.4E8)
    read_npy(dir_path+'v0.npy', v0)
    spec_h_v0 = v0
    read_npy(dir_path+'da.npy', da)
    spec_h_da = da

    #decimate (v0, v0_dec, init_params_h_N_threads_per_block)
    v0_dec = np.minimum.reduceat(v0, np.arange(0, len(v0), init_params_h_N_threads_per_block))
    host_params_h_v0_dec = v0_dec
    host_params_h_dec_size = len(v0_dec)
    #decimate (da, da_dec, init_params_h_N_threads_per_block)
    da_dec = np.minimum.reduceat(da, np.arange(0, len(da), init_params_h_N_threads_per_block))
    host_params_h_da_dec = da_dec
    print()

    # wL inits
    print("Init wL: ")
    read_npy(dir_path + 'log_2gs.npy', log_2gs)
    spec_h_log_2gs = log_2gs
    read_npy(dir_path + 'na.npy', na)
    spec_h_na = na
    init_lorentzian_params(log_2gs, na)
    print()

    # wG inits:
    print("Init wG: ")
    read_npy(dir_path + 'log_2vMm.npy', log_2vMm)
    spec_h_log_2vMm = log_2vMm
    init_gaussian_params(log_2vMm)
    print()

    # I inits:
    print("Init I: ")
    read_npy(dir_path + 'S0.npy', S0)
    spec_h_S0 = S0
    read_npy(dir_path + 'El.npy', El)
    spec_h_El = El
    print()

    init_params_h_N_lines = int(len(v0))
    print("Number of lines loaded: {0}".format(init_params_h_N_lines))
    print()

    print("Allocating device memory...")

    # start the CUDA work


	#gpuHandleError(cudaSetDevice(0));
    #gpuHandleError(cudaFuncSetAttribute(fillDLM, cudaFuncAttributeMaxDynamicSharedMemorySize, host_params_h.shared_size));

    # not needed as declared in the global space ----------------->

	#gpuHandleError(cudaEventCreate(&host_params_h.start));
	#gpuHandleError(cudaEventCreate(&host_params_h.start_DLM));
	#gpuHandleError(cudaEventCreate(&host_params_h.stop));
	#gpuHandleError(cudaEventCreate(&host_params_h.stop_DLM));


	#Device memory allocations:
	#TARGET: These will all be made redundant by CuPy memory transfers
	# gpuHandleError(cudaMalloc((void**)&host_params_h.v0_d, init_params_h.N_lines * sizeof(float)));
	# gpuHandleError(cudaMalloc((void**)&host_params_h.da_d, init_params_h.N_lines * sizeof(float)));
	# gpuHandleError(cudaMalloc((void**)&host_params_h.S0_d, init_params_h.N_lines * sizeof(float)));
	# gpuHandleError(cudaMalloc((void**)&host_params_h.El_d, init_params_h.N_lines * sizeof(float)));
	# gpuHandleError(cudaMalloc((void**)&host_params_h.log_2gs_d, init_params_h.N_lines * sizeof(float)));
	# gpuHandleError(cudaMalloc((void**)&host_params_h.na_d, init_params_h.N_lines * sizeof(float)));
	# gpuHandleError(cudaMalloc((void**)&host_params_h.log_2vMm_d, init_params_h.N_lines * sizeof(float)));

    # <------------------------------------------------------------

	#gpuHandleError(cudaMalloc((void**)&init_params_d, sizeof(initData)));
	#gpuHandleError(cudaMalloc((void**)&iter_params_d, sizeof(iterData)));

	# DLM is allocated once, but must be zero'd every iteration ---> not needed anymore
	#host_params_h_DLM_d = cp.cuda.runtime.malloc(2 * (init_params_h.N_v + 1) * init_params_h.N_wG_x_N_wL * getsizeof(float()))
	#host_params_h.DLM_d_in = (cufftReal*)host_params_h.DLM_d;
	#host_params_h.DLM_d_out = (cufftComplex*)host_params_h.DLM_d;

	#gpuHandleError(cudaMalloc((void**)&host_params_h.spectrum_d, 2 * (init_params_h.N_v + 1) * sizeof(float)));
	#host_params_h.spectrum_d_in = (cufftComplex*)host_params_h.spectrum_d;
	#host_params_h.spectrum_d_out = (cufftReal*)host_params_h.spectrum_d;
	print("Done!")


	# Copy params to device
	print("Copying data to device... ")
	# gpuHandleError(cudaMemcpyToSymbol(init_params_d, &init_params_h, sizeof(initData)));
    # ---> should I copy all the struct fields or let it be

	#Copy spectral data to device
    host_params_h_v0_d = cp.array(spec_h_v0)
    host_params_h_da_d = cp.array(spec_h_da)
    host_params_h_S0_d = cp.array(spec_h_S0)
    host_params_h_El_d = cp.array(spec_h_El)
    host_params_h_log_2gs_d = cp.array(spec_h_log_2gs)
    host_params_h_na_d = cp.array(spec_h_na)
    host_params_h_log_2vMm_d = cp.array(spec_h_log_2vMm)
	print("Done!")

	print("Planning FFT's... ")
	# Plan DLM FFT
	cdef int n_fft[] = { 2 * init_params_h_N_v };
	# This can be replaced by CuPy.fft
	cufftCreate(&host_params_h_plan_DLM);
	cufftPlanMany(&host_params_h_plan_DLM, 1, n_fft,
		n_fft, init_params_h_N_wG_x_N_wL, 1,
		n_fft, init_params_h_N_wG_x_N_wL, 1, CUFFT_R2C, init_params_h_N_wG_x_N_wL);

	cufftCreate(&host_params_h_plan_spectrum);
	cufftPlan1d(&host_params_h_plan_spectrum, n_fft[0], CUFFT_C2R, 1);
	print("Done!")
	print("Press any key to start iterations...")
	_ = input()


    # START ITERATIONS

    p = 0.1
    T = 2000.0

    T_min = 500.0
    T_max = 5000.0
    dT = 500.0

    for T in range(T_min, T_max, dT):
        iterate(p, T, spectrum_h)
    return

    #Cleanup and go home:
	#cudaEventDestroy(host_params_h_start);
	#cudaEventDestroy(host_params_h_start_DLM);
	#cudaEventDestroy(host_params_h_stop);
	#cudaEventDestroy(host_params_h_stop_DLM);

	#host_params_h_DLM_d_in = NULL;
	#host_params_h_DLM_d_out = NULL;
	#cufftDestroy(host_params_h_plan_DLM);
	#cufftDestroy(host_params_h_plan_spectrum);
	#free(DLM_h);
	#gpuHandleError(cudaDeviceReset());






########## JUNK: INGORE ALL THIS #################3
# P.S. kept for syntax/reference


#cdef  void c_init_lorentzian_params(vector[float] log_2gS, vector[float] na):

# def set_val():
#     global iter_params_h_log_wL_min 
#     iter_params_h_log_wL_min = -69
#     return

# def get_val():
#     global iter_params_h_log_wL_min 
#     print("the value you are looking for = {0}".format(iter_params_h_log_wL_min))
#     return iter_params_h_log_wL_min




# cdef   from "c_struct.c":
#     ctypedef struct spectralData:
#         float* v0
#         float* da
#         float* S0
#         float* El
#         float* log_2vMm
#         float* na
#         float* log_2gs

#     ctypedef struct initData:
#         float v_min
#         float v_max	# Host only
#         float dv

#         # DLM sizes:
#         int N_v
#         int N_wG
#         int N_wL
#         int N_wG_x_N_wL
#         int N_total

#         # Work parameters :
#         int Max_lines
#         int N_lines
#         int N_points_per_block
#         int N_threads_per_block
#         int N_blocks_per_grid
#         int N_points_per_thread
#         int	Max_iterations_per_thread

#         int shared_size_floats
    
#     ctypedef struct blockData:
#         int line_offset
#         # int N_iterations
#         int iv_offset
    
#     ctypedef struct iterData:
#         # Any info needed for the Kernel that does not change during 
#         # kernel execution but MAY CHANGE during spectral iteration step

#         # Pressure & temperature:
#         float p
#         # float T
#         float log_p
#         float hlog_T
#         float log_rT
#         float c2T
#         float rQ

#         # Spectral parameters:
#         float log_wG_min
#         float log_wL_min
#         float log_dwG
#         float log_dwL

#         # Block data:
#         blockData blocks[4096] 
    
#     spectralData init_spectralData()


# # # Then we describe a class that has a Point member "pt"
# cdef  class py_spectralData:
#     cdef  spectralData sd

#     def __init__(self):
#         self.sd = init_spectralData()
#         self.sd.v0 = NULL
#         self.sd.da = NULL
#         self.sd.S0 = NULL
#         self.sd.El = NULL
#         self.sd.log_2vMm = NULL
#         self.sd.na = NULL
#         self.sd.log_2gs = NULL

#     @v0.setter
#     def v0(self,val):
#     #    cdef  float* v0_ptr = <float*> val.data
#         self.sd.v0 = <float *> val.data
    
#     # @property
#     # def v0(self):
#     #     cdef  float* v0_ptr = <float*> self.sd.v0.data
#     #     return v0_ptr

# cdef   from "point.c":
#     ctypedef struct Point:
#         int x
#         int y
#     Point make_and_send_point(int x, int y)

# cdef  class PyPoint:
#     cdef  Point p

#     def __init__(self, x, y):
#         self.p = make_and_send_point(x, y)

#      # define properties in the normal Python way
#     @property
#     def x(self):
#         return self.p.x

#     @x.setter
#     def x(self,val):
#         self.p.x = val

#     # @property
#     # def y(self):
#     #     return self.pt.y

#     # @y.setter
#     # def y(self,val):
#     #     self.pt.y = val