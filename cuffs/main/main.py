import py_cuffs as pyc
import sys
import numpy as np
from HITEMP_path import HITEMP_path

def read_npy(fname, arr):
    print("Loading {0}...".format(fname))
    arr = np.load(fname)
    print("Done!")

def main():
    dir_path = HITEMP_path

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
    # 
    # LINE 1103
    # .
    # .
    # .
    # LINE 1164    


    # START ITERATIONS

    p = 0.1
    T = 2000.0

    T_min = 500.0
    T_max = 5000.0
    dT = 500.0

    for T in range(T_min, T_max, dT):
        iterate(p, T, spectrum_h)

if __name__ == '__main__':
    main()

