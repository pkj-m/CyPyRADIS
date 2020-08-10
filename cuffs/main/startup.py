import py_cuffs
import numpy as np

dir_path = '/home/pankaj/radis-lab/data-2000-2400/'

v0 = np.load(dir_path + 'v0.npy')
da = np.load(dir_path + 'da.npy')
log_2gs = np.load(dir_path + 'log_2gs.npy')
S0 = np.load(dir_path + 'S0.npy')
El = np.load(dir_path + 'El.npy')
log_2vMm = np.load(dir_path + 'log_2vMm.npy')
na = np.load(dir_path + 'na.npy')

varr = np.arange(2200, 2400 + 0.001, 0.001)

py_cuffs.init(varr, 4, 8, v0, da, log_2gs, na, log_2vMm, S0, El)
