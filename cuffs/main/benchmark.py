import numpy as np
import py_cuffs
import matplotlib.pyplot as plt
import cupy as cp
from matplotlib.widgets import Slider
import sys
import fss_py3 as fss_py
import HITEMP_spectra

v_min,v_max = 1800.0,2400.0 #cm-1
dv = 0.002 #cm-1
v_arr = np.arange(v_min,v_max,dv)
N_v = len(v_arr)
N_wG, N_wL = 4, 8


HT_folder = 'C:/HITEMP/'

fname_list = [HT_folder + '02_2125-2250_HITEMP2010.par',
              HT_folder + '02_2250-2500_HITEMP2010.par']

HITEMP_spectra.init_database(fname_list)

py_cuffs.set_path('C:/CDSD4000/npy/')
py_cuffs.set_N_lines(int(2.2E8))
py_cuffs.init(v_arr,N_wG,N_wL)


p = 0.1 #bar
T = 1000.0 #K

spectrum_h = py_cuffs.iterate(p, T)
p1, = plt.plot(v_arr, spectrum_h, "-",linewidth=1)

v_dat,log_wG_dat,log_wL_dat,I_dat = HITEMP_spectra.calc_stick_spectrum(p,T)
Idlm,DLM = fss_py.spectrum(v_arr,N_wG,N_wL,v_dat,log_wG_dat,log_wL_dat,I_dat)
p2, = plt.plot(v_arr, Idlm, "--",linewidth=1)

sum_d = np.sum(py_cuffs.DLM[:N_v],(1,2))
sum_h = np.sum(DLM[:N_v],(1,2))
##p1,=plt.plot(v_arr,sum_d,label='GPU')
##p2,=plt.plot(v_arr,sum_h,'--',label='CPU')
##


##print('I_ratio:',np.max(sum_d)/np.max(sum_h))
print('I_ratio:',np.max(spectrum_h)/np.max(Idlm))

plt.xlim(2400,2200)
plt.subplots_adjust(bottom=0.20)

axcolor = 'lightgoldenrodyellow'
temp_ax = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor=axcolor)
temp_slider = Slider(temp_ax, 'T(K)', 50, 5000, valinit=T)

def update(val):
    T = temp_slider.val
    
    spectrum_h = py_cuffs.iterate(p, T)
    sum_d = np.sum(py_cuffs.DLM[:N_v],(1,2))
##    p1.set_ydata(sum_d )
    p1.set_ydata(spectrum_h)

    v_dat,log_wG_dat,log_wL_dat,I_dat = HITEMP_spectra.calc_stick_spectrum(p,T)
    Idlm,DLM = fss_py.spectrum(v_arr,N_wG,N_wL,v_dat,log_wG_dat,log_wL_dat,I_dat)
    sum_h = np.sum(DLM[:N_v],(1,2))
##    p2.set_ydata(sum_h)
    p2.set_ydata(Idlm)

##    print('DLM_ratio:',np.max(sum_d)/np.max(sum_h))
    print('I_ratio:',np.max(spectrum_h)/np.max(Idlm))

    plt.gcf().canvas.draw_idle()

temp_slider.on_changed(update)


plt.show()

cp._default_memory_pool.free_all_blocks()
