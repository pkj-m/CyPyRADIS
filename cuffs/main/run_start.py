import numpy as np
import py_cuffs
import matplotlib.pyplot as plt
import cupy as cp
from matplotlib.widgets import Slider, Button, RadioButtons

v_min,v_max = 2000.0,2400.0 #cm-1
dv = 0.002 #cm-1
v_arr = np.arange(v_min,v_max,dv)
NwG, NwL = 4, 8

py_cuffs.set_path('C:/CDSD4000/npy/')
py_cuffs.set_N_lines(int(2.4E7))
py_cuffs.init(v_arr,NwG,NwL)

p = 0.1 #bar
T = 1000.0 #K
T_min = 300
T_max = 5000
DT = 500

##for T in range(T_min, T_max, DT):
spectrum_h = py_cuffs.iterate(p, T)
p1, = plt.plot(v_arr, spectrum_h, "-")
plt.xlim(v_max,v_min)
plt.yscale('log')
plt.subplots_adjust(bottom=0.20)


axcolor = 'lightgoldenrodyellow'
temp_ax = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor=axcolor)
temp_slider = Slider(temp_ax, 'T(K)', T_min, T_max, valinit=T)

def update(val):
    T = temp_slider.val
    spectrum_h = py_cuffs.iterate(p, T)
    p1.set_ydata(spectrum_h)
    plt.gcf().canvas.draw_idle()

temp_slider.on_changed(update)

plt.show()
cp._default_memory_pool.free_all_blocks()
