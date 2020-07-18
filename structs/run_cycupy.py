from cycupy_test import gpu_add
import numpy as np
from ctypes import c_void_p

a = 1
b = 2
x = 5
y = 10

x1 = np.arange(25, dtype=np.float32).reshape(5, 5)
x2 = np.arange(25, dtype=np.float32).reshape(5, 5)
l = np.arange(25, dtype=np.int32)

res_h = (x1 + x2 + a + y + x) * b
res_d = gpu_add(x1,x2,a,b,x,y,l)

print(res_h)
print(res_d)

print(("\nFailed!" if np.sum(np.abs(res_h-res_d)) else '\nPassed!'))


