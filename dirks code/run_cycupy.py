from cycupy_test import gpu_add
import numpy as np

a = 1.0
b = 2.0

x1 = np.arange(25, dtype=np.float32).reshape(5, 5)
x2 = np.arange(25, dtype=np.float32).reshape(5, 5)

res_h = a*(x1 + x2) + b
res_d = gpu_add(x1,x2,a,b)

print(res_h)
print(res_d)

print(("\nFailed!" if np.sum(np.abs(res_h-res_d)) else '\nPassed!'))


