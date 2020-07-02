import cupy as cp
import numpy as np
import scipy

pankaj_mishra = 10
loaded_from_source = r'''
extern "C"{
__constant__ float adder[1];

__global__ void test_sum(const float* x1, const float* x2, float* y, int* c, \
                        unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // int pankaj_mishra;
    if (tid < N)
    {
        y[tid] = x1[tid] + c[0]; //x2[tid];
        // pankaj_mishra = 10;
        // printf("the current thread is %d and I am pankaj = %d\n", tid, pankaj_mishra);
    }
}

__global__ void test_multiply(const float* x1, const float* x2, float* y, \
                            unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
    {
        y[tid] = x1[tid] * x2[tid];
    }
}

}'''

module = cp.RawModule(code=loaded_from_source)
ker_sum = module.get_function('test_sum')
ker_times = module.get_function('test_multiply')
# adder_ptr = module.get_global('adder')

# const_arr = cp.ndarray((1, ), cp.float32, adder_ptr)
# data = cp.arange(1, dtype=cp.float32)
# const_arr[...] = data

N = 10
cnt = 0
y1 = np.arange(N**2, dtype=cp.float32).reshape(N, N)
for i in range(N):
    for j in range(N):
        y1[i][j] = cnt
        cnt += 1

x1 = cp.asarray(y1)
c_d = 5
c = cp.array(c_d)
x2 = cp.ones((N, N), dtype=cp.float32)
y = cp.zeros((N, N), dtype=cp.float32)
ker_sum((N,), (N,), (x1, x2, y, c, N**2))   # y = x1 + x2

print("x1: ")
print(x1)
print()
print("x2: ")
print(x2)
print()
print("y: ")
print(y)
print()

if cp.allclose(y, x1 + x2):
    print("sum matches...")
else:
    print("sum not matching...")

ker_times((N,), (N,), (x1, x2, y, N**2)) # y = x1 * x2
if cp.allclose(y, x1 * x2):
    print("times matches...")
else:
    print("times not matching!")