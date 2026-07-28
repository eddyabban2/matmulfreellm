#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace mmfreelm {

template <typename T>
__global__ void ternaryMatmulKernel(
    const T* __restrict__ x,
    const int8_t* __restrict__ w,
    T* __restrict__ y,
    int m,
    int n,
    int k,
    T scale) {
  int row = blockIdx.x;
  int col = threadIdx.x;
  if (row >= m || col >= n) {
    return;
  }
  float acc = 0.f;
  const T* xrow = x + row * k;
  const int8_t* wrow = w + col * k;
  for (int i = 0; i < k; ++i) {
    acc += static_cast<float>(xrow[i]) * static_cast<float>(wrow[i]);
  }
  y[row * n + col] = static_cast<T>(acc * static_cast<float>(scale));
}

void launchTernaryMatmul(
    const void* x,
    const void* w,
    void* y,
    int m,
    int n,
    int k,
    float scale,
    cudaDataType_t dtype,
    cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(std::min(n, 256));
  if (dtype == CUDA_R_16F) {
    ternaryMatmulKernel<<<grid, block, 0, stream>>>(
        static_cast<const half*>(x),
        static_cast<const int8_t*>(w),
        static_cast<half*>(y),
        m,
        n,
        k,
        __float2half(scale));
  } else {
    ternaryMatmulKernel<<<grid, block, 0, stream>>>(
        static_cast<const float*>(x),
        static_cast<const int8_t*>(w),
        static_cast<float*>(y),
        m,
        n,
        k,
        scale);
  }
}

}  // namespace mmfreelm
