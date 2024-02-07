#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <string>
#include <thrust/binary_search.h>
#include <thrust/functional.h>

#include "TError.h"

#define ERRCHECK(err) __checkCudaErrors((err), __func__, __FILE__, __LINE__)
inline static void __checkCudaErrors(cudaError_t error, std::string func, std::string file, int line)
{
   if (error != cudaSuccess) {
      Fatal((func + "(), " + file + ":" + std::to_string(line)).c_str(), "%s", cudaGetErrorString(error));
      throw std::bad_alloc();
   }
}

namespace ROOT {
namespace Experimental {
namespace CUDAHelpers {

// Dynamic shared memory needs to be declared as "extern" in CUDA. Having templated kernels with shared memory
// of different data types results in a redeclaration error if the name of the array is the same, so we use a
// proxy function to initialize shared memory arrays of different types with different names.

template <typename T>
__device__ T *shared_memory_proxy()
{
   Fatal("template <typename T> __device__ T *shared_memory_proxy()", "Unsupported shared memory type");
   return (T *)0;
};

template <>
__device__ short *shared_memory_proxy<short>()
{
   extern __shared__ short s_short[];
   return s_short;
}

template <>
__device__ int *shared_memory_proxy<int>()
{
   extern __shared__ int s_int[];
   return s_int;
}

template <>
__device__ double *shared_memory_proxy<double>()
{
   extern __shared__ double s_double[];
   return s_double;
}

template <>
__device__ float *shared_memory_proxy<float>()
{
   extern __shared__ float s_float[];
   return s_float;
}

////////////////////////////////////////////////////////////////////////////////
/// Reduction operations.
template <typename T = double>
struct Plus {
   // Function call operator. The return value is <tt>lhs + rhs</tt>.
   __host__ __device__ constexpr T operator()(const T &lhs, const T &rhs) const { return lhs + rhs; }
};

struct Identity {};

struct Square
{
   __host__ __device__
   double operator ()(double x) { return x * x; }
};

struct Mul
{
   __host__ __device__
   double operator ()(double x, double y) { return x * y; }
};

struct MulSquare
{
   __host__ __device__
   double operator ()(double x, double y) { return x * y * y; }
};

struct Mul3
{
   __host__ __device__
   double operator ()(double x, double y, double z) { return x * y * z; }
};


////////////////////////////////////////////////////////////////////////////////
/// CUDA Kernels

// clang-format off
template <unsigned int BlockSize, typename T, typename Op>
__device__ inline void UnrolledReduce(T *sdata, unsigned int tid, Op operation)
{
   // 1024 is the maximum number of threads per block in an NVIDIA GPU:
   // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
   if (BlockSize >= 1024 && tid < 512) { sdata[tid] = operation(sdata[tid], sdata[tid + 512]); } __syncthreads();
   if (BlockSize >= 512  && tid < 256) { sdata[tid] = operation(sdata[tid], sdata[tid + 256]); } __syncthreads();
   if (BlockSize >= 256  && tid < 128) { sdata[tid] = operation(sdata[tid], sdata[tid + 128]); } __syncthreads();
   if (BlockSize >= 128  && tid < 64) { sdata[tid] = operation(sdata[tid], sdata[tid + 64]); } __syncthreads();

   // Reduction within a warp
   if (BlockSize >= 64 && tid < 32) { sdata[tid] = operation(sdata[tid], sdata[tid + 32]); } __syncthreads();
   if (BlockSize >= 32 && tid < 16) { sdata[tid] = operation(sdata[tid], sdata[tid + 16]); } __syncthreads();
   if (BlockSize >= 16 && tid < 8) { sdata[tid] = operation(sdata[tid], sdata[tid + 8]); } __syncthreads();
   if (BlockSize >= 8  && tid < 4) { sdata[tid] = operation(sdata[tid], sdata[tid + 4]); } __syncthreads();
   if (BlockSize >= 4  && tid < 2) { sdata[tid] = operation(sdata[tid], sdata[tid + 2]); } __syncthreads();
   if (BlockSize >= 2  && tid < 1) { sdata[tid] = operation(sdata[tid], sdata[tid + 1]); } __syncthreads();
}
// clang-format on

// See https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//     https://github.com/zchee/cuda-sample/blob/master/6_Advanced/reduction/reduction_kernel.cu
// The type of reduction in this kernel can be customized by passing different lambda operations
// as transformOp and reduceOp.
// The transformOp lambda function transforms the input values before the reduction step. The transformation can
// involve a variable number of input buffers.
// The reduceOp lambda function should take as input (T lhs, T rhs) and its the return value will be
// written to lhs.
template <unsigned int BlockSize, typename TOut, typename TOp, typename ROp, typename...TIn>
__global__ void TransformReduceKernel(unsigned int n, TOut *out, TOut init, TOp transformOp, ROp reduceOp, TIn...in)
{
   auto sdata = CUDAHelpers::shared_memory_proxy<TOut>();

   unsigned int local_tid = threadIdx.x;
   unsigned int i = blockIdx.x * (BlockSize * 2) + local_tid;
   unsigned int gridSize = (BlockSize * 2) * gridDim.x;

   TOut r = init;

   while (i < n) {
      if constexpr(std::is_same_v<TOp, CUDAHelpers::Identity>) {
         r = reduceOp(r, (in[i])...);
         if (i + BlockSize < n) {
            r = reduceOp(r, (in[i + BlockSize])...);
         }
      } else {
         r = reduceOp(r, transformOp((in[i])...));
         if (i + BlockSize < n) {
            r = reduceOp(r, transformOp((in[i + BlockSize])...));
         }
      }
      i += gridSize;
   }
   sdata[local_tid] = r;
   __syncthreads();

   CUDAHelpers::UnrolledReduce<BlockSize, TOut>(sdata, local_tid, reduceOp);

   // The first thread of each block writes the sum of the block into the global device array.
   if (local_tid == 0) {
      out[blockIdx.x] = reduceOp(out[blockIdx.x], sdata[0]);
   }
}

template <typename TOut, typename TOp, typename ROp, typename... TIn>
void TransformReduce(std::size_t numBlocks, std::size_t blockSize, std::size_t n,
                     TOut *out, TOut init, ROp reduceOp, TOp transformOp, TIn ...in)
{
   auto smemSize = (blockSize <= 32) ? 2 * blockSize * sizeof(double) : blockSize * sizeof(double);

   if (blockSize == 1)
      TransformReduceKernel<1, TOut><<<numBlocks, 1, smemSize>>>(n, out, init, transformOp, reduceOp, in...);
   else if (blockSize == 2)
      TransformReduceKernel<2, TOut><<<numBlocks, 2, smemSize>>>(n, out, init, transformOp, reduceOp, in...);
   else if (blockSize == 4)
      TransformReduceKernel<4, TOut><<<numBlocks, 4, smemSize>>>(n, out, init, transformOp, reduceOp, in...);
   else if (blockSize == 8)
      TransformReduceKernel<8, TOut><<<numBlocks, 8, smemSize>>>(n, out, init, transformOp, reduceOp, in...);
   else if (blockSize == 16)
      TransformReduceKernel<16, TOut><<<numBlocks, 16, smemSize>>>(n, out, init, transformOp, reduceOp, in...);
   else if (blockSize == 32)
      TransformReduceKernel<32, TOut><<<numBlocks, 32, smemSize>>>(n, out, init, transformOp, reduceOp, in...);
   else if (blockSize == 64)
      TransformReduceKernel<64, TOut><<<numBlocks, 64, smemSize>>>(n, out, init, transformOp, reduceOp, in...);
   else if (blockSize == 128)
      TransformReduceKernel<128, TOut><<<numBlocks, 128, smemSize>>>(n, out, init, transformOp, reduceOp, in...);
   else if (blockSize == 256)
      TransformReduceKernel<256, TOut><<<numBlocks, 256, smemSize>>>(n, out, init, transformOp, reduceOp, in...);
   else if (blockSize == 512)
      TransformReduceKernel<512, TOut><<<numBlocks, 512, smemSize>>>(n, out, init, transformOp, reduceOp, in...);
   else if (blockSize == 1024)
      TransformReduceKernel<1024, TOut><<<numBlocks, 1024, smemSize>>>(n, out, init, transformOp, reduceOp, in...);
   else
      Error("TransformReduce", "Unsupported block size: %lu", blockSize);
}

// CUDA version of TMath::BinarySearch
template <typename T>
__device__ Long64_t BinarySearch(Long64_t n, const T *array, T value)
{
   const T *pind;

   pind = thrust::lower_bound(thrust::seq, array, array + n, value);

   if ((pind != array + n) && (*pind == value))
      return (pind - array);
   else
      return (pind - array - 1);

   // return pind - array - !((pind != array + n) && (*pind == value)); // OPTIMIZATION: is this better?
}

// For debugging
__global__ void PrintArray(double *array, int n)
{
   if (threadIdx.x == 0) {
      for (int i = 0; i < n; i++) {
         printf("%f ", array[i]);
      }
      printf("\n");
   }
}

} // namespace CUDAHelpers
} // namespace Experimental
} // namespace ROOT
#endif
