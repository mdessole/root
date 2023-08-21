#include "InvariantMassCUDA.h"
#include "Math/Vector4D.h"

#include "TError.h"

#define ERRCHECK(err) __checkCudaErrors((err), __func__, __FILE__, __LINE__)
inline static void __checkCudaErrors(cudaError_t error, std::string func, std::string file, int line)
{
   if (error != cudaSuccess) {
      Fatal((func + "(), " + file + ":" + std::to_string(line)).c_str(), "%s", cudaGetErrorString(error));
      throw std::bad_alloc();
   }
}

using ROOT::Math::PtEtaPhiEVector;

namespace ROOT {
namespace Experimental {

struct PtEtaPhiE4DDevice {
   double fPt, fEta, fPhi, fE;
};

__global__ void
InvariantMassesKernel(const PtEtaPhiE4DDevice *v1, const PtEtaPhiE4DDevice *v2, size_t size, double *result)
{
   unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
   unsigned int stride = blockDim.x * gridDim.x;
   for (int i = tid; i < size; i += stride) {
      // Conversion from (pt, eta, phi, mass) to (x, y, z, e) coordinate system
      const auto x1 = v1[i].fPt * cos(v1[i].fPhi);
      const auto y1 = v1[i].fPt * sin(v1[i].fPhi);
      const auto z1 = v1[i].fPt * sinh(v1[i].fEta);
      const auto e1 = v1[i].fE;

      const auto x2 = v2[i].fPt * cos(v2[i].fPhi);
      const auto y2 = v2[i].fPt * sin(v2[i].fPhi);
      const auto z2 = v2[i].fPt * sinh(v2[i].fEta);
      const auto e2 = v2[i].fE;

      // Addition of particle four-vector elements
      const auto e = e1 + e2;
      const auto x = x1 + x2;
      const auto y = y1 + y2;
      const auto z = z1 + z2;

      auto mm = e * e - x * x - y * y - z * z;
      result[i] = mm < 0 ? -sqrt(-mm) : sqrt(mm);
   }
}

template <unsigned int BlockSize>
double *
InvariantMassCUDA<BlockSize>::ComputeInvariantMasses(const PtEtaPhiEVector *v1, const PtEtaPhiEVector *v2, size_t size)
{
   const int numBlocks = ceil(size / float(BlockSize));

   PtEtaPhiE4DDevice *dV1 = NULL;
   ERRCHECK(cudaMalloc((void **)&dV1, size * sizeof(PtEtaPhiE4DDevice)));

   PtEtaPhiE4DDevice *dV2 = NULL;
   ERRCHECK(cudaMalloc((void **)&dV2, size * sizeof(PtEtaPhiE4DDevice)));

   double *dResult = NULL;
   ERRCHECK(cudaMalloc((void **)&dResult, size * sizeof(double)));

   // NOTE: his assumes that data layout of PtEtaPhiEVector is the same as PtEtaPhiE4DDevice...
   ERRCHECK(cudaMemcpy(dV1, v1, size * sizeof(PtEtaPhiE4DDevice), cudaMemcpyHostToDevice));
   ERRCHECK(cudaMemcpy(dV2, v2, size * sizeof(PtEtaPhiE4DDevice), cudaMemcpyHostToDevice));

   InvariantMassesKernel<<<numBlocks, BlockSize>>>(dV1, dV2, size, dResult);
   cudaDeviceSynchronize();
   ERRCHECK(cudaPeekAtLastError());

   double *result = (double *)malloc(size * sizeof(double));
   ERRCHECK(cudaMemcpy(result, dResult, size * sizeof(double), cudaMemcpyDeviceToHost));
   return result;
}

// Template instantations
template class InvariantMassCUDA<512>;
template class InvariantMassCUDA<256>;
template class InvariantMassCUDA<128>;
template class InvariantMassCUDA<64>;

} // namespace Experimental
} // namespace ROOT
