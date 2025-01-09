#ifndef RDefH1SYCL_H
#define RDefH1SYCL_H

#include <array>
#include <optional>
#include <sycl/sycl.hpp>
#include "AxisDescriptor.h"
#include "ROOT/RVec.hxx"

#include "Math/GenVector/AccHeaders.h"
#include "Math/Vector4D.h"

typedef double (op)(double*, double*, std::size_t, std::size_t);

namespace ROOT {
namespace Experimental {

double IdentityKernel(double *buffer, double *parameters, std::size_t idx, std::size_t bulksize)
{
    return buffer[idx];
}

double InvariantMassesKernel(double *buffer, double *parameters, std::size_t idx, std::size_t bulksize)
{
    ROOT::ROOT_MATH_ARCH::PtEtaPhiEVector p1(buffer[idx], buffer[bulksize+idx], buffer[2*bulksize+idx], buffer[3*bulksize+idx]);
    ROOT::ROOT_MATH_ARCH::PtEtaPhiEVector p2(buffer[4*bulksize+idx], buffer[5*bulksize+idx], buffer[6*bulksize+idx], buffer[7*bulksize+idx]);
    return (p1+p2).M();
}

template <typename T, op Op, unsigned int nInput, unsigned int WGroupSize = 256>
class RDefH1SYCL {
private:
   // B = SYCL buffer
   // S = SYCL USM shared pointer
   // D = SYCL USM device pointer

   static constexpr unsigned int fDim = 1;
   static constexpr int kNStats = 2 +  fDim * 2 +  fDim * (fDim - 1) / 2; ///< Number of statistics.

   // clang-format off
   sycl::queue                       queue;
   std::vector<sycl::event>          prevBulk;

   T                                *fDHistogram;         ///< Pointer to histogram buffer on the GPU.
   int                               fNBins;              //< Total number of bins in the histogram WITH under/overflow

   int                              *fDNBinsAxis;         ///< Number of bins(1D) WITH u/overflow per axis
   double                           *fDMin;               ///< Low edge of first bin per axis
   double                           *fDMax;               ///< Upper edge of last bin per axis
   double                           *fDBinEdges;          ///< Bin edges array for each axis
   int                              *fDBinEdgesIdx;       ///< Start index of the binedges in kBinEdges per axis

   double                           *fDParameters;         ///< Pointer to array of parameters for computing fDCoords[idx]=Op(fDBuffer,fParameters,idx) on the GPU.
   double                           *fDCoords;         ///< Pointer to array of coordinates on the GPU.
   double                           *fDBuffer;            ///< Pointer to array of inputs for computing coordinates to be filled on the GPU.
   double                           *fDWeights;           ///< Pointer to array of weights on the GPU.
   int                              *fDBins;              ///< Pointer to array of bins (corresponding to the coordinates) to fill on the GPU.

   int                               fEntries;            ///< Number of entries that have been filled.
   double                           *fDStats;             ///< Pointer to statistics array on GPU.
   double                           *fDIntermediateStats; ///< Pointer to statistics array on GPU.

   // Kernel size parameters
   std::size_t                       fMaxBulkSize;        ///< Number of coordinates to buffer.
   std::size_t                       fMaxSmemSize;        ///< Maximum shared memory size per block on device 0.
   std::size_t const                 kStatsSmemSize;      ///< Size of shared memory per block in GetStatsKernel
   std::size_t                       fHistoSmemSize;      ///< Size of shared memory per block in HistoKernel
   // clang-format on

public:
   RDefH1SYCL() = delete;

   RDefH1SYCL(std::size_t maxBulkSize, const std::size_t nBins, const std::array<int,  fDim> &nBinsAxis,
           const std::array<double,  fDim> &xLow, const std::array<double,  fDim> &xHigh,
           const std::vector<double> &binEdges, const std::array<int,  fDim> &binEdgesIdx, 
           const std::vector<double> &parameters);

   ~RDefH1SYCL()
   {
      sycl::free(fDHistogram, queue);
      sycl::free(fDNBinsAxis, queue);
      sycl::free(fDMin, queue);
      sycl::free(fDMax, queue);
      sycl::free(fDBinEdgesIdx, queue);
      sycl::free(fDBuffer, queue);
      sycl::free(fDParameters, queue);
      sycl::free(fDCoords, queue);
      sycl::free(fDWeights, queue);
      sycl::free(fDBins, queue);
      sycl::free(fDStats, queue);
      if (fDIntermediateStats != NULL)
         sycl::free(fDIntermediateStats, queue);
      if (fDBinEdges != NULL)
         sycl::free(fDBinEdges, queue);
   }

   RDefH1SYCL(const RDefH1SYCL &) = delete;
   RDefH1SYCL &operator=(const RDefH1SYCL &) = delete;

   int GetEntries() const { return fEntries; }

   void RetrieveResults(T *histResult, double *statsResult);

   void Fill(const RVecD &buffer);

   void Fill(const RVecD &buffer, const RVecD &weights);

   std::size_t GetMaxBulkSize() { return fMaxBulkSize; }

private:
   void GetStats(std::size_t size, sycl::event &fillEvent);

   void ExecuteSYCLHisto(std::size_t size, std::vector<sycl::event> &depends);
};

} // namespace Experimental
} // namespace ROOT
#endif
