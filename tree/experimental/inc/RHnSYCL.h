#ifndef RHnSYCL_H
#define RHnSYCL_H

#include <array>
#include <optional>
#include <sycl/sycl.hpp>
#include "AxisDescriptor.h"
#include "ROOT/RVec.hxx"

namespace ROOT {
namespace Experimental {

template <typename T, unsigned int Dim, unsigned int WGroupSize = 256>
class RHnSYCL {
private:
   // B = SYCL buffer
   // S = SYCL USM shared pointer
   // D = SYCL USM device pointer

   static constexpr int kNStats = 2 + Dim * 2 + Dim * (Dim - 1) / 2; ///< Number of statistics.

   // clang-format off
   sycl::queue                                     queue;

   std::optional<sycl::buffer<T, 1>>               fBHistogram;        ///< Pointer to histogram buffer on the GPU.
   int                                             fNbins;             ///< Total number of bins in the histogram WITH under/overflow

   std::optional<sycl::buffer<AxisDescriptor, 1>>  fBAxes;             ///< Vector of Dim axis descriptors
   double                                         *fDBinEdges;         ///< Binedges per axis for non-fixed bins. TODO: remove binedges from AxisDescriptor

   std::optional<sycl::buffer<double, 1>>          fBCoords;           ///< 1D buffer with bufferSize #Dim-dimensional coordinates to fill.
   std::optional<sycl::buffer<double, 1>>          fBWeights;          ///< Buffer of weigths for each bin on the Host.
   std::optional<sycl::buffer<int, 1>>             fBBins;             ///< Pointer to array of bins (corresponding to the coordinates) to fill on the GPU.

   int                                             fEntries;           ///< Number of entries that have been filled.
   std::optional<sycl::buffer<double, 1>>          fBStats;            ///< Pointer to statistics array on GPU.
   double                                         *fDIntermediateStats;///< Pointer to statistics array on GPU.

   // Kernel size parameters
   unsigned int                                    fMaxBulkSize;       ///< Number of coordinates to buffer.
   unsigned int                                    fMaxSmemSize;       ///< Maximum shared memory size per block on device 0.
   unsigned int const                              kStatsSmemSize;     ///< Size of shared memory per block in GetStatsKernel
   unsigned int                                    fHistoSmemSize;     ///< Size of shared memory per block in HistoKernel
   // clang-format on

public:
   RHnSYCL() = delete;

   RHnSYCL(std::size_t maxBulkSize, const std::array<int, Dim> &ncells, const std::array<double, Dim> &xlow,
           const std::array<double, Dim> &xhigh, const double **binEdges = NULL);

   ~RHnSYCL()
   {
      sycl::free(fDIntermediateStats, queue);
      if (fDBinEdges != NULL) {
         sycl::free(fDBinEdges, queue);
      }
   }

   RHnSYCL(const RHnSYCL &) = delete;
   RHnSYCL &operator=(const RHnSYCL &) = delete;

   int GetEntries() const { return fEntries; }

   void RetrieveResults(T *histResult, double *statsResult);

   void Fill(const RVecD &coords);

   void Fill(const RVecD &coords, const RVecD &weights);

   std::size_t GetMaxBulkSize() { return fMaxBulkSize; }

private:
   void GetStats(std::size_t size);

   void ExecuteSYCLHisto(std::size_t size);
};

} // namespace Experimental
} // namespace ROOT
#endif
