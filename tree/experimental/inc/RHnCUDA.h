#ifndef RHnCUDA_H
#define RHnCUDA_H

#include <vector>
#include <array>
#include "AxisDescriptor.h"
#include "ROOT/RVec.hxx"

namespace ROOT {
namespace Experimental {

template <typename T, unsigned int Dim, unsigned int BlockSize = 256>
class RHnCUDA {
   // clang-format off
private:
   static constexpr int kNStats = 2 + Dim * 2 + Dim * (Dim - 1) / 2; ///< Number of statistics.
   // static constexpr std::size_t kCPUFinalThreshold = ;  ///< Threshold for summing remainder on CPU

   T                                *fDHistogram;         ///< Pointer to histogram buffer on the GPU.
   int                               fNBins;              ///< Total number of bins in the histogram WITH under/overflow

   int                              *fDNBinsAxis;         ///< Number of bins(1D) WITH u/overflow per axis
   double                           *fDMin;               ///< Low edge of first bin per axis
   double                           *fDMax;               ///< Upper edge of last bin per axis
   double                           *fDBinEdges;          ///< Bin edges array for each axis
   int                              *fDBinEdgesIdx;       ///< Start index of the binedges in kBinEdges per axis

   double                           *fDCoords;            ///< Pointer to array of coordinates to fill on the GPU.
   double                           *fDWeights;           ///< Pointer to array of weights on the GPU.
   bool                             *fDMask;              ///< Pointer to array of bins (corresponding to the coordinates) to fill on the GPU.

   int                               fEntries;            ///< Number of entries that have been filled.
   double                           *fDIntermediateStats; ///< Buffer for storing intermediate results of stat reduction on GPU.
   double                           *fDStats;             ///< Pointer to statistics array on GPU.

   // Kernel size parameters
   unsigned int                      fNumBlocks;          ///< Number of blocks used in CUDA kernels
   std::size_t                       fMaxBulkSize;        ///< Number of coordinates to buffer.
   std::size_t                       kStatsSmemSize;      ///< Size of shared memory per block in GetStatsKernel
   std::size_t                       fHistoSmemSize;      ///< Size of shared memory per block in HistoKernel

   struct CUDAProps;
   std::unique_ptr<CUDAProps>        fProps;
   // clang-format on

public:
   RHnCUDA() = delete;

   // TODO: Change RHnCUDA to SOA for axes
   RHnCUDA(std::size_t maxBulkSize, const std::size_t nBins, const std::array<int, Dim> &nBinsAxis,
           const std::array<double, Dim> &xLow, const std::array<double, Dim> &xHigh,
           const std::vector<double> &binEdges, const std::array<int, Dim> &binEdgesIdx);

   ~RHnCUDA();

   RHnCUDA(const RHnCUDA &) = delete;
   RHnCUDA &operator=(const RHnCUDA &) = delete;

   int GetEntries() { return fEntries; }

   void RetrieveResults(T *histResult, double *statsResult);

   void Fill(const RVecD &coords);

   void Fill(const RVecD &coords, const RVecD &weights);

   size_t GetMaxBulkSize() { return fMaxBulkSize; }

protected:

   void GetNumBlocksAndThreads(int n, int &blocks, int &threads);

   void GetStats(std::size_t size);

   void ExecuteCUDAHisto(std::size_t size);
};

} // namespace Experimental
} // namespace ROOT
#endif
