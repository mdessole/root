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

   T                                *fDHistogram;         ///< Pointer to histogram buffer on the GPU.
   int                               fNbins;              ///< Total number of bins in the histogram WITH under/overflow

   std::array<AxisDescriptor, Dim>   fHAxes;              ///< Vector of Dim axis descriptors
   AxisDescriptor                   *fDAxes;              ///< Pointer to axis descriptors on the GPU.

   std::vector<double>               fHCoords;            ///< 1D buffer with bufferSize #Dim-dimensional coordinates to fill in xxyyzz format.
   std::vector<double>               fHWeights;           ///< Buffer of weights for each bin on the Host.
   double                           *fDCoords;            ///< Pointer to array of coordinates to fill on the GPU.
   double                           *fDWeights;           ///< Pointer to array of weights on the GPU.
   int                              *fDBins;              ///< Pointer to array of bins (corresponding to the coordinates) to fill on the GPU.

   int                               fEntries;            ///< Number of entries that have been filled.
   double                           *fDIntermediateStats; ///< Buffer for storing intermediate results of stat reduction on GPU.
   double                           *fDStats;             ///< Pointer to statistics array on GPU.

   // Kernel size parameters
   unsigned int                      fNumBlocks;          ///< Number of blocks used in CUDA kernels
   unsigned int                      fMaxBulkSize;         ///< Number of coordinates to buffer.
   unsigned int                      fMaxSmemSize;        ///< Maximum shared memory size per block on device 0.
   unsigned int                      kStatsSmemSize;      ///< Size of shared memory per block in GetStatsKernel
   unsigned int                      fHistoSmemSize;      ///< Size of shared memory per block in HistoKernel
   // clang-format on

public:
   RHnCUDA() = delete;

   RHnCUDA(size_t maxBulkSize, std::array<int, Dim> ncells, std::array<double, Dim> xlow, std::array<double, Dim> xhigh,
           const double **binEdges = NULL);

   ~RHnCUDA();

   RHnCUDA(const RHnCUDA &) = delete;
   RHnCUDA &operator=(const RHnCUDA &) = delete;

   int GetEntries() { return fEntries; }

   void AllocateBuffers();

   void RetrieveResults(T *histResult, double *statsResult);

   void Fill(const RVecD &coords);

   void Fill(const RVecD &coords, const RVecD &weights);

   size_t GetMaxBulkSize() { return fMaxBulkSize; }

protected:
   void GetStats(unsigned int size);

   void ExecuteCUDAHisto(unsigned int size);
};

} // namespace Experimental
} // namespace ROOT
#endif
