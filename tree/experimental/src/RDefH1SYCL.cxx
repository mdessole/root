#include "RDefH1SYCL.h"
#include <sycl/sycl.hpp>
#include <iostream>
#include <array>

#include "TMath.h"
#include "SYCLHelpers.h"
#include "ROOT/RVec.hxx"

namespace ROOT {
namespace Experimental {

using mode = sycl::access::mode;

template <class T>
using AccLM = sycl::local_accessor<T, 1>;

////////////////////////////////////////////////////////////////////////////////
/// Bin calculation methods

inline int FindFixBin(double x, const double *binEdges, int binEdgesIdx, int nBins, double xMin, double xMax)
{
   int bin;

   // OPTIMIZATION: can this be done with less branching?
   if (x < xMin) { // underflow
      bin = 0;
   } else if (!(x < xMax)) { // overflow  (note the way to catch NaN)
      bin = nBins + 1;
   } else {
      if (binEdgesIdx < 0) { // fix bins
         bin = 1 + int(nBins * (x - xMin) / (xMax - xMin));
      } else { // variable bin sizes
         bin = 1 + SYCLHelpers::BinarySearch(nBins + 1, &binEdges[binEdgesIdx], x);
      }
   }

   return bin;
}

template <op Op, unsigned int fDim>
inline int GetBin(size_t tid, double *binEdges, int *binEdgesIdx, int *nBinsAxis, double *xMin, double *xMax,
                  double *buffer, double *parameters, double *coords, size_t bulkSize, int *bins)
{
   auto bin = 0;
   for (int d = fDim - 1; d >= 0; d--) {
      coords[tid] = Op(&buffer[d * bulkSize],parameters,tid,bulkSize); // Write result for computing statistics, otherwise could be avoided
      auto binD = FindFixBin(coords[tid], binEdges, binEdgesIdx[d], nBinsAxis[d] - 2, xMin[d], xMax[d]);
      bins[d * bulkSize + tid] = binD;

      if (binD < 0) {
         return -1;
      }

      bin = bin * nBinsAxis[d] + binD;
   }

   return bin;
}

///////////////////////////////////////////
/// Methods for incrementing a bin.

template <sycl::access::address_space Space>
inline void AddBinContent(double *histogram, int bin, double weight)
{
   auto atomic =
      sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device, Space>(histogram[bin]);
   atomic.fetch_add(weight);
}

template <sycl::access::address_space Space>
inline void AddBinContent(float *histogram, int bin, double weight)
{
   auto atomic =
      sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, Space>(histogram[bin]);
   atomic.fetch_add((float)weight);
}

template <sycl::access::address_space Space>
inline void AddBinContent(short *histogram, int bin, double weight)
{
   // There is no fetch_add for short so we need to operate on integers... (Assumes little endian)
   short *addr = &histogram[bin];
   int *addrInt = (int *)((char *)addr - ((size_t)addr & 2));
   int assumed, newVal, overwrite;
   bool success = false;

   do {
      assumed = *addrInt;

      if ((size_t)addr & 2) {
         newVal = (assumed >> 16) + (int)weight; // extract short from upper 16 bits
         overwrite = assumed & 0x0000ffff;       // clear upper 16 bits
         if (newVal > -32768 && newVal < 32768)
            overwrite |= (newVal << 16); // Set upper 16 bits to newVal
         else if (newVal < -32767)
            overwrite |= 0x80010000; // Set upper 16 bits to min short (-32767)
         else
            overwrite |= 0x7fff0000; // Set upper 16 bits to max short (32767)
      } else {
         newVal = (((assumed & 0xffff) << 16) >> 16) + (int)weight; // extract short from lower 16 bits + sign extend
         overwrite = assumed & 0xffff0000;                          // clear lower 16 bits
         if (newVal > -32768 && newVal < 32768)
            overwrite |= (newVal & 0xffff); // Set lower 16 bits to newVal
         else if (newVal < -32767)
            overwrite |= 0x00008001; // Set lower 16 bits to min short (-32767)
         else
            overwrite |= 0x00007fff; // Set lower 16 bits to max short (32767)
      }

      auto atomic = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, Space>(addrInt[0]);
      success = atomic.compare_exchange_strong(assumed, overwrite);
   } while (!success);
}

template <sycl::access::address_space Space>
inline void AddBinContent(int *histogram, int bin, double weight)
{
   int assumed;
   long newVal;
   bool success = false;

   // Repeat on failure/when the bin was already updated by another thread
   do {
      assumed = histogram[bin];
      newVal = sycl::max(long(-INT_MAX), sycl::min(assumed + long(weight), long(INT_MAX)));
      auto atomic =
         sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, Space>(histogram[bin]);
      success = atomic.compare_exchange_strong(assumed, (int)newVal);
   } while (!success);
}

//
// TODO: Cleaner overloads for local accessors with less duplication
//

template <sycl::access::address_space Space>
inline void AddBinContent(AccLM<double> histogram, int bin, double weight)
{
   auto atomic =
      sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device, Space>(histogram[bin]);
   atomic.fetch_add(weight);
}

template <sycl::access::address_space Space>
inline void AddBinContent(AccLM<float> histogram, int bin, double weight)
{
   auto atomic =
      sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, Space>(histogram[bin]);
   atomic.fetch_add((float)weight);
}

template <sycl::access::address_space Space>
inline void AddBinContent(AccLM<short> histogram, int bin, double weight)
{
   // There is no fetch_add for short so we need to operate on integers... (Assumes little endian)
   short *addr = &histogram[bin];
   int *addrInt = (int *)((char *)addr - ((size_t)addr & 2));
   int assumed, newVal, overwrite;
   bool success = false;

   do {
      assumed = *addrInt;

      if ((size_t)addr & 2) {
         newVal = (assumed >> 16) + (int)weight; // extract short from upper 16 bits
         overwrite = assumed & 0x0000ffff;       // clear upper 16 bits
         if (newVal > -32768 && newVal < 32768)
            overwrite |= (newVal << 16); // Set upper 16 bits to newVal
         else if (newVal < -32767)
            overwrite |= 0x80010000; // Set upper 16 bits to min short (-32767)
         else
            overwrite |= 0x7fff0000; // Set upper 16 bits to max short (32767)
      } else {
         newVal = (((assumed & 0xffff) << 16) >> 16) + (int)weight; // extract short from lower 16 bits + sign extend
         overwrite = assumed & 0xffff0000;                          // clear lower 16 bits
         if (newVal > -32768 && newVal < 32768)
            overwrite |= (newVal & 0xffff); // Set lower 16 bits to newVal
         else if (newVal < -32767)
            overwrite |= 0x00008001; // Set lower 16 bits to min short (-32767)
         else
            overwrite |= 0x00007fff; // Set lower 16 bits to max short (32767)
      }

      auto atomic = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, Space>(addrInt[0]);
      success = atomic.compare_exchange_strong(assumed, overwrite);
   } while (!success);
}

template <sycl::access::address_space Space>
inline void AddBinContent(AccLM<int> histogram, int bin, double weight)
{
   int assumed;
   long newVal;
   bool success = false;

   // Repeat on failure/when the bin was already updated by another thread
   do {
      assumed = histogram[bin];
      newVal = sycl::max(long(-INT_MAX), sycl::min(assumed + long(weight), long(INT_MAX)));
      auto atomic =
         sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, Space>(histogram[bin]);
      success = atomic.compare_exchange_strong(assumed, (int)newVal);
   } while (!success);
}

///////////////////////////////////////////
/// Histogram filling kernels

template <typename T, op Op, unsigned int Dim>
class HistogramGlobal {
public:
   HistogramGlobal(T *_histogram, double *_binEdges, int *_binEdgesIdx, int *_nBinsAxis, double *_xMin, double *_xMax,
                   double *_buffer, double *_parameters, double *_coords, double *_weights, int *_bins, std::size_t _bulkSize)
      : histogram(_histogram),
        binEdges(_binEdges),
        xMin(_xMin),
        xMax(_xMax),
        buffer(_buffer),
        parameters(_parameters),
        coords(_coords),
        weights(_weights),
        binEdgesIdx(_binEdgesIdx),
        bins(_bins),
        nBinsAxis(_nBinsAxis),
        bulkSize(_bulkSize)
   {
   }

   void operator()(sycl::item<1> item) const
   {
      size_t id = item.get_linear_id();
      auto bin = GetBin<Op,Dim>(id, binEdges, binEdgesIdx, nBinsAxis, xMin, xMax, buffer, parameters, coords, bulkSize, bins);

      if (bin >= 0) {
         AddBinContent<sycl::access::address_space::global_space>(histogram, bin, weights[id]);
      }
   }

private:
   T *histogram;
   double *binEdges, *xMin, *xMax, *buffer, *parameters, *coords, *weights;
   int *binEdgesIdx, *bins, *nBinsAxis;
   std::size_t bulkSize;
};

template <typename T, op Op, unsigned int Dim>
class HistogramLocal {
public:
   HistogramLocal(AccLM<T> _localMem, T *_histogram, double *_binEdges, int *_binEdgesIdx, int *_nBinsAxis,
                  double *_xMin, double *_xMax, 
                  double *_buffer, double *_parameters, double *_coords, double *_weights, int *_bins, size_t _nBins,
                  std::size_t _bulkSize)
      : localMem(_localMem),
        histogram(_histogram),
        binEdges(_binEdges),
        xMin(_xMin),
        xMax(_xMax),
        buffer(_buffer),
        parameters(_parameters),
        coords(_coords),
        weights(_weights),
        binEdgesIdx(_binEdgesIdx),
        bins(_bins),
        nBinsAxis(_nBinsAxis),
        nBins(_nBins),
        bulkSize(_bulkSize)
   {
   }

   void operator()(sycl::nd_item<1> item) const
   {
      auto globalId = item.get_global_id(0);
      auto localId = item.get_local_id(0);
      auto group = item.get_group();
      auto groupSize = item.get_local_range(0);
      auto stride = groupSize * item.get_group_range(0);

      // Initialize a local per-work-group histogram
      for (auto i = localId; i < nBins; i += groupSize) {
         localMem[i] = 0;
      }
      sycl::group_barrier(group);

      for (auto i = globalId; i < bulkSize; i += stride) {
         // Fill local histogram
         auto bin = GetBin<Op, Dim>(i, binEdges, binEdgesIdx, nBinsAxis, xMin, xMax, buffer, parameters, coords, bulkSize, bins);

         if (bin >= 0) {
            AddBinContent<sycl::access::address_space::local_space>(localMem, bin, weights[i]);
         }
      }
      sycl::group_barrier(group);

      // Merge results in global histogram
      for (auto i = localId; i < nBins; i += groupSize) {
         AddBinContent<sycl::access::address_space::global_space>(histogram, i, localMem[i]);
      }
   }

private:
   AccLM<T> localMem;
   T *histogram;
   double *binEdges, *xMin, *xMax, *buffer, *parameters, *coords, *weights;
   int *binEdgesIdx, *bins, *nBinsAxis;
   std::size_t nBins, bulkSize;
};

///////////////////////////////////////////
/// Statistics calculation kernels

template <unsigned int fDim>
class ExcludeUOverflowKernel {
public:
   ExcludeUOverflowKernel(int *_bins, double *_weights, int *_nBinsAxis, std::size_t _bulkSize)
      : bins(_bins), weights(_weights), nBinsAxis(_nBinsAxis), bulkSize(_bulkSize)
   {
   }

   void operator()(sycl::id<1> id) const
   {
      if (bins[id] <= 0 || bins[id] >= nBinsAxis[id / bulkSize] - 1) {
         weights[id % bulkSize] = 0.;
      }
   }

private:
   int *bins;
   double *weights;
   int *nBinsAxis;
   std::size_t bulkSize;
};

class CombineStatsKernel {
public:
   CombineStatsKernel(double *_stats, double *_intermediate) : stats(_stats), intermediate(_intermediate) {}

   void operator()(sycl::id<1> id) const { stats[id] += intermediate[id]; }

private:
   double *stats;
   double *intermediate;
};

///////////////////////////////////////////
/// RDefH1SYCL

template <typename T, op Op, unsigned int nInput, unsigned int WGroupSize>
RDefH1SYCL<T, Op, nInput, WGroupSize>::RDefH1SYCL(std::size_t maxBulkSize, const std::size_t nBins,
                                     const std::array<int, fDim> &nBinsAxis, const std::array<double, fDim> &xLow,
                                     const std::array<double, fDim> &xHigh, const std::vector<double> &binEdges,
                                     const std::array<int, fDim> &binEdgesIdx, const std::vector<double> &parameters)
   : queue(sycl::default_selector_v),
     kStatsSmemSize((WGroupSize <= 32) ? 2 * WGroupSize * sizeof(double) : WGroupSize * sizeof(double))
{
   auto device = queue.get_device();

   fMaxBulkSize = maxBulkSize;
   fNBins = nBins;
   fEntries = 0;

   // Setup device memory for filling the histogram.
   fDWeights = sycl::malloc_device<double>(fMaxBulkSize, queue);
   fDBuffer = sycl::malloc_device<double>(nInput * fDim * fMaxBulkSize, queue);
   fDCoords = sycl::malloc_device<double>(fDim * fMaxBulkSize, queue);
   fDBins = sycl::malloc_device<int>(fDim * fMaxBulkSize, queue);

   // Setup device memory for Op parameters
   fDParameters = sycl::malloc_device<double>(parameters.size(), queue);
   queue.memcpy(fDParameters, parameters.data(), parameters.size() * sizeof(double));

   // Setup device memory for histogram characteristics
   fDNBinsAxis = sycl::malloc_device<int>(fDim, queue);
   queue.memcpy(fDNBinsAxis, nBinsAxis.data(), fDim * sizeof(int));
   fDMin = sycl::malloc_device<double>(fDim, queue);
   queue.memcpy(fDMin, xLow.data(), fDim * sizeof(double));
   fDMax = sycl::malloc_device<double>(fDim, queue);
   queue.memcpy(fDMax, xHigh.data(), fDim * sizeof(double));
   fDBinEdgesIdx = sycl::malloc_device<int>(fDim, queue);
   queue.memcpy(fDBinEdgesIdx, binEdgesIdx.data(), fDim * sizeof(int));

   fDBinEdges = NULL;
   if (binEdges.size() > 0) {
      fDBinEdges = sycl::malloc_device<double>(binEdges.size(), queue);
      queue.memcpy(fDBinEdges, binEdges.data(), binEdges.size() * sizeof(double));
   }

   // Allocate and initialize device memory for the histogram and statistics.
   fDHistogram = sycl::malloc_device<T>(fNBins, queue);
   queue.memset(fDHistogram, 0, fNBins * sizeof(T));
   fDStats = sycl::malloc_device<double>(kNStats, queue);
   queue.memset(fDStats, 0, kNStats * sizeof(double));

#ifdef ROOT_RDF_ADAPTIVECPP
   fDIntermediateStats = sycl::malloc_device<double>(kNStats, queue);
   queue.memset(fDIntermediateStats, 0, kNStats * sizeof(double));
#else
   fDIntermediateStats = NULL;
#endif

   queue.wait();

   // Determine the amount of shared memory required for HistogramKernel, and the maximum available.
   fHistoSmemSize = fNBins * sizeof(T);
   auto has_local_mem = device.is_gpu() || (device.template get_info<sycl::info::device::local_mem_type>() !=
                                            sycl::info::local_mem_type::none);
   fMaxSmemSize = has_local_mem ? device.template get_info<sycl::info::device::local_mem_size>() : 0;

   if (getenv("DBG")) {
      std::cout << "Running SYCLHist on " << device.template get_info<sycl::info::device::name>() << "\n";
      printf("USM support: %s\n", device.has(sycl::aspect::usm_device_allocations) ? "yes" : "no");
      printf("Maximum shared memory size: %zu\n", fMaxSmemSize);
   }
}

template <typename T, op Op, unsigned int nInput, unsigned int WGroupSize>
void RDefH1SYCL<T, Op, nInput, WGroupSize>::Fill(const RVecD &buffer)
{
   RVecD weights(buffer.size() / (nInput * fDim), 1);
   Fill(buffer, weights);
}

template <typename T, op Op, unsigned int nInput, unsigned int WGroupSize>
void RDefH1SYCL<T, Op, nInput, WGroupSize>::Fill(const RVecD &buffer, const RVecD &weights)
{
   auto bulkSize = weights.size();

   // Add the coordinates and weight to the buffers
   std::vector<sycl::event> copyEvents(2);
   copyEvents[0] = queue.memcpy(fDBuffer, buffer.begin(), bulkSize * nInput * fDim * sizeof(double), prevBulk);
   copyEvents[1] = queue.memcpy(fDWeights, weights.begin(), bulkSize * sizeof(double), prevBulk);

   fEntries += bulkSize;

   // The histogram kernels execute asynchronously.
   ExecuteSYCLHisto(bulkSize, copyEvents);
}

template <typename T, op Op, unsigned int nInput, unsigned int WGroupSize>
void RDefH1SYCL<T, Op, nInput, WGroupSize>::GetStats(std::size_t size, sycl::event &fillEvent)
{
   // Set weights of over/underflow bins to zero. Done in separate kernel in case we want to add the option to add
   // under/overflow bins to the statistics.
   auto e = queue.submit([&](sycl::handler &cgh) {
      cgh.depends_on(fillEvent);
      cgh.parallel_for(sycl::range<1>(size * fDim), ExcludeUOverflowKernel<fDim>(fDBins, fDWeights, fDNBinsAxis, size));
   });

   std::vector<sycl::event> statsReductions;
   std::size_t reductionRange = ceil(size / 8.); // each thread/work-item reduces 8 values
   auto resultBuf = fDStats;
#ifdef ROOT_RDF_ADAPTIVECPP
   resultBuf = fDIntermediateStats;
#endif

   statsReductions.push_back(queue.submit([&](sycl::handler &cgh) {
      cgh.depends_on(e);
      auto weightsPtr = fDWeights;
      auto GetSumW = sycl::reduction(&resultBuf[0], sycl::plus<double>());
      auto GetSumW2 = sycl::reduction(&resultBuf[1], sycl::plus<double>());

      cgh.parallel_for(sycl::range<1>(reductionRange), GetSumW, GetSumW2, [=](sycl::id<1> id, auto &sumw, auto &sumw2) {
         for (unsigned int gid = id; gid < size; gid += reductionRange) {
            sumw += weightsPtr[gid];
            sumw2 += weightsPtr[gid] * weightsPtr[gid];
         }
      });
   }));

   auto offset = 2;
   for (auto d = 0U; d < fDim; d++) {
      statsReductions.push_back(queue.submit([&](sycl::handler &cgh) {
         cgh.depends_on(e);
         auto coordsPtr = fDCoords;
         auto weightsPtr = fDWeights;

         // Multiply weight with coordinate of current axis. E.g., for fDim = 2 this computes Tsumwx and Tsumwy
         auto GetSumWAxis = sycl::reduction(&resultBuf[offset++], sycl::plus<double>());
         auto GetSumWAxis2 = sycl::reduction(&resultBuf[offset++], sycl::plus<double>());

         cgh.parallel_for(sycl::range<1>(reductionRange), GetSumWAxis, GetSumWAxis2,
                          [=](sycl::id<1> id, auto &sumwaxis, auto &sumwaxis2) {
                             for (unsigned int gid = id; gid < size; gid += reductionRange) {
                                sumwaxis += weightsPtr[gid] * coordsPtr[d * size + gid];
                                sumwaxis2 += weightsPtr[gid] * coordsPtr[d * size + gid] * coordsPtr[d * size + gid];
                             }
                          });
      }));

      for (auto prev_d = 0U; prev_d < d; prev_d++) {
         statsReductions.push_back(queue.submit([&](sycl::handler &cgh) {
            cgh.depends_on(e);
            auto coordsPtr = fDCoords;
            auto weightsPtr = fDWeights;

            // Multiplies coordinate of current axis with the "previous" axis. E.g., for fDim = 2 this computes
            // Tsumwxy
            auto GetSumWAxisAxis = sycl::reduction(&resultBuf[offset++], sycl::plus<double>());

            cgh.parallel_for(sycl::range<1>(reductionRange), GetSumWAxisAxis, [=](sycl::id<1> id, auto &sumwaxisaxis) {
               for (unsigned int gid = id; gid < size; gid += reductionRange) {
                  sumwaxisaxis += weightsPtr[gid] * coordsPtr[d * size + gid] * coordsPtr[prev_d * size + gid];
               }
            });
         }));
      }
   }

#ifdef ROOT_RDF_ADAPTIVECPP
   // The AdaptiveCpp reduction interface overwrites the output array instead of combining the original value,
   // so we have to add the values to the previously reduced values.
   auto combineEvent = queue.submit([&](sycl::handler &cgh) {
      // Explicit dependency required because dependencies are only defined implicitly when creating accessors,
      // but we don't create an accessor on device pointers.
      cgh.depends_on(statsReductions);
      cgh.parallel_for(sycl::range<1>(kNStats), CombineStatsKernel(fDStats, resultBuf));
   });
   prevBulk = {combineEvent};
#else
   prevBulk = statsReductions;
#endif
}

template <typename T, op Op, unsigned int nInput,  unsigned int WGroupSize>
void RDefH1SYCL<T, Op, nInput, WGroupSize>::ExecuteSYCLHisto(std::size_t size, std::vector<sycl::event> &depends)
{
   // The SYCL specification does not require eager execution, so we need to wait for the copy events to have completed
   // before filling to avoid the overwriting the input values in the host buffer
   for (auto &e : depends) {
      e.wait();
   }

   sycl::event fillEvent;
   if (fHistoSmemSize > fMaxSmemSize) {
      fillEvent = queue.submit([&](sycl::handler &cgh) {
         // Partitions the vector pairs over available threads and computes the invariant masses.
         cgh.depends_on(depends);
         cgh.parallel_for(sycl::range<1>(size),
                          HistogramGlobal<T, Op, fDim>(fDHistogram, fDBinEdges, fDBinEdgesIdx, fDNBinsAxis, fDMin, fDMax,
                                                  fDBuffer, fDParameters, fDCoords, fDWeights, fDBins, size));
      });
   } else {
      fillEvent = queue.submit([&](sycl::handler &cgh) {
         // Similar to CUDA shared memory.
         sycl::local_accessor<T, 1> localMem(sycl::range<1>(fNBins), cgh);

         // Global range must be a multiple of local range (WGroupSize) that is equal or larger than local range.
         auto execution_range = sycl::nd_range<1>{sycl::range<1>{((size + WGroupSize - 1) / WGroupSize) * WGroupSize},
                                                  sycl::range<1>{WGroupSize}};

         cgh.depends_on(depends);
         cgh.parallel_for(execution_range,
                          HistogramLocal<T, Op, fDim>(localMem, fDHistogram, fDBinEdges, fDBinEdgesIdx, fDNBinsAxis, fDMin,
                                                 fDMax, fDBuffer, fDParameters, fDCoords, fDWeights, fDBins, fNBins, size));
      });
   } // end of scope, ensures data copied back to host

   GetStats(size, fillEvent);
}

template <typename T, op Op, unsigned int nInput,  unsigned int WGroupSize>
void RDefH1SYCL<T, Op, nInput, WGroupSize>::RetrieveResults(T *histResult, double *statsResult)
{
   queue.wait();
   queue.memcpy(histResult, fDHistogram, fNBins * sizeof(T));
   queue.memcpy(statsResult, fDStats, kNStats * sizeof(double));
   queue.wait();
}

#include "RDefH1SYCL-impl.cxx"

} // namespace Experimental
} // namespace ROOT
