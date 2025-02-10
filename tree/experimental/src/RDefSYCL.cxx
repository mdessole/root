#include "RDefSYCL.h"
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


///////////////////////////////////////////
/// Eval kernel

template <typename T, op Op, unsigned int Dim>
class EvalKernel {
public:
   EvalKernel(double *_buffer, double *_parameters, double *_coords, std::size_t _bulkSize)
      : buffer(_buffer),
        parameters(_parameters),
        coords(_coords),
        bulkSize(_bulkSize)
   {
   }

   void operator()(sycl::item<1> item) const
   {
      size_t id = item.get_linear_id();
      coords[id] = Op(buffer,parameters,id,bulkSize); 
   }

private:
   double *buffer, *parameters, *coords;
   std::size_t bulkSize;
};


///////////////////////////////////////////
/// RDefSYCL

template <typename T, op Op, unsigned int nInput, unsigned int WGroupSize>
RDefSYCL<T, Op, nInput, WGroupSize>::RDefSYCL(std::size_t maxBulkSize, const std::vector<double> &parameters)
   : queue(sycl::default_selector_v)
{
   auto device = queue.get_device();

   fMaxBulkSize = maxBulkSize;

   // Setup device memory for filling the histogram.
   fDBuffer = sycl::malloc_device<double>(nInput * fMaxBulkSize, queue);
   fDCoords = sycl::malloc_device<double>(fMaxBulkSize, queue);

   // Setup device memory for Op parameters
   fDParameters = sycl::malloc_device<double>(parameters.size(), queue);
   queue.memcpy(fDParameters, parameters.data(), parameters.size() * sizeof(double));

   queue.wait();

   if (getenv("DBG")) {
      std::cout << "Running SYCLDefHist on " << device.template get_info<sycl::info::device::name>() << "\n";
      printf("USM support: %s\n", device.has(sycl::aspect::usm_device_allocations) ? "yes" : "no");
   }
}


template <typename T, op Op, unsigned int nInput,  unsigned int WGroupSize>
void RDefSYCL<T, Op, nInput, WGroupSize>::ExecuteSYCLEval(std::size_t size, std::vector<sycl::event> &depends)
{
   // The SYCL specification does not require eager execution, so we need to wait for the copy events to have completed
   // before filling to avoid the overwriting the input values in the host buffer
   for (auto &e : depends) {
      e.wait();
   }

   sycl::event fillEvent;
   {
      fillEvent = queue.submit([&](sycl::handler &cgh) {
         // Partitions the vector pairs over available threads and computes the invariant masses.
         cgh.depends_on(depends);
         cgh.parallel_for(sycl::range<1>(size),
                          EvalKernel<T, Op, 1>(fDBuffer, fDParameters, fDCoords, size));
      });
   } // end of scope, ensures data copied back to host

}

template <typename T, op Op, unsigned int nInput, unsigned int WGroupSize>
void RDefSYCL<T, Op, nInput, WGroupSize>::EvalBulkExpr(const RVecD &buffer)
{
   size_t bulkSize = buffer.size() / nInput;

   // Add the coordinates and weight to the buffers
   std::vector<sycl::event> copyEvents(1);
   copyEvents[0] = queue.memcpy(fDBuffer, buffer.begin(), bulkSize * nInput * sizeof(double), prevBulk);
   //copyEvents[1] = queue.memcpy(fDWeights, weights.begin(), bulkSize * sizeof(double), prevBulk);


   // The histogram kernels execute asynchronously.
   ExecuteSYCLEval(bulkSize, copyEvents);
}


template <typename T, op Op, unsigned int nInput,  unsigned int WGroupSize>
void RDefSYCL<T, Op, nInput, WGroupSize>::RetrieveResults(T *execResult, size_t size)
{
   queue.wait();
   queue.memcpy(execResult, fDCoords, size * sizeof(T));
   queue.wait();
}


#include "RDefSYCL-impl.cxx"

} // namespace Experimental
} // namespace ROOT
