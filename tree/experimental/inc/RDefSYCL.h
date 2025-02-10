#ifndef RDefSYCL_H
#define RDefSYCL_H

#include <array>
#include <optional>
#include <sycl/sycl.hpp>
#include "ROOT/RVec.hxx"

#include "RDefKernel.h"

typedef double (op)(double*, double*, std::size_t, std::size_t);

namespace ROOT {
namespace Experimental {

template <typename T, op Op, unsigned int nInput, unsigned int WGroupSize = 256>
class RDefSYCL {
private:

   static constexpr unsigned int fnInput = nInput;
   
   // clang-format off
   sycl::queue                       queue;
   std::vector<sycl::event>          prevBulk;

   double                           *fDParameters;        ///< Pointer to array of parameters for computing fDCoords[idx]=Op(fDBuffer,fParameters,idx) on the GPU.
   double                           *fDCoords;            ///< Pointer to array of coordinates on the GPU.
   double                           *fDBuffer;            ///< Pointer to array of inputs for computing coordinates to be filled on the GPU.
 
   std::size_t                       fMaxBulkSize;        ///< Number of coordinates to buffer.
  
public:
   RDefSYCL() = delete;

   RDefSYCL(std::size_t maxBulkSize, 
           const std::vector<double> &parameters);

   ~RDefSYCL()
   {
      sycl::free(fDBuffer, queue);
      sycl::free(fDParameters, queue);
      sycl::free(fDCoords, queue);
   }

   RDefSYCL(const RDefSYCL &) = delete;
   RDefSYCL &operator=(const RDefSYCL &) = delete;

   unsigned int GetnInput() const { return fnInput; }

   void RetrieveResults(T *execResult, size_t size);

   void EvalBulkExpr(const RVecD &buffer);

   void Fill(const RVecD &buffer, const RVecD &weights);

   std::size_t GetMaxBulkSize() { return fMaxBulkSize; }

   private:

   void ExecuteSYCLEval(std::size_t size, std::vector<sycl::event> &depends);
};

} // namespace Experimental
} // namespace ROOT
#endif