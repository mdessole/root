#ifndef INVARIANT_MASS_CUDA
#define INVARIANT_MASS_CUDA

#include "Math/Vector4D.h"
#include <Math/PtEtaPhiE4D.h>

namespace ROOT {
namespace Experimental {

template <unsigned int BlockSize = 256>
class InvariantMassCUDA {
public:
   static double *
   ComputeInvariantMasses(const ROOT::Math::PtEtaPhiEVector *v1, const ROOT::Math::PtEtaPhiEVector *v2, size_t size);
};

} // namespace Experimental
} // namespace ROOT
#endif
