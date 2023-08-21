#ifndef INVARIANT_MASS_SYCL
#define INVARIANT_MASS_SYCL

#include "Math/Vector4D.h"

namespace ROOT {
namespace Experimental {

double *InvariantMassSYCL(const ROOT::Math::PtEtaPhiEVector *v1, const ROOT::Math::PtEtaPhiEVector *v2, size_t size);

} // namespace Experimental
} // namespace ROOT
#endif
