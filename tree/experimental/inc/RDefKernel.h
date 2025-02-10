#ifndef RDefKernel_H
#define RDefKernel_H

#include "Math/GenVector/AccHeaders.h"
#include "Math/Vector4D.h"

namespace ROOT {
namespace Experimental {

extern SYCL_EXTERNAL double IdentityKernel(double *buffer, double *parameters, std::size_t idx, std::size_t bulksize);

// Invariant masses of the sum of two sequences of particles 
// p0i = (pt0i, eta0i, phi0i, m0i), p1i = (pt1i, eta1i, phi1i, m1i), i=0,...,bulksize-1=n
// Underlying memory is 
// pt00,pt01,...,p0n,eta00,eta01,...,eta0n,ph00,phi01,...,ph0n,m00,m01,...,m0n,
// pt10,pt11,...,p1n,eta10,eta11,...,eta1n,ph10,phi11,...,ph1n,m10,m11,...,m1n
// Efficient when we have 8 vectors on the host side
extern SYCL_EXTERNAL double InvariantMassesKernel(double *buffer, double *parameters, std::size_t idx, std::size_t bulksize);

// Invariant masses of a sequences of particles 
// p0i = (pt0i, eta0i, phi0i, m0i), p1i = (pt1i, eta1i, phi1i, m1i), i=0,...,bulksize-1=n
// Underlying memory is 
// pt00,pt01,...,p0n,eta00,eta01,...,eta0n,ph00,phi01,...,ph0n,m00,m01,...,m0n
// Efficient when we have 4 vectors on the host side
extern SYCL_EXTERNAL double InvariantMassKernel(double *buffer, double *parameters, std::size_t idx, std::size_t bulksize);


// Invariant masses of the sum of two sequences of particles 
// p0i = (pt0i, eta0i, phi0i, m0i), p1i = (pt1i, eta1i, phi1i, m1i), i=0,...,bulksize-1=n
// Underlying memory is 
// pt00,pt10,eta00,eta10,phi00,phi10,m00,m10,pt01,pt11,eta01,eta11,phi01,phi10,m01,m11,...,
// pt0n,pt1n,eta0n,eta1n,phi0n,phi1n,m0n,m1n
// Unefficient, but often used after flattening a structure of jagged arrays of particles
extern SYCL_EXTERNAL double InvariantMassesTransKernel(double *buffer, double *parameters, std::size_t idx, std::size_t bulksize);

}
}

#endif