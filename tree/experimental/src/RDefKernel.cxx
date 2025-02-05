#include "RDefKernel.h"

#include "Math/GenVector/AccHeaders.h"
#include "Math/Vector4D.h"


namespace ROOT {
namespace Experimental {

double IdentityKernel(double *buffer, double *parameters, std::size_t idx, std::size_t bulksize)
{
    return buffer[idx];
}

// Invariant masses of the sum of two sequences of particles 
// p0i = (pt0i, eta0i, phi0i, m0i), p1i = (pt1i, eta1i, phi1i, m1i), i=0,...,bulksize-1=n
// Underlying memory is 
// pt00,pt01,...,p0n,eta00,eta01,...,eta0n,ph00,phi01,...,ph0n,m00,m01,...,m0n,
// pt10,pt11,...,p1n,eta10,eta11,...,eta1n,ph10,phi11,...,ph1n,m10,m11,...,m1n
// Efficient when we have 8 vectors on the host side
double InvariantMassesKernel(double *buffer, double *parameters, std::size_t idx, std::size_t bulksize)
{
    ROOT::ROOT_MATH_ARCH::PtEtaPhiMVector p1(buffer[0*bulksize+idx], buffer[1*bulksize+idx], buffer[2*bulksize+idx], buffer[3*bulksize+idx]);
    ROOT::ROOT_MATH_ARCH::PtEtaPhiMVector p2(buffer[4*bulksize+idx], buffer[5*bulksize+idx], buffer[6*bulksize+idx], buffer[7*bulksize+idx]);
    return (p1+p2).M();
}

// Invariant masses of the sum of two sequences of particles 
// p0i = (pt0i, eta0i, phi0i, m0i), p1i = (pt1i, eta1i, phi1i, m1i), i=0,...,bulksize-1=n
// Underlying memory is 
// pt00,pt10,eta00,eta10,phi00,phi10,m00,m10,pt01,pt11,eta01,eta11,phi01,phi10,m01,m11,...,
// pt0n,pt1n,eta0n,eta1n,phi0n,phi1n,m0n,m1n
// Unefficient, but often used after flattening a structure of jagged arrays of particles
// double InvariantMassesTransKernel(double *buffer, double *parameters, std::size_t idx, std::size_t bulksize)
// {
//     const std::size_t offset = idx * 8;
//     ROOT::ROOT_MATH_ARCH::PtEtaPhiMVector p1(buffer[offset+0], buffer[offset+2], buffer[offset+4], buffer[offset+6]);
//     ROOT::ROOT_MATH_ARCH::PtEtaPhiMVector p2(buffer[offset+1], buffer[offset+3], buffer[offset+5], buffer[offset+7]);
//     return (p1+p2).M();

// }

double InvariantMassesTransKernel(double *buffer, double *parameters, std::size_t idx, std::size_t bulksize)
{
    const std::size_t offset = bulksize*2;
    const std::size_t idx0 = idx*2; //first particle
    const std::size_t idx1 = idx0+1; //second particle
    // ROOT::ROOT_MATH_ARCH::PtEtaPhiMVector p1(buffer[offset+0], buffer[offset+2], buffer[offset+4], buffer[offset+6]);
    // ROOT::ROOT_MATH_ARCH::PtEtaPhiMVector p2(buffer[offset+1], buffer[offset+3], buffer[offset+5], buffer[offset+7]);
    ROOT::ROOT_MATH_ARCH::PtEtaPhiMVector p1(buffer[idx0+0*offset], buffer[idx0+1*offset], buffer[idx0+2*offset], buffer[idx0+3*offset]);
    ROOT::ROOT_MATH_ARCH::PtEtaPhiMVector p2(buffer[idx1+0*offset], buffer[idx1+1*offset], buffer[idx1+2*offset], buffer[idx1+3*offset]);
    return (p1+p2).M();
    
}

}
}
