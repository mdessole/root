/// Template instantations for RDefH1SYCL.
///
/// The implementation of the template class cannot be added to the header file RDefH1SYCL.h because this
/// header is included in cpp files to access the SYCL histogramming class. The cpp files are compiled
/// with a cpp compiler which will not be recognize any SYCL calls, so any calls to the SYCL API needs
/// to be separated and only compiled by the nvcc compiler.
/// TODO: support for char and short histograms.

/// @brief Return the invariant mass of two particles given their
/// transverse momentum (pt), rapidity (eta), azimuth (phi) and mass.

double IdentityKernel(double *buffer, double *parameters, std::size_t idx, std::size_t bulksize)
{
    return buffer[idx];
}

// clang-format off
// template class RDefH1SYCL<char,   IdentityKernel, 1,  64>;
template class RDefH1SYCL<short,  IdentityKernel, 1,  64>;
template class RDefH1SYCL<int,    IdentityKernel, 1,  64>;
template class RDefH1SYCL<float,  IdentityKernel, 1,  64>;
template class RDefH1SYCL<double, IdentityKernel, 1,  64>;

// template class RDefH1SYCL<char,   IdentityKernel, 1,  128>;
template class RDefH1SYCL<short,  IdentityKernel, 1,  128>;
template class RDefH1SYCL<int,    IdentityKernel, 1,  128>;
template class RDefH1SYCL<float,  IdentityKernel, 1,  128>;
template class RDefH1SYCL<double, IdentityKernel, 1,  128>;

// template class RDefH1SYCL<char,   IdentityKernel, 1,  256>;
template class RDefH1SYCL<short,  IdentityKernel, 1,  256>;
template class RDefH1SYCL<int,    IdentityKernel, 1,  256>;
template class RDefH1SYCL<float,  IdentityKernel, 1,  256>;
template class RDefH1SYCL<double, IdentityKernel, 1,  256>;

// template class RDefH1SYCL<char,   IdentityKernel, 1,  512>;
template class RDefH1SYCL<short,  IdentityKernel, 1,  512>;
template class RDefH1SYCL<int,    IdentityKernel, 1,  512>;
template class RDefH1SYCL<float,  IdentityKernel, 1,  512>;
template class RDefH1SYCL<double, IdentityKernel, 1,  512>;

// template class RDefH1SYCL<char,   IdentityKernel, 1,  1024>;
template class RDefH1SYCL<short,  IdentityKernel, 1,  1024>;
template class RDefH1SYCL<int,    IdentityKernel, 1,  1024>;
template class RDefH1SYCL<float,  IdentityKernel, 1,  1024>;
template class RDefH1SYCL<double, IdentityKernel, 1,  1024>;
// clang-format on
