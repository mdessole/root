/// Template instantations for RDefSYCL.
///
/// The implementation of the template class cannot be added to the header file RDefSYCL.h because this
/// header is included in cpp files to access the SYCL histogramming class. The cpp files are compiled
/// with a cpp compiler which will not be recognize any SYCL calls, so any calls to the SYCL API needs
/// to be separated and only compiled by the nvcc compiler.
/// TODO: support for char and short histograms.

/// @brief Return the invariant mass of two particles given their
/// transverse momentum (pt), rapidity (eta), azimuth (phi) and mass.


// clang-format off
// template class RDefSYCL<char,   IdentityKernel, 1,  64>;
template class RDefSYCL<short,  IdentityKernel, 1,  64>;
template class RDefSYCL<int,    IdentityKernel, 1,  64>;
template class RDefSYCL<float,  IdentityKernel, 1,  64>;
template class RDefSYCL<double, IdentityKernel, 1,  64>;

// template class RDefSYCL<char,   IdentityKernel, 1,  128>;
template class RDefSYCL<short,  IdentityKernel, 1,  128>;
template class RDefSYCL<int,    IdentityKernel, 1,  128>;
template class RDefSYCL<float,  IdentityKernel, 1,  128>;
template class RDefSYCL<double, IdentityKernel, 1,  128>;

// template class RDefSYCL<char,   IdentityKernel, 1,  256>;
template class RDefSYCL<short,  IdentityKernel, 1,  256>;
template class RDefSYCL<int,    IdentityKernel, 1,  256>;
template class RDefSYCL<float,  IdentityKernel, 1,  256>;
template class RDefSYCL<double, IdentityKernel, 1,  256>;

// template class RDefSYCL<char,   IdentityKernel, 1,  512>;
template class RDefSYCL<short,  IdentityKernel, 1,  512>;
template class RDefSYCL<int,    IdentityKernel, 1,  512>;
template class RDefSYCL<float,  IdentityKernel, 1,  512>;
template class RDefSYCL<double, IdentityKernel, 1,  512>;

// template class RDefSYCL<char,   IdentityKernel, 1,  1024>;
template class RDefSYCL<short,  IdentityKernel, 1,  1024>;
template class RDefSYCL<int,    IdentityKernel, 1,  1024>;
template class RDefSYCL<float,  IdentityKernel, 1,  1024>;
template class RDefSYCL<double, IdentityKernel, 1,  1024>;
// clang-format on


// clang-format off
// template class RDefSYCL<char,   InvariantMassesKernel, 8,  64>;
template class RDefSYCL<short,  InvariantMassesKernel, 8,  64>;
template class RDefSYCL<int,    InvariantMassesKernel, 8,  64>;
template class RDefSYCL<float,  InvariantMassesKernel, 8,  64>;
template class RDefSYCL<double, InvariantMassesKernel, 8,  64>;

// template class RDefSYCL<char,   InvariantMassesKernel, 8,  128>;
template class RDefSYCL<short,  InvariantMassesKernel, 8,  128>;
template class RDefSYCL<int,    InvariantMassesKernel, 8,  128>;
template class RDefSYCL<float,  InvariantMassesKernel, 8,  128>;
template class RDefSYCL<double, InvariantMassesKernel, 8,  128>;

// template class RDefSYCL<char,   InvariantMassesKernel, 8,  256>;
template class RDefSYCL<short,  InvariantMassesKernel, 8,  256>;
template class RDefSYCL<int,    InvariantMassesKernel, 8,  256>;
template class RDefSYCL<float,  InvariantMassesKernel, 8,  256>;
template class RDefSYCL<double, InvariantMassesKernel, 8,  256>;

// template class RDefSYCL<char,   InvariantMassesKernel, 8,  512>;
template class RDefSYCL<short,  InvariantMassesKernel, 8,  512>;
template class RDefSYCL<int,    InvariantMassesKernel, 8,  512>;
template class RDefSYCL<float,  InvariantMassesKernel, 8,  512>;
template class RDefSYCL<double, InvariantMassesKernel, 8,  512>;

// template class RDefSYCL<char,   InvariantMassesKernel, 8,  1024>;
template class RDefSYCL<short,  InvariantMassesKernel, 8,  1024>;
template class RDefSYCL<int,    InvariantMassesKernel, 8,  1024>;
template class RDefSYCL<float,  InvariantMassesKernel, 8,  1024>;
template class RDefSYCL<double, InvariantMassesKernel, 8,  1024>;
// clang-format on


// clang-format off
// template class RDefSYCL<char,   InvariantMassKernel, 8,  64>;
template class RDefSYCL<float,  InvariantMassKernel, 4,  64>;
template class RDefSYCL<double, InvariantMassKernel, 4,  64>;

// template class RDefSYCL<char,   InvariantMassKernel, 8,  128>;
template class RDefSYCL<float,  InvariantMassKernel, 4,  128>;
template class RDefSYCL<double, InvariantMassKernel, 4,  128>;

// template class RDefSYCL<char,   InvariantMassKernel, 8,  256>;
template class RDefSYCL<float,  InvariantMassKernel, 4,  256>;
template class RDefSYCL<double, InvariantMassKernel, 4,  256>;

// template class RDefSYCL<char,   InvariantMassKernel, 8,  512>;
template class RDefSYCL<float,  InvariantMassKernel, 4,  512>;
template class RDefSYCL<double, InvariantMassKernel, 4,  512>;

// template class RDefSYCL<char,   InvariantMassKernel, 8,  1024>;
template class RDefSYCL<float,  InvariantMassKernel, 4,  1024>;
template class RDefSYCL<double, InvariantMassKernel, 4,  1024>;
// clang-format on