#ifndef ROOT_AccHeaders_H
#define ROOT_AccHeaders_H

#if defined(ROOT_MATH_SYCL)

#include <sycl/sycl.hpp>
#elif defined(ROOT_MATH_CUDA)
#include <math.h>
#else
#include <cmath>

#ifndef ROOT_MATH_ARCH 
#define ROOT_MATH_ARCH MathSYCL
#endif

#endif

#if defined(ROOT_MATH_CUDA) && defined(__CUDACC__)

#define __roodevice__ __device__
#define __roohost__ __host__
#define __rooglobal__ __global__

#ifndef ROOT_MATH_ARCH 
#define ROOT_MATH_ARCH MathCUDA
#endif

#else

#define __roodevice__
#define __roohost__
#define __rooglobal__

#endif

#if !defined(ROOT_MATH_SYCL) &&  !defined(ROOT_MATH_CUDA)

#ifndef ROOT_MATH_ARCH 
#define ROOT_MATH_ARCH Math
#endif

#endif

#endif // ROOT_AccHeaders_H

