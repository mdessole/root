// @(#)root/mathcore:$Id: 9ef2a4a7bd1b62c1293920c2af2f64791c75bdd8 $
// Authors: W. Brown, M. Fischler, L. Moneta    2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for Vector Utility functions
//
// Created by: moneta  at Tue May 31 21:10:29 2005
//
// Last update: Tue May 31 21:10:29 2005
//

#ifndef MathUtil_H
#define MathUtil_H

#ifndef M_PI
#define M_PI 3.14159265358979323846264338328 // Pi
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923132169164 // Pi/2
#endif

#ifndef M_PI_4
#define M_PI_4 0.78539816339744830961566084582 // Pi/4
#endif

#ifdef ROOT_MATH_SYCL
#include <sycl/sycl.hpp>
#else
#include <cmath>
#endif

#include <limits>

namespace ROOT {

namespace Math {

#ifdef ROOT_MATH_SYCL
template <class Scalar>
Scalar mysin(Scalar x)
{
   return sycl::sin(x);
}

template <class Scalar>
Scalar mycos(Scalar x)
{
   return sycl::cos(x);
}

template <class Scalar>
Scalar myasin(Scalar x)
{
   return sycl::asin(x);
}

template <class Scalar>
Scalar myacos(Scalar x)
{
   return sycl::acos(x);
}

template <class Scalar>
Scalar mysinh(Scalar x)
{
   return sycl::sinh(x);
}

template <class Scalar>
Scalar mycosh(Scalar x)
{
   return sycl::cosh(x);
}

template <class Scalar>
Scalar myatan2(Scalar x, Scalar y)
{
   return sycl::atan2(x, y);
}

template <class Scalar>
Scalar myatan(Scalar x)
{
   return sycl::atan(x);
}

template <class Scalar>
Scalar mysqrt(Scalar x)
{
   return sycl::sqrt(x);
}

template <class Scalar>
Scalar myfloor(Scalar x)
{
   return sycl::floor(x);
}

template <class Scalar>
Scalar myexp(Scalar x)
{
   return sycl::exp(x);
}

template <class Scalar>
Scalar mylog(Scalar x)
{
   return sycl::log(x);
}

template <class Scalar>
Scalar mytan(Scalar x)
{
   return sycl::tan(x);
}

template <class Scalar>
Scalar myfabs(Scalar x)
{
   return sycl::fabs(x);
}

template <class Scalar>
Scalar mypow(Scalar x, Scalar y)
{
   return sycl::pow(x, y);
}

template <class T>
T etaMax2()
{
   return static_cast<T>(22756.0);
}

template <typename Scalar>
inline Scalar Eta_FromRhoZ(Scalar rho, Scalar z)
{
   if (rho > 0) {
      // value to control Taylor expansion of sqrt
      // static const Scalar
      Scalar epsilon = static_cast<Scalar>(2e-16);
      const Scalar big_z_scaled = sycl::pow(epsilon, static_cast<Scalar>(-.25));

      Scalar z_scaled = z / rho;
      if (sycl::fabs(z_scaled) < big_z_scaled) {
         return sycl::log(z_scaled + sycl::sqrt(z_scaled * z_scaled + 1.0));
      } else {
         // apply correction using first order Taylor expansion of sqrt
         return z > 0 ? sycl::log(2.0 * z_scaled + 0.5 / z_scaled) : -sycl::log(-2.0 * z_scaled);
      }
      return z_scaled;
   }
   // case vector has rho = 0
   else if (z == 0) {
      return 0;
   } else if (z > 0) {
      return z + etaMax2<Scalar>();
   } else {
      return z - etaMax2<Scalar>();
   }
}

/**
   Implementation of eta from -log(tan(theta/2)).
   This is convenient when theta is already known (for example in a polar coorindate system)
*/
template <typename Scalar>
Scalar Eta_FromTheta(Scalar theta, Scalar r)
{
   Scalar tanThetaOver2 = mytan(theta / 2.);
   if (tanThetaOver2 == 0) {
      return r + etaMax2<Scalar>();
   } else if (tanThetaOver2 > std::numeric_limits<Scalar>::max()) {
      return -r - etaMax2<Scalar>();
   } else {
      return -mylog(tanThetaOver2);
   }
}

#else

template <class Scalar>
inline Scalar mysin(Scalar x)
{
   return std::sin(x);
}

template <class Scalar>
inline Scalar mycos(Scalar x)
{
   return std::cos(x);
}

template <class Scalar>
inline Scalar myasin(Scalar x)
{
   return std::asin(x);
}

template <class Scalar>
inline Scalar myacos(Scalar x)
{
   return std::acos(x);
}

template <class Scalar>
inline Scalar mysinh(Scalar x)
{
   return std::sinh(x);
}

template <class Scalar>
inline Scalar mycosh(Scalar x)
{
   return std::cosh(x);
}

template <class Scalar>
inline Scalar myatan2(Scalar x, Scalar y)
{
   return std::atan2(x, y);
}

template <class Scalar>
Scalar myatan(Scalar x)
{
   return std::atan(x);
}

template <class Scalar>
inline Scalar mysqrt(Scalar x)
{
   return std::sqrt(x);
}

template <class Scalar>
inline Scalar myfloor(Scalar x)
{
   return std::floor(x);
}

template <class Scalar>
inline Scalar myexp(Scalar x)
{
   return std::exp(x);
}

template <class Scalar>
inline Scalar mylog(Scalar x)
{
   return std::log(x);
}

template <class Scalar>
inline Scalar mytan(Scalar x)
{
   return std::tan(x);
}

template <class Scalar>
inline Scalar myfabs(Scalar x)
{
   return std::fabs(x);
}

template <class Scalar>
inline Scalar mypow(Scalar x, Scalar y)
{
   return std::pow(x, y);
}

template <class T>
inline T etaMax2()
{
   return static_cast<T>(22756.0);
}

template <typename Scalar>
inline Scalar Eta_FromRhoZ(Scalar rho, Scalar z)
{
   if (rho > 0) {

      // value to control Taylor expansion of sqrt
      static const Scalar big_z_scaled = mypow(std::numeric_limits<Scalar>::epsilon(), static_cast<Scalar>(-.25));

      Scalar z_scaled = z / rho;
      if (myfabs(z_scaled) < big_z_scaled) {
         return mylog(z_scaled + mysqrt(z_scaled * z_scaled + 1.0));
      } else {
         // apply correction using first order Taylor expansion of sqrt
         return z > 0 ? mylog(2.0 * z_scaled + 0.5 / z_scaled) : -mylog(-2.0 * z_scaled);
      }
   }
   // case vector has rho = 0
   else if (z == 0) {
      return 0;
   } else if (z > 0) {
      return z + etaMax2<Scalar>();
   } else {
      return z - etaMax2<Scalar>();
   }
}

/**
   Implementation of eta from -log(tan(theta/2)).
   This is convenient when theta is already known (for example in a polar coorindate system)
*/
template <typename Scalar>
inline Scalar Eta_FromTheta(Scalar theta, Scalar r)
{
   Scalar tanThetaOver2 = mytan(theta / 2.);
   if (tanThetaOver2 == 0) {
      return r + etaMax2<Scalar>();
   } else if (tanThetaOver2 > std::numeric_limits<Scalar>::max()) {
      return -r - etaMax2<Scalar>();
   } else {
      return -mylog(tanThetaOver2);
   }
}

#endif

} // namespace Math

} // end namespace ROOT

#endif