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
#define M_PI       3.14159265358979323846264338328      // Pi
#endif

#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923132169164      // Pi/2
#endif

#ifndef M_PI_4
#define M_PI_4     0.78539816339744830961566084582      // Pi/4
#endif


#ifdef ROOT_MATH_SYCL 
#include <CL/sycl.hpp>
#endif 

#include <limits>

namespace ROOT {

namespace Math {


#ifdef ROOT_MATH_SYCL 
template <class Scalar> Scalar mysin(Scalar x){
      return cl::sycl::sin(x);}

template <class Scalar>
 Scalar mycos(Scalar x)
 {    return cl::sycl::cos(x);}

template <class Scalar> 
Scalar mysinh(Scalar x){
      return cl::sycl::sinh(x);}

template <class Scalar>
 Scalar mycosh(Scalar x)
 {    return cl::sycl::cosh(x);}

template <class Scalar>
 Scalar myatan2(Scalar x, Scalar y)
 {    return cl::sycl::atan2(x,y);}

template <class Scalar>
 Scalar mysqrt(Scalar x)
 {      return cl::sycl::sqrt(x);}

template <class Scalar>
 Scalar myfloor(Scalar x)
 {       return cl::sycl::floor(x);}

template <class Scalar>
 Scalar myexp(Scalar x)
 {       return cl::sycl::exp(x);}

template <class Scalar>
 Scalar mylog(Scalar x)
 {       return cl::sycl::log(x);}

template <class Scalar>
 Scalar mytan(Scalar x)
 {       return cl::sycl::tan(x);}

template <class Scalar>
 Scalar myfabs(Scalar x)
 {    return cl::sycl::fabs(x);}

    template <class T>
    inline
    T etaMax2() {
      return static_cast<T>(22756.0);
    }


        template<typename Scalar>
        inline Scalar Eta_FromRhoZ(Scalar rho, Scalar z) {
           if (rho > 0) {

              // value to control Taylor expansion of sqrt
              static const Scalar big_z_scaled = pow(std::numeric_limits<Scalar>::epsilon(), static_cast<Scalar>(-.25));

              Scalar z_scaled = z/rho;
              if (myfabs(z_scaled) < big_z_scaled) {
                 return log(z_scaled + mysqrt(z_scaled * z_scaled + 1.0));
              } else {
                 // apply correction using first order Taylor expansion of sqrt
                 return z > 0 ? mylog(2.0 * z_scaled + 0.5 / z_scaled) : -mylog(-2.0 * z_scaled);
              }
           }
           // case vector has rho = 0
           else if (z==0) {
              return 0;
           }
           else if (z>0) {
              return z + etaMax2<Scalar>();
           }
           else {
              return z - etaMax2<Scalar>();
           }

        }


        /**
           Implementation of eta from -log(tan(theta/2)).
           This is convenient when theta is already known (for example in a polar coorindate system)
        */
        template<typename Scalar>
        inline Scalar Eta_FromTheta(Scalar theta, Scalar r) {
           Scalar tanThetaOver2 = mytan(theta / 2.);
           if (tanThetaOver2 == 0) {
              return r + etaMax2<Scalar>();
           }
           else if (tanThetaOver2 > std::numeric_limits<Scalar>::max()) {
              return -r - etaMax2<Scalar>();
           }
           else {
              return -mylog(tanThetaOver2);
           }

        }

#else 

template <class Scalar> Scalar mysin(Scalar x){
      return std::sin(x);}

template <class Scalar>
 Scalar mycos(Scalar x)
 {          return std::cos(x);}

template <class Scalar> 
Scalar mysinh(Scalar x){
      return std::sinh(x);}

template <class Scalar>
 Scalar mycosh(Scalar x)
 {    return std::cosh(x);}

template <class Scalar>
 Scalar myatan2(Scalar x, Scalar y)
 {    return std::atan2(x,y);}

template <class Scalar>
 Scalar mysqrt(Scalar x)
 {      return std::sqrt(x);}

template <class Scalar>
 Scalar myfloor(Scalar x)
 {       return std::floor(x);} 

template <class Scalar>
 Scalar myexp(Scalar x)
 {       return std::exp(x);}

template <class Scalar>
 Scalar mylog(Scalar x)
 {       return std::log(x);}

template <class Scalar>
 Scalar mytan(Scalar x)
 {       return std::tan(x);}

template <class Scalar>
 Scalar myfabs(Scalar x)
 {    return std::fabs(x);}

    template <class T>
    inline
    T etaMax2() {
      return static_cast<T>(22756.0);
    }


        template<typename Scalar>
        inline Scalar Eta_FromRhoZ(Scalar rho, Scalar z) {
           if (rho > 0) {

              // value to control Taylor expansion of sqrt
              static const Scalar big_z_scaled = pow(std::numeric_limits<Scalar>::epsilon(), static_cast<Scalar>(-.25));

              Scalar z_scaled = z/rho;
              if (myfabs(z_scaled) < big_z_scaled) {
                 return log(z_scaled + mysqrt(z_scaled * z_scaled + 1.0));
              } else {
                 // apply correction using first order Taylor expansion of sqrt
                 return z > 0 ? mylog(2.0 * z_scaled + 0.5 / z_scaled) : -mylog(-2.0 * z_scaled);
              }
           }
           // case vector has rho = 0
           else if (z==0) {
              return 0;
           }
           else if (z>0) {
              return z + etaMax2<Scalar>();
           }
           else {
              return z - etaMax2<Scalar>();
           }

        }


        /**
           Implementation of eta from -log(tan(theta/2)).
           This is convenient when theta is already known (for example in a polar coorindate system)
        */
        template<typename Scalar>
        inline Scalar Eta_FromTheta(Scalar theta, Scalar r) {
           Scalar tanThetaOver2 = mytan(theta / 2.);
           if (tanThetaOver2 == 0) {
              return r + etaMax2<Scalar>();
           }
           else if (tanThetaOver2 > std::numeric_limits<Scalar>::max()) {
              return -r - etaMax2<Scalar>();
           }
           else {
              return -mylog(tanThetaOver2);
           }

        }

#endif 
   

}// end namespace Experimental

} // end namespace ROOT

#endif