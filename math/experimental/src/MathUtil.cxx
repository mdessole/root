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


#ifdef ROOT_MATH_SYCL 
#include <CL/sycl.hpp>
#endif 

#include "MathUtil.h"

namespace ROOT {

namespace Experimental {

template <class Scalar>
 Scalar mysin(Scalar x){
 #ifdef ROOT_MATH_SYCL 
      if (getenv("SYCL_MATH")) {return cl::sycl::sin(x);}
      else {return std::sin(x);}
#else 
      return std::sin(x);
#endif 
 } 
 
template <class Scalar>
 Scalar mycos(Scalar x) {
#ifdef ROOT_MATH_SYCL 
      if (getenv("SYCL_MATH")) {return cl::sycl::cos(x);}
      else {return std::cos(x);}
#else
      return std::cos(x);
#endif 
 }
 
template <class Scalar>
 Scalar myatan2(Scalar x, Scalar y){
#ifdef ROOT_MATH_SYCL 
      if (getenv("SYCL_MATH")) {return cl::sycl::atan2(x,y);}
      else {return std::atan2(x,y);}
#else
      return std::atan2(x,y);
#endif 
 }

template <class Scalar>
 Scalar mysqrt(Scalar x) {
#ifdef ROOT_MATH_SYCL 
      if (getenv("SYCL_MATH")) {return cl::sycl::sqrt(x);}
      else {return std::sqrt(x);}
#else
      return std::sqrt(x);
#endif 
   } 
 

template <class Scalar>
 Scalar myfloor(Scalar x){ 
#ifdef ROOT_MATH_SYCL 
      if (getenv("SYCL_MATH")) {return cl::sycl::floor(x);}
      else {return std::floor(x);}
#else
      return std::floor(x);
#endif 
  } 

#include "MathUtil-impl.cxx"

}// end namespace Experimental

} // end namespace ROOT
