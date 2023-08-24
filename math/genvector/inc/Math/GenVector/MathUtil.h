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

#ifdef ROOT_MATH_SYCL 
#include <CL/sycl.hpp>
#endif 


namespace ROOT {

namespace Math {


#ifdef ROOT_MATH_SYCL 
template <class Scalar> Scalar mysin(Scalar x){
      return cl::sycl::sin(x);}

template <class Scalar>
 Scalar mycos(Scalar x)
 {    return cl::sycl::cos(x);}

template <class Scalar>
 Scalar myatan2(Scalar x, Scalar y)
 {    return cl::sycl::atan2(x,y);}

template <class Scalar>
 Scalar mysqrt(Scalar x)
 {      return cl::sycl::sqrt(x);}

template <class Scalar>
 Scalar myfloor(Scalar x)
 {       return cl::sycl::floor(x);}


#else 

template <class Scalar> Scalar mysin(Scalar x){
      return std::sin(x);}

template <class Scalar>
 Scalar mycos(Scalar x)
 {          return std::cos(x);}

template <class Scalar>
 Scalar myatan2(Scalar x, Scalar y)
 {    return std::atan2(x,y);}

template <class Scalar>
 Scalar mysqrt(Scalar x)
 {      return std::sqrt(x);}

template <class Scalar>
 Scalar myfloor(Scalar x)
 {       return std::floor(x);} 
#endif 
   

}// end namespace Experimental

} // end namespace ROOT

#endif