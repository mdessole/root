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
#ifndef ROOT_Math_GenVector_MathUtil
#define ROOT_Math_GenVector_MathUtil  1

#include <CL/sycl.hpp>

namespace ROOT {

   namespace Math {

template <class Scalar>
inline sin(Scalar x){
#ifdef ROOT_RDF_SYCL 
      using cl::sycl::sin;
#else
      using std::sin;
#endif 
      return sin(x); }

template <class Scalar>
inline cos(Scalar x){
#ifdef ROOT_RDF_SYCL 
      using cl::sycl::cos;
#else
      using std::cos;
#endif 
      return cos(x); }

template <class Scalar>
inline atan2(Scalar x, Scalar y){
#ifdef ROOT_RDF_SYCL 
      using cl::sycl::atan2;
#else
      using std::atan2;
#endif 
      return atan2(x,y); }

template <class Scalar>
inline sqrt(Scalar x){
#ifdef ROOT_RDF_SYCL 
      using cl::sycl::sqrt;
#else
      using std::sqrt;
#endif 
      return sqrt(x); }

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_GenVector_MathUtil  */