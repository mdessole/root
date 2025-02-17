/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/**
\file RooConstVar.cxx
\class RooConstVar
\ingroup Roofitcore

Represents a constant real-valued object.
**/

#include "RooConstVar.h"
#include "RooNumber.h"



////////////////////////////////////////////////////////////////////////////////
/// Constructor with value
RooConstVar::RooConstVar(const char *name, const char *title, double value) :
  RooAbsReal(name,title)
{
  _fast = true;
  _value = value;
  setAttribute("Constant",true) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor
RooConstVar::RooConstVar(const RooConstVar& other, const char* name) :
  RooAbsReal(other, name)
{
  _fast = true;
}

////////////////////////////////////////////////////////////////////////////////
/// Write object contents to stream

void RooConstVar::writeToStream(std::ostream& os, bool /*compact*/) const
{
  os << _value ;
}
