// @(#)root/roostats:$Id$
// Author: L. Moneta
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class RooStats::HistFactory::HistRef
 * \ingroup HistFactory
 * Internal class wrapping an histogram and managing its content.
 * convenient for dealing with histogram pointers in the
 * HistFactory class
 */

#include "RooStats/HistFactory/HistRef.h"

#include "TH1.h"
#include "TDirectory.h"

namespace RooStats{
namespace HistFactory {

   TH1 * HistRef::CopyObject(const TH1 * h) {
      // implementation of method copying the contained pointer
      // (just use Clone)
      if (!h) return nullptr;

      TDirectory::TContext ctx{nullptr}; // Don't associate histogram with currently open file
      return static_cast<TH1*>(h->Clone());
   }
}
}


