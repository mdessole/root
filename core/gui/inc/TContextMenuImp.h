// @(#)root/base:$Id$
// Author: Nenad Buncic   08/02/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TContextMenuImp
#define ROOT_TContextMenuImp

#include "Rtypes.h"

class TContextMenu;
class TObject;
class TMethod;
class TFunction;

// interface to GUI independent context sensitive popup menus.
class TContextMenuImp {

protected:
   TContextMenu *fContextMenu{nullptr}; //TContextMenu associated with this implementation

   TContextMenuImp(const TContextMenuImp &cmi) : fContextMenu(cmi.fContextMenu) {}
   TContextMenuImp &operator=(const TContextMenuImp &cmi)
   {
      if (this != &cmi)
         fContextMenu = cmi.fContextMenu;
      return *this;
   }

public:
   TContextMenuImp(TContextMenu *c = nullptr) : fContextMenu(c) {}
   virtual ~TContextMenuImp() {}

   virtual TContextMenu *GetContextMenu() const { return fContextMenu; }

   virtual void Dialog(TObject *object, TFunction *function) { (void) object; (void) function; }
   virtual void Dialog(TObject *object, TMethod *method) { (void) object; (void) method; }
   virtual void DisplayPopup(Int_t x, Int_t y) { (void) x; (void) y; }

   ClassDef(TContextMenuImp,0) //Context sensitive popup menu implementation
};

#endif
