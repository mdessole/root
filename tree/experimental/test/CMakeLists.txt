# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

if (sycl OR cuda)
    ROOT_ADD_GTEST(InvariantMass invariantMass.cxx LIBRARIES ROOTVecOps Physics)
endif()

if (cuda)
    target_link_libraries(InvariantMass InvariantMassCUDA)
endif()

if (sycl)
    target_link_libraries(InvariantMass InvariantMassSYCL)
endif()
