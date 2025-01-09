# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

set (testLibs ROOTDataFrame)

if (cuda)
    list(APPEND testLibs CUDAHist)
endif()

if (sycl)
    list(APPEND testLibs SYCLHist)
endif()

if (sycl OR cuda)
    ROOT_ADD_GTEST(HistoFill histoFill.cxx LIBRARIES ${testLibs} Hist)
    ROOT_ADD_GTEST(HistoRDF histoRDF.cxx LIBRARIES ${testLibs})
    ROOT_ADD_GTEST(DefHistoFill DefHistoFill.cxx LIBRARIES ${testLibs})
endif()

if(sycl)
 target_compile_definitions(HistoRDF PUBLIC ROOT_RDF_SYCL)
 target_compile_definitions(DefHistoFill PUBLIC ROOT_RDF_SYCL)
endif()