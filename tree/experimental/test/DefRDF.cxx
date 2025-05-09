////////////////////////////////////////////////////////////////////////////////////
/// Compares results of RHnCUDA and TH* with Histo*D
/// Note that these histograms are only of type double
///

#include <stdlib.h>
#include <iostream>
#include "gtest/gtest.h"

#include "ROOT/RDF/REventMask.hxx"
#include <ROOT/RDataFrame.hxx>
#include "ROOT/RVec.hxx"
#include "TH1.h"
#include "TAxis.h"

#include "RDefSYCL.h"

using ROOT::RDF::Experimental::REventMask;
using ROOT::VecOps::RVec;
using ROOT::Experimental::RDefSYCL;
using ROOT::Experimental::InvariantMassesKernel;
using ROOT::Experimental::IdentityKernel;
using RDefSYCL_t = RDefSYCL<double, IdentityKernel, 1>;


auto bulkReturnX = [](const REventMask &m, ROOT::RVec<double> &output, const ROOT::RVec<ULong64_t> &rdfentries) {
   // ignoring event mask
   std::copy(rdfentries.begin(), rdfentries.begin() + m.Size(), output.begin());
};

class DefIdentityFixture : public testing::TestWithParam<const char *> {
protected:
   uint numRows; // -2 to also test filling u/overflow.
   uint maxBulkSize; // -2 to also test filling u/overflow.


   DefIdentityFixture()
   {
      numRows = 512;
      maxBulkSize = 256;
   }

   auto GetDefineSYCLResult()
   {
      auto df = ROOT::RDataFrame(numRows, maxBulkSize); // default maxbulksize = 256
      auto df2 = df.DefineSYCL<RDefSYCL_t>("id", bulkReturnX, {"rdfentry_"});
      auto id = df2.Take<double, RVec<double>>("id").GetValue();
      return id;
   }


   auto GetDefineResult()
   {
      auto df = ROOT::RDataFrame(numRows, maxBulkSize);
      auto df2 = df.Define("id", bulkReturnX, {"rdfentry_"});
      auto id = df2.Take<double, RVec<double>>("id").GetValue();
      return id;
   }

   auto GetExpectedResult()
   {
      auto id = std::vector<double>(numRows);    
      for (size_t i = 0; i < numRows; i++)
         id[i] = static_cast<double>(i);
      return id;
   }
   
};


// Helper functions for element-wise comparison of arrays.
#define CHECK_ARRAY(a, b, n)                              \
   {                                                      \
      for (auto i : ROOT::TSeqI(n)) {                     \
         EXPECT_EQ(a[i], b[i]) << "  at index i = " << i; \
      }                                                   \
   }

#define CHECK_ARRAY_FLOAT(a, b, n)                              \
   {                                                            \
      for (auto i : ROOT::TSeqI(n)) {                           \
         EXPECT_NEAR(a[i], b[i], 1e-4) << "  at index i = " << i; \
      }                                                         \
   }

#define CHECK_ARRAY_DOUBLE(a, b, n)                              \
   {                                                             \
      for (auto i : ROOT::TSeqI(n)) {                            \
         EXPECT_NEAR(a[i], b[i], 1e-4) << "  at index i = " << i; \
      }                                                          \
   }

template <typename T>
void CompareArrays(T *result, T *expected, int n)
{
   CHECK_ARRAY(result, expected, n)
}

template <>
void CompareArrays(float *result, float *expected, int n)
{
   CHECK_ARRAY_FLOAT(result, expected, n)
}

template <>
void CompareArrays(double *result, double *expected, int n)
{
   CHECK_ARRAY_DOUBLE(result, expected, n)
}

/***
 * Test Define identity
 */

TEST_F(DefIdentityFixture, IdentityDefine)
{

   auto id1 = GetDefineSYCLResult();

   auto id2 = GetDefineResult(); // GetExpectedResult(); 


   CompareArrays(id1.data(), id2.data(), numRows); 
}

