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
using RDefIdSYCL_t = RDefSYCL<double, IdentityKernel, 1>;
using RDefIMSYCL_t = RDefSYCL<double, InvariantMassesKernel, 8>;

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

auto bulkReturnId = [](const REventMask &m, ROOT::RVec<double> &output, const ROOT::RVec<ULong64_t> &rdfentries) {
   std::copy(rdfentries.begin(), rdfentries.begin() + m.Size(), output.begin());
};

auto bulkReturnOne = [](const REventMask &m, ROOT::RVec<double> &output) {
   std::fill(output.begin(), output.begin() + m.Size(), 1.);
};

auto bulkReturnTwo = [](const REventMask &m, ROOT::RVec<double> &output) {
   std::fill(output.begin(), output.begin() + m.Size(), 2.);
};

auto bulkReturnThree = [](const REventMask &m, ROOT::RVec<double> &output) {
   std::fill(output.begin(), output.begin() + m.Size(), 3.);
};

auto bulkReturnFour = [](const REventMask &m, ROOT::RVec<double> &output) {
   std::fill(output.begin(), output.begin() + m.Size(), 4.);
};

auto bulkReturnFive = [](const REventMask &m, ROOT::RVec<double> &output) {
   std::fill(output.begin(), output.begin() + m.Size(), 5.);
};

auto bulkReturnSix = [](const REventMask &m, ROOT::RVec<double> &output) {
   std::fill(output.begin(), output.begin() + m.Size(), 6.);
};

auto bulkReturnSeven = [](const REventMask &m, ROOT::RVec<double> &output) {
   std::fill(output.begin(), output.begin() + m.Size(), 7.);
};

auto bulkReturnEight = [](const REventMask &m, ROOT::RVec<double> &output) {
   std::fill(output.begin(), output.begin() + m.Size(), 8.);
};

auto bulkReturnIM = [](const REventMask &m, ROOT::RVec<double> &output, 
   const ROOT::RVec<double> &pt1, const ROOT::RVec<double> &eta1, const ROOT::RVec<double> &phi1, const ROOT::RVec<double> &m1,
   const ROOT::RVec<double> &pt2, const ROOT::RVec<double> &eta2, const ROOT::RVec<double> &phi2, const ROOT::RVec<double> &m2
) {
   // Result cross-validated using:
   // TLorentzVector t1, t2;
   // t1.SetPtEtaPhiM(1,2,3,4); t2.SetPtEtaPhiM(5,6,7,8);
   // (t1+t2).M()
   std::fill(output.begin(), output.begin() + m.Size(), 62.030584);
};

class DefIdentityFixture : public testing::TestWithParam<const char *> {
protected:
   uint numRows;
   uint maxBulkSize; 


   DefIdentityFixture()
   {
      numRows = 512;
      maxBulkSize = 256;
   }

   auto GetDefineSYCLResult()
   {
      auto df = ROOT::RDataFrame(numRows, maxBulkSize);
      auto df2 = df.DefineSYCL<RDefIdSYCL_t>("id", bulkReturnId, {"rdfentry_"});
      auto id = df2.Take<double, RVec<double>>("id").GetValue();
      return id;
   }


   auto GetDefineResult()
   {
      auto df = ROOT::RDataFrame(numRows, maxBulkSize);
      auto df2 = df.Define("id", bulkReturnId, {"rdfentry_"});
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

class DefInvariantMassesFixture : public testing::TestWithParam<const char *> {
   protected:
      uint numRows;
      uint maxBulkSize; 
   
   
      DefInvariantMassesFixture()
      {
         numRows = 512;
         maxBulkSize = 256;
      }
   
      auto GetDF()
      {
         auto df = ROOT::RDataFrame(numRows, maxBulkSize); 
         auto df2 = df.Define("pt1", bulkReturnOne, {});
         auto df3 = df2.Define("eta1", bulkReturnTwo, {});
         auto df4 = df3.Define("phi1", bulkReturnThree, {});
         auto df5 = df4.Define("m1", bulkReturnFour, {});
         auto df6 = df5.Define("pt2", bulkReturnFive, {});
         auto df7 = df6.Define("eta2", bulkReturnSix, {});
         auto df8 = df7.Define("phi2", bulkReturnSeven, {});
         auto df9 = df8.Define("m2", bulkReturnEight, {});
         return df9;
      }

      auto GetDefineSYCLResult(ROOT::RDF::RNode df)
      {
         auto df2 = df.DefineSYCL<RDefIMSYCL_t>("IM", bulkReturnIM, {"pt1", "eta1", "phi1", "m1", "pt2", "eta2", "phi2", "m2"});
         auto id = df2.Take<double, RVec<double>>("IM").GetValue();
         return id;
      }
   
   
      auto GetDefineResult(ROOT::RDF::RNode df)
      {
         auto df2 = df.Define("IM", bulkReturnIM, {"pt1", "eta1", "phi1", "m1", "pt2", "eta2", "phi2", "m2"});
         auto id = df2.Take<double, RVec<double>>("IM").GetValue();
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
   

/***
 * Test Define Identity
 */

TEST_F(DefIdentityFixture, IdentityDefine)
{

   auto id1 = GetDefineSYCLResult();

   auto id2 = GetDefineResult(); 

   CompareArrays(id1.data(), id2.data(), numRows); 
}


/***
 * Test Define Invariant Masses
 */

 TEST_F(DefInvariantMassesFixture, InvariantMassesDefine)
 {

    auto df = GetDF();
 
    auto id2 = GetDefineResult(df); 

    auto id1 = GetDefineSYCLResult(df);

 
    CompareArrays(id1.data(), id2.data(), numRows); 
 }
 
 