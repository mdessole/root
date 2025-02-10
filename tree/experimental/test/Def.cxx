////////////////////////////////////////////////////////////////////////////////////
/// Tests for filling RHnCUDA histograms with different data types and dimensions.
///
#include <climits>
#include <stdlib.h>
#include "gtest/gtest.h"

#include "ROOT/RDataFrame.hxx"
#include "TH1.h"
#include "TAxis.h"


#include "RDefSYCL.h"

using ROOT::Experimental::RDefSYCL;
using ROOT::Experimental::InvariantMassesKernel;
struct SYCLDef {
   template <typename T>
   using type = RDefSYCL<T, InvariantMassesKernel, 8>;

};


std::vector<const char *> test_environments = {"SYCL_HIST"};

/**
 * Helper functions for toggling ON/OFF GPU histogramming.
 */

void DisableGPU()
{
   for (unsigned int i = 0; i < test_environments.size(); i++)
      unsetenv(test_environments[i]);
}

void EnableGPU(const char *env)
{
   DisableGPU();
   setenv(env, "1", 1);
}

// Returns an array with the given value repeated n times.
template <typename T, int n>
std::array<T, n> Repeat(T val)
{
   std::array<T, n> result;
   result.fill(val);
   return result;
}

// Helper functions for element-wise comparison of histogram arrays.
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


template <typename T>
class DefineFixture : public ::testing::Test {
protected:


   std::vector<double> params{};

   // int, double, float
   using dataType = T;

   // cuda or sycl
   const char *env = test_environments[0];


   SYCLDef::type<T> define;

   DefineFixture()
      : define(32768, params)
   {
   }

   void SetUp() override
   {
      EnableGPU(env);
   }

};

 
using DefTestTypes = ::testing::Types<double>;
TYPED_TEST_SUITE(DefineFixture, DefTestTypes);


/////////////////////////////////////
/// Test Cases

TYPED_TEST(DefineFixture, InvariantMasses)
{
   // int, double, or float
   using t = typename TestFixture::dataType;
   auto &d = this->define;
   size_t nPart = 5;
   t result[nPart];
   
   // Dummy particle collections
   ROOT::RVec<double> mass1 = {40,  50,  50,   50,   100};
   ROOT::RVec<double> pt1 =   {0,   5,   5,    10,   10};
   ROOT::RVec<double> eta1 =  {0.0, 0.0, -1.0, 0.5,  2.5};
   ROOT::RVec<double> phi1 =  {0.0, 0.0, 0.0,  -0.5, -2.4};

   ROOT::RVec<double> mass2 = {40,  40,  40,  40,  30};
   ROOT::RVec<double> pt2 =   {0,   5,   5,   10,  2};
   ROOT::RVec<double> eta2 =  {0.0, 0.0, 0.5, 0.4, 1.2};
   ROOT::RVec<double> phi2 =  {0.0, 0.0, 0.0, 0.5, 2.4};

   // Results
   ROOT::RVec<double> InvMasses = {80, 90.00685740426075654, 90.37681989852670483, 90.53569752667735315, 132.74260423154888144};

   ROOT::RVecD coords(nPart*8);
   for (size_t i = 0; i < nPart; i++){
      coords[i+0*nPart] = pt1[i];
      coords[i+1*nPart] = eta1[i];
      coords[i+2*nPart] = phi1[i];
      coords[i+3*nPart] = mass1[i];
      coords[i+4*nPart] = pt2[i];
      coords[i+5*nPart] = eta2[i];
      coords[i+6*nPart] = phi2[i];
      coords[i+7*nPart] = mass2[i];
   }


   
   d.EvalBulkExpr(coords);
   d.RetrieveResults(result, nPart);

   {
      SCOPED_TRACE("Check Define result");
      CompareArrays(result, InvMasses.data(), nPart);
   }


}
