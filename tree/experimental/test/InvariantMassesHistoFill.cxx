////////////////////////////////////////////////////////////////////////////////////
/// Tests for filling RHnCUDA histograms with different data types and dimensions.
///
#include <climits>
#include <stdlib.h>
#include "gtest/gtest.h"

#include "ROOT/RDataFrame.hxx"
#include "TH1.h"
#include "TAxis.h"


#include "RDefH1SYCL.h"

using ROOT::Experimental::RDefH1SYCL;
using ROOT::Experimental::InvariantMassesKernel;
struct SYCLHist {
   template <typename T>
   using type = RDefH1SYCL<T, InvariantMassesKernel, 8>;

   static constexpr int histIdx = 1;
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
class FillTestFixture : public ::testing::Test {
protected:
   // Includes u/overflow bins. Uneven number chosen to have a center bin.
   const static int numBins = 4;

   // Variables for defining fixed bins.
   const double startBin = 90;
   const double endBin = 130;

   std::vector<double> params{};

   // int, double, float
   using dataType = T;

   // 1, 2, or 3
   static constexpr int dim = 1;

   // cuda or sycl
   const char *env = test_environments[0];

   // Total number of cells
   const static int nCells = pow(numBins, dim);
   dataType result[nCells], expectedHist[nCells];

   double *stats, *expectedStats;
   int nStats;

   SYCLHist::type<T> histogram;

   FillTestFixture()
      : histogram(32768, nCells, Repeat<int, dim>(numBins), Repeat<double, dim>(startBin), Repeat<double, dim>(endBin),
                  {}, Repeat<int, dim>(-1), params)
   {
   }

   void SetUp() override
   {
      EnableGPU(env);
      nStats = 2 + dim * 2 + dim * (dim - 1) / 2; // 2+2 = 4 -> sumw, sumw2, sumwx, sumwx2

      stats = new double[nStats];
      expectedStats = new double[nStats];

      memset(stats, 0, nStats * sizeof(double));
      memset(expectedStats, 0, nStats * sizeof(double));
      memset(expectedHist, 0, nCells * sizeof(dataType));
   }

   void TearDown() override { delete[] stats; }

   bool UOverflow(ROOT::RVecD coord)
   {
      for (auto d = 0; d < dim; d++) {
         if (coord[d] < startBin || coord[d] > endBin)
            return true;
      }
      return false;
   }

   void GetExpectedStats(ROOT::RVecD coords, dataType weight)
   {
      for (auto i = 0; i < (int)coords.size(); i++) {
         if  (coords[i] < startBin || coords[i] > endBin)
            continue;

         // Tsumw
         expectedStats[0] += weight;
         // Tsumw2
         expectedStats[1] += weight * weight;
         // e.g. Tsumwx
         expectedStats[2] += weight * coords[i];
         // e.g. Tsumwx2
         expectedStats[3] += weight * pow(coords[i], 2);
      }
   }
};


#if defined(ROOT_RDF_SYCL)
using FillTestTypes = ::testing::Types<double, float>;
#endif
TYPED_TEST_SUITE(FillTestFixture, FillTestTypes);



/////////////////////////////////////
/// Test Cases

TYPED_TEST(FillTestFixture, FillFixedBins)
{
   // int, double, or float
   using t = typename TestFixture::dataType;
   auto &h = this->histogram;

   size_t nPart = 5;
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


   auto weight = (t)1;

   std::vector<int> expectedHistBins = {0, 1, 1, 1, this->nCells - 1}; // length is equal to nPart
   
   h.Fill(coords);
   for (auto i = 0; i < nPart; i++) {
      this->expectedHist[expectedHistBins[i]] += weight;
   }

   h.RetrieveResults(this->result, this->stats);

   {
      SCOPED_TRACE("Check fill result");
      CompareArrays(this->result, this->expectedHist, this->nCells);
   }

   {
      SCOPED_TRACE("Check statistics");
      this->GetExpectedStats(InvMasses, weight);
      CompareArrays(this->stats, this->expectedStats, this->nStats);
   }
}

TYPED_TEST(FillTestFixture, FillFixedBinsWeighted)
{
   // int, double, or float
   using t = typename TestFixture::dataType;
   auto &h = this->histogram;

     size_t nPart = 5;
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

   auto weight = ROOT::RVecD(nPart, 7); 
   
   std::vector<int> expectedHistBins = {0, 1, 1, 1, this->nCells - 1}; // length is equal to nPart
   
   h.Fill(coords, weight);
   for (auto i = 0; i < nPart; i++) {
      this->expectedHist[expectedHistBins[i]] += weight[i];
   }

   h.RetrieveResults(this->result, this->stats);

   {
      SCOPED_TRACE("Check fill result");
      CompareArrays(this->result, this->expectedHist, this->nCells);
   }

   {
      SCOPED_TRACE("Check statistics");
      this->GetExpectedStats(InvMasses, (t)weight[0]);
      CompareArrays(this->stats, this->expectedStats, this->nStats);
   }
}


