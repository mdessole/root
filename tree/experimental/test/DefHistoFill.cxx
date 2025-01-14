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
using ROOT::Experimental::IdentityKernel;
struct SYCLHist {
   template <typename T>
   using type = RDefH1SYCL<T, IdentityKernel, 1>;

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
         EXPECT_FLOAT_EQ(a[i], b[i]) << "  at index i = " << i; \
      }                                                         \
   }

#define CHECK_ARRAY_DOUBLE(a, b, n)                              \
   {                                                             \
      for (auto i : ROOT::TSeqI(n)) {                            \
         EXPECT_DOUBLE_EQ(a[i], b[i]) << "  at index i = " << i; \
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
   const static int numBins = 7;

   // Variables for defining fixed bins.
   const double startBin = 1;
   const double endBin = 4;

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
using FillTestTypes = ::testing::Types<double, float, int, short>;
#endif
TYPED_TEST_SUITE(FillTestFixture, FillTestTypes);


/////////////////////////////////////
/// Test Cases

TYPED_TEST(FillTestFixture, FillFixedBins)
{
   // double or float
   using t = typename TestFixture::dataType;
   auto &h = this->histogram;

   ROOT::RVecD coords = { this->startBin - 1,  (this->startBin + this->endBin) / 2., this->endBin + 1};

   auto weight = (t)1;

   std::vector<int> expectedHistBins = {0, this->nCells / 2, this->nCells - 1};
   
   h.Fill(coords);
   for (auto i = 0; i < (int)coords.size(); i++) {
      this->expectedHist[expectedHistBins[i]] = weight;
   }

   h.RetrieveResults(this->result, this->stats);

   {
      SCOPED_TRACE("Check fill result");
      CompareArrays(this->result, this->expectedHist, this->nCells);
   }

   {
      SCOPED_TRACE("Check statistics");
      this->GetExpectedStats(coords, weight);
      CompareArrays(this->stats, this->expectedStats, this->nStats);
   }
}

TYPED_TEST(FillTestFixture, FillFixedBinsWeighted)
{
   // double or float
   using t = typename TestFixture::dataType;
   auto &h = this->histogram;

   ROOT::RVecD coords = { this->startBin - 1,  (this->startBin + this->endBin) / 2., this->endBin + 1};

   auto weight = ROOT::RVecD(3, 7); 

   std::vector<int> expectedHistBins = {0, this->nCells / 2, this->nCells - 1};


   h.Fill(coords, weight); 
   for (auto i = 0; i < (int)coords.size(); i++) {
      this->expectedHist[expectedHistBins[i]] = (t)weight[0];
   }

   h.RetrieveResults(this->result, this->stats);

   {
      SCOPED_TRACE("Check fill result");
      CompareArrays(this->result, this->expectedHist, this->nCells);
   }

   {
      SCOPED_TRACE("Check statistics");
      this->GetExpectedStats(coords, (t)weight[0]);
      CompareArrays(this->stats, this->expectedStats, this->nStats);
   }
}

