#include "RtypesCore.h"
#include <ROOT/RDataFrame.hxx>

#include <gtest/gtest.h>

TEST(RDFBulkAPI, BulkDefine)
{
   auto bulkReturnX = [](const ROOT::RDF::Experimental::REventMask &m, ROOT::RVec<float> &output,
                         const ROOT::RVec<ULong64_t> &rdfentries) {
      // ignoring event mask
      std::copy(rdfentries.begin(), rdfentries.begin() + m.Size(), output.begin());
   };

   auto bulkSqr = [](const ROOT::RDF::Experimental::REventMask &m, ROOT::RVec<float> &output,
                     const ROOT::RVec<float> &xs) {
      // ignoring event mask
      std::transform(xs.begin(), xs.begin() + m.Size(), output.begin(), [](float x) { return x * x; });
   };

   auto m =
      ROOT::RDataFrame(10).Define("x", bulkReturnX, {"rdfentry_"}).Define("xx", bulkSqr, {"x"}).Mean<float>("xx");
   EXPECT_DOUBLE_EQ(m.GetValue(), 28.5);
}


struct BulkHelper : ROOT::Detail::RDF::RActionImpl<BulkHelper> {
   static constexpr bool kUseBulk = true;
   using Result_t = int;
   std::shared_ptr<int> x = std::make_shared<int>(42);

   std::shared_ptr<int> GetResultPtr() const { return x; }

   void Exec(const ROOT::RDF::Experimental::REventMask &m, const ROOT::RVecULL& es) {
      EXPECT_EQ(m.FirstEntry(), 0ull);
      EXPECT_EQ(m.Size(), 10ull);
      for (ULong64_t i = 0ull; i < 10ull; ++i)
         EXPECT_EQ(i, es[i]);
   }

   void Initialize() {}
   void InitTask(TTreeReader *, unsigned) {}
   void Finalize() {}
   std::string GetActionName() { return "custom"; }
};


TEST(RDFBulkAPI, BulkAction)
{
  auto df = ROOT::RDataFrame(10);
  auto r = df.Book<ULong64_t>(BulkHelper{}, {"rdfentry_"});
  r.GetValue();
}


/*
TEST(RDFBulkAPI, BulkFilter)
{
   auto bulkReturnX = [](const ROOT::RDF::Experimental::REventMask &m, ROOT::RVec<float> &output,
                         const ROOT::RVec<ULong64_t> &rdfentries) {
      // ignoring event mask
      std::copy(rdfentries.begin(), rdfentries.begin() + m.Size(), output.begin());
   };

   auto bulkSqr = [](const ROOT::RDF::Experimental::REventMask &m, std::size_t bulkSize, ROOT::RVec<float> &output,
                     const ROOT::RVec<float> &xs) {
      for (std::size_t i = 0u; i < m.Size(); ++i) {
         if (m[i]) {
            output[i] = xs[i] * xs[i];
         }
      }
   };

   auto m =
      ROOT::RDataFrame(10)
         .Filter([](const ROOT::RDF::Experimental::REventMask &, std::size_t bulkSize, ROOT::RVec<bool> &outputMask,
                    const ROOT::RVec<ULong64_t> &rdfentries) { outputMask = rdfentries > 5; },
                 {"rdfentry_"})
         .Define("x", bulkReturnX, {"rdfentry_"})
         .Define("xx", bulkSqr, {"x"})
         .Mean<double>("xx");
   EXPECT_DOUBLE_EQ(m.GetValue(), 57.5);
}
*/
