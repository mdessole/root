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
