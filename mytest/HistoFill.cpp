#include "ROOT/RDataFrame.hxx"
#include "TH1.h"
#include "TAxis.h"

template <typename T = double, typename HIST = TH1D>
struct HistProperties {
   T *array;
   int dim, nCells;
   double *stats;
   int nStats;

   HistProperties(ROOT::RDF::RResultPtr<HIST> &h)
   {
      dim = h->GetDimension();
      nStats = 2 + 2 * dim;
      if (dim > 1)
         nStats += TMath::Binomial(dim, 2);
      nCells = h->GetNcells();

      // Create a copy in case the array gets cleaned up by RDataframe before checking the results
      array = (T *)malloc(nCells * sizeof(T));
      auto histogram = h->GetArray();
      std::copy(histogram, histogram + nCells, array);

      stats = (double *)calloc(nStats, sizeof(double));
      h->GetStats(stats);
   }

   ~HistProperties()
   {
      free(array);
      free(stats);
   }
};
 
template <typename Hist, typename... Cols>
   auto GetHisto1D(Hist histMdl, Cols... cols)
   {
      int numRows = 42;
      int numBins = numRows - 2; // -2 to also test filling u/overflow.
      double startBin = 0;
      double startFill = startBin - 1;
      double endBin = numBins;
      double x = startFill;
      auto df = ROOT::RDataFrame(numRows).Define("x", [&]() { return x++; }).Define("w", [&]() { return x; });
      auto hptr = df.Histo1D(histMdl, cols...);
      auto h = HistProperties<double>(hptr);
      return h;
   }

   int main(){
      int numRows = 42;
      int numBins = numRows - 2; // -2 to also test filling u/overflow.
      double startBin = 0;
      double startFill = startBin - 1;
      double endBin = numBins;
      double x = startFill;
      auto mdl = ::TH1D("h", "h", numBins, startBin, endBin);
      std::string column = "x";
      auto h1 = GetHisto1D(mdl, column);
   }