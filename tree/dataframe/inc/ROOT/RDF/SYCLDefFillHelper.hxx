

/**
 \file ROOT/RDF/SYCLDefFillHelper.hxx
 \ingroup dataframe
 \author Jolly Chen, CERN
 \date 2023-06
 TODO: This file is basically an exact copy of CUDAFillHelper but with "CUDA" exchanged with "SYCL".
       Should try to make it a single class
*/

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_SYCLDEFFILLHELPER
#define ROOT_SYCLDEFFILLHELPER

#include "ROOT/RVec.hxx"
#include <ROOT/RDF/RAction.hxx>
#include "ROOT/RDF/RActionImpl.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "RDefKernel.h"
#include "RDefH1SYCL.h"
#include "TH1.h"
#include "TStatistic.h"

#include <vector>
#include <string>
#include <array>

using ROOT::Internal::RDF::Disjunction;
using ROOT::Internal::RDF::FindIdxTrue;
using ROOT::Internal::RDF::GetNthElement;
using ROOT::Internal::RDF::IsDataContainer;
using ROOT::Internal::RDF::RActionImpl;
using ROOT::Internal::RDF::RMergeableFill;
using ROOT::Internal::RDF::RMergeableValueBase;

namespace ROOT {
namespace Experimental {
using Hist_t = ::TH1D;

template <typename SYCLHist, typename HIST = Hist_t>
class R__CLING_PTRCHECK(off) SYCLDefFillHelper : public RActionImpl<SYCLDefFillHelper<SYCLHist, HIST>> {
   // clang-format off
   static constexpr size_t getHistDim(TH3 *) { return 3; }
   static constexpr size_t getHistDim(TH2 *) { return 2; }
   static constexpr size_t getHistDim(TH1 *) { return 1; }

   static constexpr char getHistType(TH1C *) { return (char) 0; }
   static constexpr char getHistType(TH2C *) { return (char) 0; }
   static constexpr char getHistType(TH3C *) { return (char) 0; }
   static constexpr short getHistType(TH1S *) { return (short) 0; }
   static constexpr short getHistType(TH2S *) { return (short) 0; }
   static constexpr short getHistType(TH3S *) { return (short) 0; }
   static constexpr int getHistType(TH1I *) { return 0; }
   static constexpr int getHistType(TH2I *) { return 0; }
   static constexpr int getHistType(TH3I *) { return 0; }
   static constexpr float getHistType(TH1F *) { return (float) 0; }
   static constexpr float getHistType(TH2F *) { return (float) 0; }
   static constexpr float getHistType(TH3F *) { return (float) 0; }
   static constexpr double getHistType(TH1D *) { return (double) 0; }
   static constexpr double getHistType(TH2D *) { return (double) 0; }
   static constexpr double getHistType(TH3D *) { return (double) 0; }
   // clang-format on

   static constexpr size_t dim = getHistDim((HIST *)nullptr);

   using SYCLHist_t = SYCLHist;

   HIST *fObject;
   std::unique_ptr<SYCLHist_t> fSYCLHist;
   std::vector<decltype(getHistType((HIST *)nullptr))> fParams{};

   template <typename H = HIST, typename = decltype(std::declval<H>().Reset())>
   void ResetIfPossible(H *h)
   {
      h->Reset();
   }

   void ResetIfPossible(TStatistic *h) { *h = TStatistic(); }

   // cannot safely re-initialize variations of the result, hence error out
   void ResetIfPossible(...)
   {
      throw std::runtime_error(
         "A systematic variation was requested for a custom Fill action, but the type of the object to be filled does"
         "not implement a Reset method, so we cannot safely re-initialize variations of the result. Aborting.");
   }

   void UnsetDirectoryIfPossible(TH1 *h) { h->SetDirectory(nullptr); }

   void UnsetDirectoryIfPossible(...) {}

   // Bulk fill overloads
   // TODO: masking on GPU?
   template <std::size_t... Is, typename... ValTypes>
   void Fill(const ROOT::RDF::Experimental::REventMask &m, std::index_sequence<Is...>, const ValTypes &...x)
   {
      RVecD coords;
      coords.reserve(m.Size() * dim);
      [[maybe_unused]] RVecD weights;
      if constexpr (sizeof...(ValTypes) > dim)
         weights.reserve(m.Size());

      auto maskedInsert = [&](auto &arr, auto &out) {
         for (std::size_t i = 0ul; i < m.Size(); ++i) {
            if (m[i])
               out.emplace_back(arr[i]);
         }
      };

      // Converts the arrays in the parameter pack x for coordinates in each dimension,
      // RVec(x1, x2, ...), RVec(y1, y2, ...), RVec(z1, z2, ...), ... into a single RVec in the form of
      // RVec(x1, x2, ... y1, y2, ... z1, x2, ....)
      // The parameter pack x may or may not include a vector containing the weights as the last element
      // which needs to be placed in the weights array.
      (maskedInsert(x, Is < dim ? coords : weights), ...);

      if constexpr (sizeof...(ValTypes) > dim)
         fSYCLHist->Fill(coords, weights);
      else
         fSYCLHist->Fill(coords);
   }

   // Merge overload for types with Merge(TCollection*), like TH1s
   template <typename H, typename = std::enable_if_t<std::is_base_of<TObject, H>::value, int>>
   auto Merge(std::vector<H *> &objs, int /*toincreaseoverloadpriority*/)
      -> decltype(objs[0]->Merge((TCollection *)nullptr), void())
   {
      TList l;
      for (auto it = ++objs.begin(); it != objs.end(); ++it)
         l.Add(*it);
      objs[0]->Merge(&l);
   }

   // Merge overload for types with Merge(const std::vector&)
   template <typename H>
   auto Merge(std::vector<H *> &objs, double /*toloweroverloadpriority*/)
      -> decltype(objs[0]->Merge(std::vector<HIST *>{}), void())
   {
      objs[0]->Merge({++objs.begin(), objs.end()});
   }

   // Merge overload to error out in case no valid HIST::Merge method was detected
   template <typename T>
   void Merge(T, ...)
   {
      static_assert(
         sizeof(T) < 0,
         "The type passed to Fill does not provide a Merge(TCollection*) or Merge(const std::vector&) method.");
   }

   // class which wraps a pointer and implements a no-op increment operator
   template <typename T>
   class ScalarConstIterator {
      const T *obj_;

   public:
      ScalarConstIterator(const T *obj) : obj_(obj) {}
      const T &operator*() const { return *obj_; }
      ScalarConstIterator<T> &operator++() { return *this; }
   };

   // helper functions which provide one implementation for scalar types and another for containers
   // TODO these could probably all be replaced by inlined lambdas and/or constexpr if statements
   // in c++17 or later

   // return unchanged value for scalar
   template <typename T, std::enable_if_t<!IsDataContainer<T>::value, int> = 0>
   ScalarConstIterator<T> MakeBegin(const T &val)
   {
      return ScalarConstIterator<T>(&val);
   }

   // return iterator to beginning of container
   template <typename T, std::enable_if_t<IsDataContainer<T>::value, int> = 0>
   auto MakeBegin(const T &val)
   {
      return std::begin(val);
   }

   // return 1 for scalars
   template <typename T, std::enable_if_t<!IsDataContainer<T>::value, int> = 0>
   std::size_t GetSize(const T &)
   {
      return 1;
   }

   // return container size
   template <typename T, std::enable_if_t<IsDataContainer<T>::value, int> = 0>
   std::size_t GetSize(const T &val)
   {
#if __cplusplus >= 201703L
      return std::size(val);
#else
      return val.size();
#endif
   }

   template <std::size_t ColIdx, typename End_t, typename... Its>
   void ExecLoop(End_t end, Its... its)
   {
      // loop increments all of the iterators while leaving scalars unmodified
      // TODO this could be simplified with fold expressions or std::apply in C++17
      auto nop = [](auto &&...) {};
      for (; GetNthElement<ColIdx>(its...) != end; nop(++its...)) {
         fObject->Fill(*its...);
      }
   }

public:
   static constexpr bool kUseBulk = true;

   SYCLDefFillHelper(SYCLDefFillHelper &&) = default;
   SYCLDefFillHelper(const SYCLDefFillHelper &) = delete;

   // Initialize fSYCLHist
   inline void init_SYCL(HIST *obj, int i, size_t maxBulkSize)
   {
      if (getenv("DBG"))
         printf("Init SYCL DefHist %d\n", i);
      auto dims = obj->GetDimension();

      std::array<int, dim> ncells;
      std::array<double, dim> xlow;
      std::array<double, dim> xHigh;
      std::vector<double> binEdges;
      std::array<int, dim> binEdgesIdx;
      TAxis *ax;

      Size_t numBins = 1;
      int numBinEdges = 0;
      for (auto d = 0; d < dims; d++) {
         if (d == 0) {
            ax = obj->GetXaxis();
         } else if (d == 1) {
            ax = obj->GetYaxis();
         } else {
            ax = obj->GetZaxis();
         }

         auto ncells_d = ax->GetNbins() + 2;
         auto edges = ax->GetXbins()->GetArray();
         if (edges) {
            binEdges.insert(binEdges.end(), edges, edges + (ncells_d - 1));
            binEdgesIdx[d] = numBinEdges;
            numBinEdges += ncells_d - 1;
         } else {
            binEdgesIdx[d] = -1;
         }

         ncells[d] = ncells_d;
         xlow[d] = ax->GetXmin();
         xHigh[d] = ax->GetXmax();
         numBins *= ncells_d;

         if (getenv("DBG"))
            printf("\tdim %d --- nbins: %d xlow: %f xHigh: %f\n", d, ncells[d], xlow[d], xHigh[d]);
      }

      fSYCLHist = std::make_unique<SYCLHist_t>(maxBulkSize, numBins, ncells, xlow, xHigh, binEdges, binEdgesIdx, fParams);
   }

   SYCLDefFillHelper(){
      fObject = nullptr;
      fSYCLHist = nullptr;
   }

   SYCLDefFillHelper(const std::shared_ptr<HIST> &h, std::size_t maxBulkSize)
   {
      fObject = h.get();
      init_SYCL(fObject, 0, maxBulkSize);
   }

   void InitTask(TTreeReader *, unsigned int) {}

   // Bulk overload
   // TODO: Fill with containers
   template <typename... ValTypes>
   auto Exec(unsigned int slot, const ROOT::RDF::Experimental::REventMask &m, const ValTypes &...x)
   {
   if constexpr (std::conjunction_v<std::is_same<ValTypes, RVecD>...>) {
         Fill(m, std::index_sequence_for<ValTypes...>{}, x...);
      } else {
         // Non-bulk fall back for container types
         for (std::size_t i = 0ul; i < m.Size(); ++i) {
            if (m[i]) {
               Exec((x[i])...);
            }
         }
      }
   }

   // no container arguments
   template <typename... ValTypes, std::enable_if_t<!Disjunction<IsDataContainer<ValTypes>...>::value, int> = 0>
   auto Exec(unsigned int slot, const ValTypes &...x) -> decltype(fObject->Fill(x...), void())
   {
      fObject->Fill(x...);
   }

   // at least one container argument
   template <typename... Xs, std::enable_if_t<Disjunction<IsDataContainer<Xs>...>::value, int> = 0>
   auto Exec(unsigned int slot, const Xs &...xs) -> decltype(fObject->Fill(*MakeBegin(xs)...), void())
   {
      // array of bools keeping track of which inputs are containers
      constexpr std::array<bool, sizeof...(Xs)> isContainer{IsDataContainer<Xs>::value...};

      // index of the first container input
      constexpr std::size_t colidx = FindIdxTrue(isContainer);
      // if this happens, there is a bug in the implementation
      static_assert(colidx < sizeof...(Xs), "Error: index of collection-type argument not found.");

      // get the end iterator to the first container
      auto const xrefend = std::end(GetNthElement<colidx>(xs...));

      // array of container sizes (1 for scalars)
      std::array<std::size_t, sizeof...(xs)> sizes = {{GetSize(xs)...}};

      for (std::size_t i = 0; i < sizeof...(xs); ++i) {
         if (isContainer[i] && sizes[i] != sizes[colidx]) {
            throw std::runtime_error("Cannot fill histogram with values in containers of different sizes.");
         }
      }

      ExecLoop<colidx>(xrefend, MakeBegin(xs)...);
   }

   template <typename T = HIST>
   void Exec(...)
   {
      static_assert(sizeof(T) < 0,
                    "When filling an object with RDataFrame (e.g. via a Fill action) the number or types of the "
                    "columns passed did not match the signature of the object's `Fill` method.");
   }

   void Initialize()
   { /* noop */
   }

   void Finalize()
   {
      double stats[13];

      HIST *h = fObject;
      fSYCLHist->RetrieveResults(h->GetArray(), stats);
      h->PutStats(stats);
      h->SetEntries(fSYCLHist->GetEntries());

      if (getenv("DBG")) {
         printf("SYCL stats:");
         for (int j = 0; j < 13; j++) {
            printf("%f ", stats[j]);
         }
         printf(" %f\n", fObject->GetEntries());

         fObject->GetStats(stats);
         printf("stats:");
         for (int j = 0; j < 13; j++) {
            printf("%f ", stats[j]);
         }

         printf(" %f\n", fObject->GetEntries());
         if (atoi(getenv("DBG")) > 1) {

            printf("histogram:");
            for (int j = 0; j < fObject->GetNcells(); ++j) {
               printf("%f ", fObject->GetArray()[j]);
            }
            printf("\n");
         }
      }
   }

   HIST &PartialUpdate(unsigned int slot)
   {
      (void)slot; // silence unused warnings
      return *fObject;
   }

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      return std::make_unique<RMergeableFill<HIST>>(*fObject);
   }

   // if the fObjects vector type is derived from TObject, return the name of the object
   template <typename T = HIST, std::enable_if_t<std::is_base_of<TObject, T>::value, int> = 0>
   std::string GetActionName()
   {
      return std::string(fObject->IsA()->GetName()) + "\\n" + std::string(fObject->GetName());
   }

   // if fObjects is not derived from TObject, indicate it is some other object
   template <typename T = HIST, std::enable_if_t<!std::is_base_of<TObject, T>::value, int> = 0>
   std::string GetActionName()
   {
      return "Fill SYCL histogram";
   }

   template <typename H = HIST>
   SYCLDefFillHelper MakeNew(void *newResult)
   {
      auto &result = *static_cast<std::shared_ptr<H> *>(newResult);
      ResetIfPossible(result.get());
      UnsetDirectoryIfPossible(result.get());
      return SYCLDefFillHelper(result, fSYCLHist->GetMaxBulkSize());
   }
};

} // namespace Experimental
} // namespace ROOT
#endif
