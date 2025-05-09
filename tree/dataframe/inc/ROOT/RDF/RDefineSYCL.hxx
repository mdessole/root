// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RDEFINESYCL
#define ROOT_RDF_RDEFINESYCL

#include "ROOT/RDF/ColumnReaderUtils.hxx"
#include "ROOT/RDF/RColumnReaderBase.hxx"
#include "ROOT/RDF/RDefineBase.hxx"
#include "ROOT/RDF/RDefine.hxx"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RDF/REventMask.hxx"
#include "ROOT/RDF/RMaskedEntryRange.hxx"
#include "ROOT/RStringView.hxx"
#include "ROOT/TypeTraits.hxx"
#include "RtypesCore.h"
#include "RDefKernel.h"
#include "RDefSYCL.h"

#include <array>
#include <deque>
#include <limits>
#include <type_traits>
#include <utility> // std::index_sequence
#include <vector>

class TTreeReader;

namespace ROOT {
namespace Experimental {

using namespace ROOT::TypeTraits;
using namespace ROOT::Detail::RDF;
using namespace ROOT;

template <typename F, typename SYCLDef, typename ExtraArgsTag = ExtraArgsForDefine::None> 
class R__CLING_PTRCHECK(off) RDefineSYCL final : public RDefineBase {
   // shortcuts
   using NoneTag = ExtraArgsForDefine::None;
   using SlotTag = ExtraArgsForDefine::Slot;
   using SlotAndEntryTag = ExtraArgsForDefine::SlotAndEntry;
   // other types
   using FunParamTypes_t = typename CallableTraits<F>::arg_types;
   constexpr static auto kUsingBulkAPI =
      std::is_same<TakeFirstParameter_t<FunParamTypes_t>, ROOT::RDF::Experimental::REventMask>::value;

public:
   using ColumnTypes_t = ExtractColumnTypes_t<kUsingBulkAPI, ExtraArgsTag, FunParamTypes_t>;
   using TypeInd_t = std::make_index_sequence<ColumnTypes_t::list_size>;
   using RetType_t = std::conditional_t<kUsingBulkAPI, RDFInternal::ReturnTypeForBulkExpr_t<FunParamTypes_t>,
                                        // ret_type is simply the return type of the expression
                                        typename CallableTraits<F>::ret_type>;
   using SYCLDef_t = SYCLDef; // ROOT::Experimental::IdentityKernel; //

private:
   F fExpression;
   std::vector<ROOT::RVec<RetType_t>> fLastResults;

   /// Column readers per slot and per input column
   std::vector<std::array<RColumnReaderBase *, ColumnTypes_t::list_size>> fValueReaders;

   /// Arrays of type-erased raw pointers to the beginning of bulks of column values, one per slot.
   std::vector<std::array<void *, ColumnTypes_t::list_size>> fValuePtrs;

   /// Define objects corresponding to systematic variations other than nominal for this defined column.
   /// The map key is the full variation name, e.g. "pt:up".
   std::unordered_map<std::string, std::unique_ptr<RDefineBase>> fVariedDefines;

   std::unique_ptr<SYCLDef_t> fSYCLDef;
   const unsigned int nInput = fSYCLDef->GetnInput();
   std::vector<Double_t> fParams{}; // double or RetType_t?
   double fGPUtime = 0;

   template <typename... ColTypes, std::size_t... S>
   auto EvalExpr(unsigned int slot, std::size_t idx, Long64_t /*entry*/, TypeList<ColTypes...>,
                 std::index_sequence<S...>, NoneTag)
   {
      // counting on copy elision
      return fExpression(*(static_cast<ColTypes *>(fValuePtrs[slot][S]) + idx)...);
      // avoid unused variable warnings (gcc 12)
      (void)slot;
      (void)idx;
   }

   template <typename... ColTypes, std::size_t... S>
   auto EvalExpr(unsigned int slot, std::size_t idx, Long64_t /*entry*/, TypeList<ColTypes...>,
                 std::index_sequence<S...>, SlotTag)
   {
      // counting on copy elision
      return fExpression(slot, *(static_cast<ColTypes *>(fValuePtrs[slot][S]) + idx)...);
      (void)idx; // avoid unused variable warnings (gcc 12)
   }

   template <typename... ColTypes, std::size_t... S>
   auto EvalExpr(unsigned int slot, std::size_t idx, Long64_t entry, TypeList<ColTypes...>, std::index_sequence<S...>,
                 SlotAndEntryTag)
   {
      // counting on copy elision
      return fExpression(slot, entry, *(static_cast<ColTypes *>(fValuePtrs[slot][S]) + idx)...);
      (void)idx; // avoid unused variable warnings (gcc 12)
   }

   // non-bulk overload, calls EvalExpr in a loop
   template <typename FirstInputCol>
   void UpdateHelper(FirstInputCol *, unsigned int slot, const ROOT::Internal::RDF::RMaskedEntryRange &requestedMask,
                     std::size_t bulkSize, std::size_t firstNewIdx)
   {
      auto &results = fLastResults[slot * RDFInternal::CacheLineStep<RetType_t>()];
      auto &valueMask = fMask[slot * RDFInternal::CacheLineStep<RDFInternal::RMaskedEntryRange>()];
      const auto rdfentry_start = fLoopManager->GetUniqueRDFEntry(slot);
      if (firstNewIdx == std::numeric_limits<std::size_t>::max()) {
         for (std::size_t i = 0u; i < bulkSize; ++i) {
            if (requestedMask[i])
               results[i] = EvalExpr(slot, i, rdfentry_start + i, ColumnTypes_t{}, TypeInd_t{}, ExtraArgsTag{});
         }
      } else { // not a new bulk and requestedMask isn't contained in valueMask (we checked before)
         // we know the entry at firstNewIdx has to be calculated
         results[firstNewIdx] =
            EvalExpr(slot, firstNewIdx, rdfentry_start + firstNewIdx, ColumnTypes_t{}, TypeInd_t{}, ExtraArgsTag{});
         ++firstNewIdx;

         // for the others we need to check the event masks masks
         for (std::size_t i = firstNewIdx; i < bulkSize; ++i) {
            if (requestedMask[i] && !valueMask[i]) // we don't have a value for this entry yet
               results[i] = EvalExpr(slot, i, rdfentry_start + i, ColumnTypes_t{}, TypeInd_t{}, ExtraArgsTag{});
         }
      }

      valueMask.Union(requestedMask);
   }

   // no container arguments for GPU offloading
   template <typename... ColTypes, std::size_t... S,  std::enable_if_t<!Disjunction<IsDataContainer<ColTypes>...>::value, int> = 0>
   auto EvalBulkExpr(const ROOT::RDF::Experimental::REventMask &m, unsigned int slot, TypeList<ColTypes...>,
                     std::index_sequence<S...>)
   {
      auto &results = fLastResults[slot * RDFInternal::CacheLineStep<RetType_t>()];
      const auto bulkSize = m.Size();

      RVecD coords;
      coords.reserve(bulkSize * nInput);

      auto maskedInsert = [&](auto arr, auto &out) {
         for (std::size_t i = 0ul; i < m.Size(); ++i) {
            if (m[i])
               out.emplace_back(static_cast<double>(arr[i]));
         }
      };

      // Converts the arrays in the parameter pack fValuePtrs[slot][S] for column of coordinates,
      // RVec(x1, x2, ...), RVec(y1, y2, ...), RVec(z1, z2, ...), ... into a single RVec in the form of
      // RVec(x1, x2, ... y1, y2, ... z1, x2, ....)
      (maskedInsert( static_cast<ColTypes *>(fValuePtrs[slot][S]),  coords), ...);

      // fExpression(m, results, ROOT::RVec<ColTypes>(static_cast<ColTypes *>(fValuePtrs[slot][S]), bulkSize)...);
#ifdef TIMING
      auto start = Clock::now();
#endif
      fSYCLDef->EvalBulkExpr(coords);
      fSYCLDef->RetrieveResults(results.data(), bulkSize); // or bulkSize/nInput??
#ifdef TIMING
      auto end = Clock::now();
      fGPUtime += std::chrono::duration_cast<fsecs>(end - start).count();
#endif
      (void)bulkSize; // avoid unused variable warnings when the variadic parameter pack is empty
   }

   // at least one container argument
   // standard execution
   template <typename... ColTypes, std::size_t... S, std::enable_if_t<Disjunction<IsDataContainer<ColTypes>...>::value, int> = 0>
   auto EvalBulkExpr(const ROOT::RDF::Experimental::REventMask &m, unsigned int slot, TypeList<ColTypes...>,
                     std::index_sequence<S...>)
   {
      auto &results = fLastResults[slot * RDFInternal::CacheLineStep<RetType_t>()];
      const auto bulkSize = m.Size();
      fExpression(m, results, ROOT::RVec<ColTypes>(static_cast<ColTypes *>(fValuePtrs[slot][S]), bulkSize)...);
      (void)bulkSize; // avoid unused variable warnings when the variadic parameter pack is empty
   }

   // bulk overload (first input column is a REventMask)
   void UpdateHelper(ROOT::RDF::Experimental::REventMask *, unsigned int slot,
                     const ROOT::Internal::RDF::RMaskedEntryRange &requestedMask, std::size_t bulkSize,
                     std::size_t /*firstNewIdx*/)
   {
      const auto eventMask = ROOT::RDF::Experimental::REventMask(requestedMask, bulkSize);
      EvalBulkExpr(eventMask, slot, ColumnTypes_t{}, TypeInd_t{});
      auto &valueMask = fMask[slot * RDFInternal::CacheLineStep<RDFInternal::RMaskedEntryRange>()];
      valueMask = requestedMask;
   }

public:
   // Initialize fSYCLDef
   inline void init_SYCL(int i, size_t maxBulkSize)
   {
      if (getenv("DBG")){
         printf("Init SYCL DefHist %d\n", i);
#ifdef TIMING
         printf("\tTiming ON");
#endif
         printf("\tmaxBulkSize: %ld\n", maxBulkSize);
      }

#ifdef TIMING
      auto start = Clock::now();
#endif
      fSYCLDef = std::make_unique<SYCLDef_t>(maxBulkSize, fParams);
#ifdef TIMING
      auto end = Clock::now();
      fGPUtime += std::chrono::duration_cast<fsecs>(end - start).count();
#endif
   }

   RDefineSYCL(std::string_view name, std::string_view type, F expression, const ROOT::RDF::ColumnNames_t &columns,
           const RDFInternal::RColumnRegister &colRegister, RLoopManager &lm,
           const std::string &variationName = "nominal")
      : RDefineBase(name, type, colRegister, lm, columns, variationName),
        fExpression(std::move(expression)),
        fLastResults(lm.GetNSlots() * RDFInternal::CacheLineStep<RetType_t>()),
        fValueReaders(lm.GetNSlots()),
        fValuePtrs(lm.GetNSlots())
   {
      for (auto &r : fLastResults)
         r.resize(fLoopManager->GetMaxEventsPerBulk());
      fLoopManager->Register(this);
      init_SYCL(0, lm.GetMaxEventsPerBulk());
   }

   RDefineSYCL(const RDefineSYCL &) = delete;
   RDefineSYCL &operator=(const RDefineSYCL &) = delete;
   ~RDefineSYCL() { fLoopManager->Deregister(this); }

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      RDFInternal::RColumnReadersInfo info{fColumnNames, fColRegister, fIsDefine.data(), *fLoopManager};
      fValueReaders[slot] = RDFInternal::GetColumnReaders(slot, r, ColumnTypes_t{}, info, fVariation);
      fMask[slot * RDFInternal::CacheLineStep<RDFInternal::RMaskedEntryRange>()].SetFirstEntry(-1ll);
   }

   /// Return the (type-erased) address of the Define'd value for the given processing slot.
   void *GetValuePtr(unsigned int slot) final
   {
      return static_cast<void *>(fLastResults[slot * RDFInternal::CacheLineStep<RetType_t>()].data());
   }

   /// Update the value at the address returned by GetValuePtr with the content corresponding to the given entry
   void Update(unsigned int slot, const ROOT::Internal::RDF::RMaskedEntryRange &requestedMask, std::size_t bulkSize) final
   {
      auto &valueMask = fMask[slot * RDFInternal::CacheLineStep<RDFInternal::RMaskedEntryRange>()];
      // Index of the first entry in the bulk for which we do not already have a value
      std::size_t firstNewIdx;
      if (valueMask.FirstEntry() != requestedMask.FirstEntry()) { // new bulk
         // if it turns out that we do these two operations together very often, maybe it's worth having a ad-hoc method
         valueMask.SetAll(false);
         valueMask.SetFirstEntry(requestedMask.FirstEntry());
         firstNewIdx = std::numeric_limits<std::size_t>::max(); // we use this to mean "no entries are loaded yet"
      } else if (auto firstNonContainedIndex = valueMask.Contains(requestedMask, bulkSize);
                 firstNonContainedIndex == std::numeric_limits<std::size_t>::max()) {
         // this is a common occurrence: it happens when the same Define is used multiple times downstream of the same
         // Filters -- nothing to do.
         return;
      } else {
         firstNewIdx = firstNonContainedIndex;
      }

      std::transform(fValueReaders[slot].begin(), fValueReaders[slot].end(), fValuePtrs[slot].begin(),
                     [&requestedMask, &bulkSize](auto *v) { return v->Load(requestedMask, bulkSize); });

      // dispatch either to the bulk version or to the event-by-event version based on the type of the first input col
      using FirstArg_t = TakeFirstParameter_t<FunParamTypes_t>;
      UpdateHelper((FirstArg_t *)nullptr, slot, requestedMask, bulkSize, firstNewIdx);
   }

   void Update(unsigned int /*slot*/, const ROOT::RDF::RSampleInfo & /*id*/) final {}

   const std::type_info &GetTypeId() const final { return typeid(RetType_t); }

   /// Clean-up operations to be performed at the end of a task.
   void FinalizeSlot(unsigned int slot) final
   {
      fValueReaders[slot].fill(nullptr);

      for (auto &e : fVariedDefines)
         e.second->FinalizeSlot(slot);
   }

   /// Create clones of this Define that work with values in varied "universes".
   void MakeVariations(const std::vector<std::string> &variations) final
   {
      for (const auto &variation : variations) {
         if (std::find(fVariationDeps.begin(), fVariationDeps.end(), variation) == fVariationDeps.end()) {
            // this Defined quantity does not depend on this variation, so no need to create a varied RDefine
            continue;
         }
         if (fVariedDefines.find(variation) != fVariedDefines.end())
            continue; // we already have this variation stored

         // the varied defines get a copy of the callable object.
         // TODO document this
         auto variedDefine = std::unique_ptr<RDefineBase>(
            new RDefine(fName, fType, fExpression, fColumnNames, fColRegister, *fLoopManager, variation));
         // TODO switch to fVariedDefines.insert({variationName, std::move(variedDefine)}) when we drop gcc 5
         fVariedDefines[variation] = std::move(variedDefine);
      }
   }

   /// Return a clone of this Define that works with values in the variationName "universe".
   RDefineBase &GetVariedDefine(const std::string &variationName) final
   {
      auto it = fVariedDefines.find(variationName);
      if (it == fVariedDefines.end()) {
         // We don't have a varied RDefine for this variation.
         // This means we don't depend on it and we can return ourselves, i.e. the RDefine for the nominal universe.
         assert(std::find(fVariationDeps.begin(), fVariationDeps.end(), variationName) == fVariationDeps.end());
         return *this;
      }

      return *(it->second);
   }

   std::size_t GetTypeSize() const final { return sizeof(RetType_t); }

   bool IsDefinePerSample() const final { return false; }
};

} // namespace RDF
} // namespace Experimental

#endif // ROOT_RDF_RDEFINESYCL
