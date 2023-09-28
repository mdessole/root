#ifndef ROOT_SYCLFOREACHSLOTHELPER
#define ROOT_SYCLFOREACHSLOTHELPER

#include <ROOT/RDF/RAction.hxx>
#include "ROOT/RDF/RActionImpl.hxx"

namespace ROOT {
namespace Experimental {

/// The container type for each thread's partial result in an action helper
// We have to avoid to instantiate std::vector<bool> as that makes it impossible to return a reference to one of
// the thread-local results. In addition, a common definition for the type of the container makes it easy to swap
// the type of the underlying container if e.g. we see problems with false sharing of the thread-local results..
template <typename T>
using Results = std::conditional_t<std::is_same<T, bool>::value, std::deque<T>, std::vector<T>>;

template <typename F>
class R__CLING_PTRCHECK(off) SYCLForeachSlotHelper : public RActionImpl<SYCLForeachSlotHelper<F>> {
   F fCallable;

public:
   using ColumnTypes_t = RemoveFirstParameter_t<typename CallableTraits<F>::arg_types>;
   SYCLForeachSlotHelper(F &&f) : fCallable(f) {}
   SYCLForeachSlotHelper(SYCLForeachSlotHelper &&) = default;
   SYCLForeachSlotHelper(const SYCLForeachSlotHelper &) = delete;

   void InitTask(TTreeReader *, unsigned int) {}

   template <typename... Args>
   void Exec(unsigned int slot, Args &&... args)
   {
      // check that the decayed types of Args are the same as the branch types
      static_assert(std::is_same<TypeList<std::decay_t<Args>...>, ColumnTypes_t>::value, "");
      fCallable(slot, std::forward<Args>(args)...);
   }

   void Initialize() { /* noop */}

   void Finalize() { /* noop */}

   std::string GetActionName() { return "ForeachSlot"; }
};

} // namespace Experimental
} // namespace ROOT
#endif // define ROOT_SYCLFOREACHSLOTHELPER