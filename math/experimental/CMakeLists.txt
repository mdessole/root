ROOT_LINKER_LIBRARY(MathUtil src/MathUtil.cxx TYPE SHARED)
#set_target_properties(MathUtil PROPERTIES LINKER_LANGUAGE CXX)


if (sycl)
  #set_target_properties(GenVector PROPERTIES LINKER_LANGUAGE CXX)
  target_compile_definitions(MathUtil PUBLIC ROOT_MATH_SYCL CL_TARGET_OPENCL_VERSION=300)
  target_compile_options(MathUtil INTERFACE -O2)
  target_include_directories(MathUtil INTERFACE ${OpenSYCL_INCLUDE_DIRS})
  target_link_libraries(MathUtil INTERFACE OpenSYCL::hipSYCL-rt)
  add_sycl_to_target(TARGET MathUtil SOURCES src/MathUtil.cxx)
endif()