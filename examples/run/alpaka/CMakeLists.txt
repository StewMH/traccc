# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-alpaka-functions )

set(EXTRA_LIBS)

set(TRACCC_ALPAKA_EXAMPLE_SOURCES
    seq_example_alpaka.cpp
    seeding_example_alpaka.cpp
)

if(alpaka_ACC_GPU_CUDA_ENABLE)
  enable_language(CUDA)
  include( traccc-compiler-options-cuda )
  set_source_files_properties(${TRACCC_ALPAKA_EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)

  list (APPEND EXTRA_LIBS CUDA::cudart vecmem::cuda)
elseif(alpaka_ACC_GPU_HIP_ENABLE)
  enable_language(HIP)
  find_package( HIPToolkit REQUIRED )
  list(APPEND EXTRA_LIBS HIP::hiprt vecmem::hip)
 endif()

set(LIBRARIES vecmem::core traccc::io traccc::performance
    traccc::core traccc::device_common traccc::alpaka alpaka::alpaka
    traccc::options ${EXTRA_LIBS})
set(DETRAY detray::io detray::detectors)

traccc_add_executable( seq_example_alpaka "seq_example_alpaka.cpp"
    LINK_LIBRARIES ${LIBRARIES} ${DETRAY} )
traccc_add_executable( seeding_example_alpaka "seeding_example_alpaka.cpp"
    LINK_LIBRARIES ${LIBRARIES} )

#
# Set up the "throughput applications".
#
add_library( traccc_examples_alpaka STATIC
   "full_chain_algorithm.hpp"
   "full_chain_algorithm.cpp" )
target_link_libraries( traccc_examples_alpaka
   PUBLIC alpaka::alpaka vecmem::core detray::core detray::detectors
   traccc::core traccc::device_common traccc::alpaka ${EXTRA_LIBS})

traccc_add_executable( throughput_st_alpaka "throughput_st.cpp"
   LINK_LIBRARIES ${LIBRARIES} ${DETRAY} traccc_examples_alpaka)

traccc_add_executable( throughput_mt_alpaka "throughput_mt.cpp"
   LINK_LIBRARIES TBB::tbb ${LIBRARIES} ${DETRAY} traccc_examples_alpaka )

#Can only do this once target is defined, so need another if here
if(alpaka_ACC_GPU_HIP_ENABLE)
  set_target_properties( traccc_seq_example_alpaka PROPERTIES
     POSITION_INDEPENDENT_CODE TRUE )
  set_target_properties( traccc_seeding_example_alpaka PROPERTIES
     POSITION_INDEPENDENT_CODE TRUE )
endif()
