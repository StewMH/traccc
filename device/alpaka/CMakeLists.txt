# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-alpaka-functions )
include( traccc-compiler-options-cpp )

set(PUBLIC_LIBRARIES traccc::core detray::core detray::utils vecmem::core covfie::core)
set(PRIVATE_LIBRARIES traccc::device_common alpaka::alpaka)

if(alpaka_ACC_GPU_CUDA_ENABLE)
  enable_language(CUDA)
  include( traccc-compiler-options-cuda )

  list(APPEND PRIVATE_LIBRARIES CUDA::cudart vecmem::cuda traccc::Thrust)
elseif(alpaka_ACC_GPU_HIP_ENABLE)
  enable_language(HIP)
  find_package( HIPToolkit REQUIRED )
  list(APPEND PRIVATE_LIBRARIES HIP::hiprt vecmem::hip rocThrust)
else()
  list(APPEND PRIVATE_LIBRARIES traccc::Thrust)
endif()

traccc_add_alpaka_library( traccc_alpaka alpaka TYPE SHARED
  # Utility definitions.
  "include/traccc/alpaka/utils/get_device_info.hpp"
  "include/traccc/alpaka/utils/make_prefix_sum_buff.hpp"
  "src/utils/make_prefix_sum_buff.cpp"
  "src/utils/get_device_info.cpp"
  "src/utils/barrier.hpp"
  "src/utils/utils.hpp"
  # Seed finding code.
  "include/traccc/alpaka/seeding/track_params_estimation.hpp"
  "include/traccc/alpaka/seeding/seed_finding.hpp"
  "include/traccc/alpaka/seeding/seeding_algorithm.hpp"
  "include/traccc/alpaka/seeding/spacepoint_binning.hpp"
  # CCL Code.
  "src/seeding/track_params_estimation.cpp"
  "src/seeding/seed_finding.cpp"
  "src/seeding/spacepoint_binning.cpp"
  "src/seeding/seeding_algorithm.cpp"
  # Clusterization
  "include/traccc/alpaka/clusterization/clusterization_algorithm.hpp"
  "src/clusterization/clusterization_algorithm.cpp"
  "include/traccc/alpaka/clusterization/measurement_sorting_algorithm.hpp"
  "src/clusterization/measurement_sorting_algorithm.cpp"
  "include/traccc/alpaka/clusterization/spacepoint_formation_algorithm.hpp"
  "src/clusterization/spacepoint_formation_algorithm.cpp"
  # Seeding code
  "src/seeding/spacepoint_binning.cpp"
  "src/seeding/seed_finding.cpp"
  "src/seeding/seeding_algorithm.cpp"
  "src/seeding/track_params_estimation.cpp"
  # Finding
  "include/traccc/alpaka/finding/finding_algorithm.hpp"
  "src/finding/finding_algorithm.cpp"
  # Fitting
  "include/traccc/alpaka/fitting/fitting_algorithm.hpp"
  "src/fitting/fitting_algorithm.cpp"
)

target_link_libraries(traccc_alpaka PUBLIC ${PUBLIC_LIBRARIES} PRIVATE ${PRIVATE_LIBRARIES})
