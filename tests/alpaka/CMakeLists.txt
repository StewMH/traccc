# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2022 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

set(TRACCC_ALPAKA_TEST_SOURCES
  alpaka_basic.cpp
  test_cca.cpp
)

set (DEVICE_LIBRARIES)

include (traccc-alpaka-functions)
traccc_enable_alpaka_backend_lang(TRACCC_ALPAKA_TEST_SOURCES)
traccc_get_alpaka_vecmem_lib(DEVICE_LIBRARIES)
traccc_get_alpaka_native_lib(DEVICE_LIBRARIES)


traccc_add_test( alpaka
   ${TRACCC_ALPAKA_TEST_SOURCES}
   LINK_LIBRARIES
   GTest::gtest_main
   traccc_tests_common
   alpaka::alpaka
   vecmem::core
   traccc::alpaka
   ${DEVICE_LIBRARIES}
)

#Can only do this once target is defined, so need another if here
if(alpaka_ACC_GPU_HIP_ENABLE)
  set_target_properties( traccc_test_alpaka PROPERTIES
     POSITION_INDEPENDENT_CODE TRUE )
endif()
