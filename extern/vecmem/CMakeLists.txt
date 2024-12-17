# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# CMake include(s).
cmake_minimum_required( VERSION 3.14 )
include( FetchContent )

# Silence FetchContent warnings with CMake >=3.24.
if( POLICY CMP0135 )
   cmake_policy( SET CMP0135 NEW )
endif()

# Tell the user what's happening.
message( STATUS "Building VecMem as part of the TRACCC project" )

# Declare where to get VecMem from.
set( TRACCC_VECMEM_SOURCE
   "URL;https://github.com/acts-project/vecmem/archive/refs/tags/v1.11.0.tar.gz;URL_MD5;11c51d8212a119ed5862f16e23699382"
   CACHE STRING "Source for VecMem, when built as part of this project" )
mark_as_advanced( TRACCC_VECMEM_SOURCE )
FetchContent_Declare( VecMem ${TRACCC_VECMEM_SOURCE} )

# Options used in the build of VecMem.
set( VECMEM_BUILD_TESTING FALSE CACHE BOOL
   "Turn off the build of the VecMem unit tests" )

# Check the options VecMem is built with, and declare appropriate options in TRACCC.
if( VECMEM_BUILD_CUDA_LIBRARY )
   set ( TRACCC_VECMEM_HAS_CUDA TRUE )
endif()

if ( VECMEM_BUILD_HIP_LIBRARY )
   set ( TRACCC_VECMEM_HAS_HIP TRUE )
endif()

if ( VECMEM_BUILD_SYCL_LIBRARY )
   set ( TRACCC_VECMEM_HAS_SYCL TRUE )
endif()

# Get it into the current directory.
FetchContent_MakeAvailable( VecMem )
