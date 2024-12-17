/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "get_device_info.hpp"
#include "traccc/alpaka/utils/get_device_info.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

/// This abstraction means is included as part of the traccc::alpaka namespace,
/// to avoid having to include alpaka headers in any users of the library.
namespace traccc::alpaka::vecmem_resource {

// For all CPU accelerators (except SYCL), just use host
// Otherwise, template on the alpaka accelerator enum
template <alpaka_accelerator T>
struct host_device_types {
    using device_memory_resource = ::vecmem::host_memory_resource;
    using host_memory_resource = ::vecmem::host_memory_resource;
    using managed_memory_resource = ::vecmem::host_memory_resource;
    using device_copy = ::vecmem::copy;
};

#ifdef TRACCC_VECMEM_HAS_CUDA
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

template <>
struct host_device_types<alpaka_accelerator::gpu_cuda> {
    using device_memory_resource = ::vecmem::cuda::device_memory_resource;
    using host_memory_resource = ::vecmem::cuda::host_memory_resource;
    using managed_memory_resource = ::vecmem::cuda::managed_memory_resource;
    using device_copy = ::vecmem::cuda::copy;
};
#endif

#ifdef TRACCC_VECMEM_HAS_HIP
#include <vecmem/memory/hip/device_memory_resource.hpp>
#include <vecmem/memory/hip/host_memory_resource.hpp>
#include <vecmem/memory/hip/managed_memory_resource.hpp>
#include <vecmem/utils/hip/copy.hpp>

template <>
struct host_device_types<alpaka_accelerator::gpu_hip> {
    using device_memory_resource = ::vecmem::hip::device_memory_resource;
    using host_memory_resource = ::vecmem::hip::host_memory_resource;
    using managed_memory_resource = ::vecmem::hip::managed_memory_resource;
    using device_copy = ::vecmem::hip::copy;
};
#endif

#ifdef TRACCC_VECMEM_HAS_SYCL
#include <vecmem/memory/sycl/device_memory_resource.hpp>
#include <vecmem/memory/sycl/host_memory_resource.hpp>
#include <vecmem/memory/sycl/shared_memory_resource.hpp>
#include <vecmem/utils/sycl/copy.hpp>

template <>
struct host_device_types<alpaka_accelerator::cpu_sycl> {
    using device_memory_resource = ::vecmem::sycl::device_memory_resource;
    using host_memory_resource = ::vecmem::sycl::host_memory_resource;
    using managed_memory_resource = ::vecmem::sycl::shared_memory_resource;
    using device_copy = ::vecmem::sycl::copy;
};
template <>
struct host_device_types<alpaka_accelerator::fpga_sycl_intel> {
    using device_memory_resource = ::vecmem::sycl::device_memory_resource;
    using host_memory_resource = ::vecmem::sycl::host_memory_resource;
    using managed_memory_resource = ::vecmem::sycl::shared_memory_resource;
    using device_copy = ::vecmem::sycl::copy;
};
template <>
struct host_device_types<alpaka_accelerator::gpu_sycl_intel> {
    using device_memory_resource = ::vecmem::sycl::device_memory_resource;
    using host_memory_resource = ::vecmem::sycl::host_memory_resource;
    using managed_memory_resource = ::vecmem::sycl::shared_memory_resource;
    using device_copy = ::vecmem::sycl::copy;
};
#endif

using device_memory_resource = typename host_device_types<acc_type>::device_memory_resource;
using host_memory_resource = typename host_device_types<acc_type>::host_memory_resource;
using managed_memory_resource = typename host_device_types<acc_type>::managed_memory_resource;
using device_copy = typename host_device_types<acc_type>::device_copy;

}  // namespace traccc::alpaka::vecmem_resource
