/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "utils.hpp"

// Project include(s).
#include "traccc/alpaka/utils/get_device_info.hpp"

// System include(s).
#include <iostream>

namespace traccc::alpaka {

void get_device_info() {
    int device = 0;
    auto devAcc = ::alpaka::getDevByIdx(::alpaka::Platform<Acc>{}, 0u);
    std::cout << "Using Alpaka device: " << ::alpaka::getName(devAcc)
              << " [id: " << device << "] " << std::endl;
}

alpaka_accelerator get_device_type() {

    if constexpr (::alpaka::accMatchesTags<Acc, ::alpaka::TagGpuCudaRt>) {
        return alpaka_accelerator::gpu_cuda;
    } else if constexpr (::alpaka::accMatchesTags<Acc, ::alpaka::TagGpuHipRt>) {
        return alpaka_accelerator::gpu_hip;
    } else if constexpr (::alpaka::accMatchesTags<Acc, ::alpaka::TagCpuSycl>) {
        return alpaka_accelerator::cpu_sycl;
    } else if constexpr (::alpaka::accMatchesTags<Acc, ::alpaka::TagFpgaSyclIntel>) {
        return alpaka_accelerator::fpga_sycl_intel;
    } else if constexpr (::alpaka::accMatchesTags<Acc, ::alpaka::TagGpuSyclIntel>) {
        return alpaka_accelerator::gpu_sycl_intel;
    } else {
        return alpaka_accelerator::cpu;
    }
}

}  // namespace traccc::alpaka
