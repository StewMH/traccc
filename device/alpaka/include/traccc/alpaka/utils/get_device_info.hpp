/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::alpaka {

// Define an enum to represent the different types of Alpaka accelerators
enum class alpaka_accelerator {
    cpu,
    gpu_cuda,
    gpu_hip,
    cpu_sycl,
    gpu_sycl_intel,
    fpga_sycl_intel
};

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    constexpr alpaka_accelerator acc_type = alpaka_accelerator::gpu_cuda;
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    constexpr alpaka_accelerator acc_type = alpaka_accelerator::gpu_hip;
#elif defined(ALPAKA_ACC_CPU_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_CPU)
    constexpr alpaka_accelerator acc_type = alpaka_accelerator::cpu_sycl;
#elif defined(ALPAKA_ACC_GPU_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU)
    constexpr alpaka_accelerator acc_type = alpaka_accelerator::gpu_sycl_intel;
#elif defined(ALPAKA_ACC_FPGA_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_FPGA)
    constexpr alpaka_accelerator acc_type = alpaka_accelerator::fpga_sycl_intel;
#else
    constexpr alpaka_accelerator acc_type = alpaka_accelerator::cpu;
#endif

/// Function that prints the current device information to the console.
/// Included as part of the traccc::alpaka namespace, to avoid having to include
/// alpaka headers in any users of the library.
void get_device_info();

/// Function to get the current device type.
/// Included as part of the traccc::alpaka namespace, to avoid having to include
/// alpaka headers in any users of the library.
alpaka_accelerator get_device_type();

}  // namespace traccc::alpaka
