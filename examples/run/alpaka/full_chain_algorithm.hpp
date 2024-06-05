/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/alpaka/clusterization/clusterization_algorithm.hpp"
#include "traccc/alpaka/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/alpaka/clusterization/spacepoint_formation_algorithm.hpp"
#include "traccc/alpaka/finding/finding_algorithm.hpp"
#include "traccc/alpaka/fitting/fitting_algorithm.hpp"
#include "traccc/alpaka/seeding/seeding_algorithm.hpp"
#include "traccc/alpaka/seeding/track_params_estimation.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/utils/algorithm.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>
#endif

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
#include <vecmem/memory/hip/device_memory_resource.hpp>
#include <vecmem/utils/hip/copy.hpp>
#endif

#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/binary_page_memory_resource.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <memory>

namespace traccc::alpaka {

/// Algorithm performing the full chain of track reconstruction
///
/// At least as much as is implemented in the project at any given moment.
///
class full_chain_algorithm
    : public algorithm<vecmem::vector<fitting_result<default_algebra>>(
          const cell_collection_types::host&,
          const cell_module_collection_types::host&)> {

    public:
    /// @name Type declaration(s)
    /// @{

    /// (Host) Detector type used during track finding and fitting
    using host_detector_type = detray::detector<detray::default_metadata,
                                                detray::host_container_types>;
    /// (Device) Detector type used during track finding and fitting
    using device_detector_type =
        detray::detector<detray::default_metadata,
                         detray::device_container_types>;

    /// Stepper type used by the track finding and fitting algorithms
    using stepper_type =
        detray::rk_stepper<detray::bfield::const_field_t::view_t,
                           device_detector_type::algebra_type,
                           detray::constrained_step<>>;
    /// Navigator type used by the track finding and fitting algorithms
    using navigator_type = detray::navigator<const device_detector_type>;

    /// Track finding algorithm type
    using finding_algorithm =
        traccc::alpaka::finding_algorithm<stepper_type, navigator_type>;
    /// Track fitting algorithm type
    using fitting_algorithm = traccc::alpaka::fitting_algorithm<
        traccc::kalman_fitter<stepper_type, navigator_type>>;

    /// @}

    /// Algorithm constructor
    ///
    /// @param mr The memory resource to use for the intermediate and result
    ///           objects
    /// @param target_cells_per_partition The average number of cells in each
    /// partition.
    ///
    full_chain_algorithm(vecmem::memory_resource& host_mr,
                         const unsigned short target_cells_per_partiton,
                         const seedfinder_config& finder_config,
                         const spacepoint_grid_config& grid_config,
                         const seedfilter_config& filter_config,
                         const finding_algorithm::config_type& finding_config,
                         const fitting_algorithm::config_type& fitting_config,
                         host_detector_type* detector);

    /// Copy constructor
    ///
    /// An explicit copy constructor is necessary because in the MT tests
    /// we do want to copy such objects, but a default copy-constructor can
    /// not be generated for them.
    ///
    /// @param parent The parent algorithm chain to copy
    ///
    full_chain_algorithm(const full_chain_algorithm& parent);

    /// Algorithm destructor
    ~full_chain_algorithm();

    /// Reconstruct track parameters in the entire detector
    ///
    /// @param cells The cells for every detector module in the event
    /// @return The track parameters reconstructed
    ///
    output_type operator()(
        const cell_collection_types::host& cells,
        const cell_module_collection_types::host& modules) const override;

    private:
    /// Host memory resource
    vecmem::memory_resource& m_host_mr;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    /// Device memory resource
    vecmem::cuda::device_memory_resource m_device_mr;
    /// Memory copy object
    vecmem::cuda::copy m_copy;
#elif ALPAKA_ACC_GPU_HIP_ENABLED
    /// Device memory resource
    vecmem::hip::device_memory_resource m_device_mr;
    /// Memory copy object
    vecmem::hip::copy m_copy;
#else
    /// Device memory resource
    vecmem::memory_resource& m_device_mr;
    /// Memory copy object
    vecmem::copy m_copy;
#endif
    /// Device caching memory resource
    std::unique_ptr<vecmem::binary_page_memory_resource> m_cached_device_mr;

    /// Constant B field for the (seed) track parameter estimation
    traccc::vector3 m_field_vec;
    /// Constant B field for the track finding and fitting
    detray::bfield::const_field_t m_field;

    /// Host detector
    host_detector_type* m_detector;
    /// Buffer holding the detector's payload on the device
    host_detector_type::buffer_type m_device_detector;
    /// View of the detector's payload on the device
    host_detector_type::view_type m_device_detector_view;

    /// @name Sub-algorithms used by this full-chain algorithm
    /// @{

    /// The average number of cells in each partition.
    /// Adapt to different GPUs' capabilities.
    unsigned short m_target_cells_per_partition;
    /// Clusterization algorithm
    clusterization_algorithm m_clusterization;
    /// Measurement sorting algorithm
    measurement_sorting_algorithm m_measurement_sorting;
    /// Spacepoint formation algorithm
    spacepoint_formation_algorithm m_spacepoint_formation;
    /// Seeding algorithm
    seeding_algorithm m_seeding;
    /// Track parameter estimation algorithm
    track_params_estimation m_track_parameter_estimation;

    /// Track finding algorithm
    finding_algorithm m_finding;
    /// Track fitting algorithm
    fitting_algorithm m_fitting;

    /// @}

    /// @name Algorithm configurations
    /// @{

    /// Configuration for the seed finding
    seedfinder_config m_finder_config;
    /// Configuration for the spacepoint grid formation
    spacepoint_grid_config m_grid_config;
    /// Configuration for the seed filtering
    seedfilter_config m_filter_config;

    /// Configuration for the track finding
    finding_algorithm::config_type m_finding_config;
    /// Configuration for the track fitting
    fitting_algorithm::config_type m_fitting_config;

    /// @}

};  // class full_chain_algorithm

}  // namespace traccc::alpaka
