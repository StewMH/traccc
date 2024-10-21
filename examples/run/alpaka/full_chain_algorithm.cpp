/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "full_chain_algorithm.hpp"

// System include(s).
#include <iostream>
#include <stdexcept>

namespace traccc::alpaka {

full_chain_algorithm::full_chain_algorithm(
    vecmem::memory_resource& host_mr,
    const unsigned short target_cells_per_partition,
    const seedfinder_config& finder_config,
    const spacepoint_grid_config& grid_config,
    const seedfilter_config& filter_config,
    const finding_algorithm::config_type& finding_config,
    const fitting_algorithm::config_type& fitting_config,
    host_detector_type* detector)
    : m_host_mr(host_mr),
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
      m_device_mr(),
#else
      m_device_mr(host_mr),
#endif
      m_copy(),
      m_cached_device_mr(
          std::make_unique<vecmem::binary_page_memory_resource>(m_device_mr)),
      m_field_vec{0.f, 0.f, finder_config.bFieldInZ},
      m_field(detray::bfield::create_const_field(m_field_vec)),
      m_detector(detector),
      m_target_cells_per_partition(target_cells_per_partition),
      m_clusterization(memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy,
                       m_target_cells_per_partition),
      m_measurement_sorting(m_copy),
      m_spacepoint_formation(memory_resource{*m_cached_device_mr, &m_host_mr},
                             m_copy),
      m_seeding(finder_config, grid_config, filter_config,
                memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy),
      m_track_parameter_estimation(
          memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy),
      m_finding(finding_config,
                memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy),
      m_fitting(fitting_config,
                memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy),
      m_finder_config(finder_config),
      m_grid_config(grid_config),
      m_filter_config(filter_config),
      m_finding_config(finding_config),
      m_fitting_config(fitting_config) {

    traccc::alpaka::get_device_info();

    // Copy the detector to the device.
    if (m_detector != nullptr) {
        m_device_detector = detray::get_buffer(detray::get_data(*m_detector),
                                               m_device_mr, m_copy);
        m_device_detector_view = detray::get_data(m_device_detector);
    }
}

full_chain_algorithm::full_chain_algorithm(const full_chain_algorithm& parent)
    : m_host_mr(parent.m_host_mr),
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
      m_device_mr(),
#else
      m_device_mr(parent.m_host_mr),
#endif
      m_copy(),
      m_cached_device_mr(
          std::make_unique<vecmem::binary_page_memory_resource>(m_device_mr)),
      m_field_vec(parent.m_field_vec),
      m_field(parent.m_field),
      m_detector(parent.m_detector),
      m_target_cells_per_partition(parent.m_target_cells_per_partition),
      m_clusterization(memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy,
                       m_target_cells_per_partition),
      m_measurement_sorting(m_copy),
      m_spacepoint_formation(memory_resource{*m_cached_device_mr, &m_host_mr},
                             m_copy),
      m_seeding(parent.m_finder_config, parent.m_grid_config,
                parent.m_filter_config,
                memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy),
      m_track_parameter_estimation(
          memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy),
      m_finding(parent.m_finding_config,
                memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy),
      m_fitting(parent.m_fitting_config,
                memory_resource{*m_cached_device_mr, &m_host_mr}, m_copy),
      m_finder_config(parent.m_finder_config),
      m_grid_config(parent.m_grid_config),
      m_filter_config(parent.m_filter_config),
      m_finding_config(parent.m_finding_config),
      m_fitting_config(parent.m_fitting_config) {

    // Copy the detector to the device.
    if (m_detector != nullptr) {
        m_device_detector = detray::get_buffer(detray::get_data(*m_detector),
                                               m_device_mr, m_copy);
        m_device_detector_view = detray::get_data(m_device_detector);
    }
}

full_chain_algorithm::~full_chain_algorithm() = default;

full_chain_algorithm::output_type full_chain_algorithm::operator()(
    const cell_collection_types::host& cells,
    const cell_module_collection_types::host& modules) const {

    // Create device copy of input collections
    cell_collection_types::buffer cells_buffer(cells.size(),
                                               *m_cached_device_mr);
    m_copy(vecmem::get_data(cells), cells_buffer)->ignore();
    cell_module_collection_types::buffer modules_buffer(modules.size(),
                                                        *m_cached_device_mr);
    m_copy(vecmem::get_data(modules), modules_buffer)->ignore();

    // Run the clusterization
    const clusterization_algorithm::output_type measurements =
        m_clusterization(cells_buffer, modules_buffer);
    m_measurement_sorting(measurements);

    // Run the seed-finding
    const spacepoint_formation_algorithm::output_type spacepoints =
        m_spacepoint_formation(measurements, modules_buffer);
    const track_params_estimation::output_type track_params =
        m_track_parameter_estimation(spacepoints, m_seeding(spacepoints),
                                     m_field_vec);

    // If we have a Detray detector, run the track finding and fitting.
    if (m_detector != nullptr) {

        // Create the buffer needed by track finding and fitting.
        auto navigation_buffer = detray::create_candidates_buffer(
            *m_detector,
            m_finding_config.navigation_buffer_size_scaler *
                m_copy.get_size(track_params),
            *m_cached_device_mr, &m_host_mr);

        // Run the track finding
        const finding_algorithm::output_type track_candidates =
            m_finding(m_device_detector_view, m_field, navigation_buffer,
                      measurements, track_params);

        // Run the track fitting
        const fitting_algorithm::output_type track_states =
            m_fitting(m_device_detector_view, m_field, navigation_buffer,
                      track_candidates);

        // Copy a limited amount of result data back to the host.
        output_type result{&m_host_mr};
        m_copy(track_states.headers, result)->wait();
        return result;

    }
    // If not, copy the track parameters back to the host, and return a dummy
    // object.
    else {

        // Copy the track parameters back to the host.
        bound_track_parameters_collection_types::host track_params_host(
            &m_host_mr);
        m_copy(track_params, track_params_host)->wait();

        // Return an empty object.
        return {};
    }
}

}  // namespace traccc::alpaka
