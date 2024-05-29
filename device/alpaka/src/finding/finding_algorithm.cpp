/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/alpaka/finding/finding_algorithm.hpp"

#include "../utils/utils.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/device/finding_global_counter.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/device/add_links_for_holes.hpp"
#include "traccc/finding/device/apply_interaction.hpp"
#include "traccc/finding/device/build_tracks.hpp"
#include "traccc/finding/device/count_measurements.hpp"
#include "traccc/finding/device/find_tracks.hpp"
#include "traccc/finding/device/make_barcode_sequence.hpp"
#include "traccc/finding/device/propagate_to_next_surface.hpp"
#include "traccc/finding/device/prune_tracks.hpp"

// detray include(s).
#include "detray/core/detector.hpp"
#include "detray/core/detector_metadata.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_buffer.hpp>
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/containers/vector.hpp>

// Thrust include(s).
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

// System include(s).
#include <vector>

namespace traccc::alpaka {

struct MakeBarcodeSequenceKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        measurement_collection_types::const_view measurements_view,
        vecmem::data::vector_view<detray::geometry::barcode> barcodes_view)
        const {

        int globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];
        device::make_barcode_sequence(globalThreadIdx, measurements_view,
                                      barcodes_view);
    }
};

template <typename detector_t>
struct ApplyInteractionKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const typename detector_t::view_type& det_data,
        const int n_params,
        bound_track_parameters_collection_types::view params_view) const {

        int globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];

        device::apply_interaction<detector_t>(globalThreadIdx, det_data,
                                              n_params, params_view);
    }
};

struct CountMeasurementsKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        const bound_track_parameters_collection_types::const_view& params_view,
        const vecmem::data::vector_view<detray::geometry::barcode>
            barcodes_view,
        const vecmem::data::vector_view<unsigned int> upper_bounds_view,
        const unsigned int n_in_params,
        vecmem::data::vector_view<unsigned int> n_measurements_view,
        vecmem::data::vector_view<unsigned int> ref_meas_idx_view,
        device::finding_global_counter* counter) const {

        int globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];

        device::count_measurements(globalThreadIdx, params_view, barcodes_view,
                                   upper_bounds_view, n_in_params,
                                   n_measurements_view, ref_meas_idx_view,
                                   counter->n_measurements_sum);
    }
};

template <typename detector_t, typename config_t>
struct FindTracksKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const config_t cfg,
        typename detector_t::view_type det_data,
        measurement_collection_types::const_view measurements_view,
        bound_track_parameters_collection_types::const_view in_params_view,
        vecmem::data::vector_view<const unsigned int>
            n_measurements_prefix_sum_view,
        vecmem::data::vector_view<const unsigned int> ref_meas_idx_view,
        vecmem::data::vector_view<const candidate_link> prev_links_view,
        vecmem::data::vector_view<const unsigned int> prev_param_to_link_view,
        const unsigned int step, const unsigned int n_max_candidates,
        bound_track_parameters_collection_types::view out_params_view,
        vecmem::data::vector_view<unsigned int> n_candidates_view,
        vecmem::data::vector_view<candidate_link> links_view,
        device::finding_global_counter* counter) const {

        int globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];

        device::find_tracks<detector_t, config_t>(
            globalThreadIdx, cfg, det_data, measurements_view, in_params_view,
            n_measurements_prefix_sum_view, ref_meas_idx_view, prev_links_view,
            prev_param_to_link_view, step, n_max_candidates, out_params_view,
            n_candidates_view, links_view, counter->n_candidates);
    }
};

struct AddLinksForHolesKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        vecmem::data::vector_view<const unsigned int> n_candidates_view,
        bound_track_parameters_collection_types::const_view in_params_view,
        vecmem::data::vector_view<const candidate_link> prev_links_view,
        vecmem::data::vector_view<const unsigned int> prev_param_to_link_view,
        const unsigned int step, const unsigned int n_max_candidates,
        bound_track_parameters_collection_types::view out_params_view,
        vecmem::data::vector_view<candidate_link> links_view,
        device::finding_global_counter* counter) const {

        int globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];

        device::add_links_for_holes(
            globalThreadIdx, n_candidates_view, in_params_view, prev_links_view,
            prev_param_to_link_view, step, n_max_candidates, out_params_view,
            links_view, counter->n_candidates);
    }
};

template <typename propagator_t, typename bfield_t, typename config_t>
struct PropagateToNextSurfaceKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const config_t cfg,
        typename propagator_t::detector_type::view_type det_data,
        bfield_t field_data,
        vecmem::data::jagged_vector_view<
            typename propagator_t::intersection_type>
            nav_candidates_buffer,
        bound_track_parameters_collection_types::const_view in_params_view,
        vecmem::data::vector_view<const candidate_link> links_view,
        const unsigned int step,
        bound_track_parameters_collection_types::view out_params_view,
        vecmem::data::vector_view<unsigned int> param_to_link_view,
        vecmem::data::vector_view<typename candidate_link::link_index_type>
            tips_view,
        vecmem::data::vector_view<unsigned int> n_tracks_per_seed_view,
        device::finding_global_counter* counter) const {

        int globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];

        device::propagate_to_next_surface<propagator_t, bfield_t, config_t>(
            globalThreadIdx, cfg, det_data, field_data, nav_candidates_buffer,
            in_params_view, links_view, step, counter->n_candidates,
            out_params_view, param_to_link_view, tips_view,
            n_tracks_per_seed_view, counter->n_out_params);
    }
};

template <typename config_t>
struct BuildTracksKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const config_t cfg,
        const measurement_collection_types::const_view& measurements_view,
        const bound_track_parameters_collection_types::const_view& seeds_view,
        const vecmem::data::jagged_vector_view<const candidate_link>&
            links_view,
        const vecmem::data::jagged_vector_view<const unsigned int>&
            param_to_link_view,
        const vecmem::data::vector_view<
            const typename candidate_link::link_index_type>& tips_view,
        track_candidate_container_types::view* track_candidates_view,
        vecmem::data::vector_view<unsigned int> valid_indices_view,
        device::finding_global_counter* counter) const {

        int globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];

        device::build_tracks(globalThreadIdx, cfg, measurements_view,
                             seeds_view, links_view, param_to_link_view,
                             tips_view, *track_candidates_view,
                             valid_indices_view, counter->n_valid_tracks);
    }
};

struct PruneTracksKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        track_candidate_container_types::const_view* track_candidates_view,
        const vecmem::data::vector_view<const unsigned int> valid_indices_view,
        track_candidate_container_types::view* pruned_candidates_view) const {

        int globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];

        device::prune_tracks(globalThreadIdx, *track_candidates_view,
                             valid_indices_view, *pruned_candidates_view);
    }
};

template <typename stepper_t, typename navigator_t>
finding_algorithm<stepper_t, navigator_t>::finding_algorithm(
    const config_type& cfg, const traccc::memory_resource& mr,
    vecmem::copy& copy)
    : m_cfg(cfg), m_mr(mr), m_copy(copy){};

template <typename stepper_t, typename navigator_t>
track_candidate_container_types::buffer
finding_algorithm<stepper_t, navigator_t>::operator()(
    const typename detector_type::view_type& det_view,
    const bfield_type& field_view,
    const vecmem::data::jagged_vector_view<
        typename navigator_t::intersection_type>& navigation_buffer,
    const typename measurement_collection_types::view& measurements,
    const bound_track_parameters_collection_types::buffer& seeds_buffer) const {

    // Setup alpaka
    auto devHost = ::alpaka::getDevByIdx(::alpaka::Platform<Host>{}, 0u);
    auto devAcc = ::alpaka::getDevByIdx(::alpaka::Platform<Acc>{}, 0u);
    auto queue = Queue{devAcc};
    auto const deviceProperties = ::alpaka::getAccDevProps<Acc>(devAcc);
    auto maxThreads = deviceProperties.m_blockThreadExtentMax[0];
    auto threadsPerBlock = maxThreads;

    // Copy setup
    m_copy.setup(seeds_buffer);
    m_copy.setup(navigation_buffer);

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    auto thrustExecPolicy = thrust::device;
#else
    auto thrustExecPolicy = thrust::host;
#endif

    // Get number of seeds
    const auto n_seeds = m_copy.get_size(seeds_buffer);

    // Prepare input parameters with seeds
    bound_track_parameters_collection_types::buffer in_params_buffer(n_seeds,
                                                                     m_mr.main);
    bound_track_parameters_collection_types::device in_params(in_params_buffer);
    bound_track_parameters_collection_types::const_device seeds(seeds_buffer);
    thrust::copy(thrustExecPolicy, seeds.begin(), seeds.end(),
                 in_params.begin());

    // Number of tracks per seed
    vecmem::data::vector_buffer<unsigned int> n_tracks_per_seed_buffer(
        n_seeds, m_mr.main);

    // Create a map for links
    std::map<unsigned int, vecmem::data::vector_buffer<candidate_link>>
        link_map;

    // Create a map for parameter ID to link ID
    std::map<unsigned int, vecmem::data::vector_buffer<unsigned int>>
        param_to_link_map;

    // Create a map for tip links
    std::map<unsigned int, vecmem::data::vector_buffer<
                               typename candidate_link::link_index_type>>
        tips_map;

    // Link size
    std::vector<std::size_t> n_candidates_per_step;
    n_candidates_per_step.reserve(m_cfg.max_track_candidates_per_track);

    std::vector<std::size_t> n_parameters_per_step;
    n_parameters_per_step.reserve(m_cfg.max_track_candidates_per_track);

    // Global counter object in Device memory
    auto bufAcc_counter =
        ::alpaka::allocBuf<device::finding_global_counter, Idx>(devAcc, 1u);

    // Global counter object in Host memory
    auto bufHost_counter =
        ::alpaka::allocBuf<device::finding_global_counter, Idx>(devHost, 1u);
    device::finding_global_counter* const pBufHost_counter =
        ::alpaka::getPtrNative(bufHost_counter);
    ::alpaka::memset(queue, bufHost_counter, 0);
    ::alpaka::memcpy(queue, bufAcc_counter, bufHost_counter);

    /*****************************************************************
     * Measurement Operations
     *****************************************************************/

    measurement_collection_types::const_view::size_type n_measurements =
        m_copy.get_size(measurements);

    // Get copy of barcode uniques
    measurement_collection_types::buffer uniques_buffer{n_measurements,
                                                        m_mr.main};
    measurement_collection_types::device uniques(uniques_buffer);

    measurement* end =
        thrust::unique_copy(thrustExecPolicy, measurements.ptr(),
                            measurements.ptr() + n_measurements,
                            uniques.begin(), measurement_equal_comp());
    unsigned int n_modules = end - uniques.begin();

    // Get upper bounds of unique elements
    vecmem::data::vector_buffer<unsigned int> upper_bounds_buffer{n_modules,
                                                                  m_mr.main};
    vecmem::device_vector<unsigned int> upper_bounds(upper_bounds_buffer);

    thrust::upper_bound(thrustExecPolicy, measurements.ptr(),
                        measurements.ptr() + n_measurements, uniques.begin(),
                        uniques.begin() + n_modules, upper_bounds.begin(),
                        measurement_sort_comp());

    /*****************************************************************
     * Kernel1: Create barcode sequence
     *****************************************************************/

    vecmem::data::vector_buffer<detray::geometry::barcode> barcodes_buffer{
        n_modules, m_mr.main};

    auto blocksPerGrid =
        (barcodes_buffer.size() + threadsPerBlock - 1) / threadsPerBlock;
    auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

    ::alpaka::exec<Acc>(queue, workDiv, MakeBarcodeSequenceKernel{},
                        vecmem::get_data(uniques_buffer),
                        vecmem::get_data(barcodes_buffer));
    ::alpaka::wait(queue);

    for (unsigned int step = 0; step < m_cfg.max_track_candidates_per_track;
         step++) {

        // Previous step
        const unsigned int prev_step = (step == 0 ? 0 : step - 1);

        // Reset the number of tracks per seed
        m_copy.memset(n_tracks_per_seed_buffer, 0)->ignore();

        // Global counter object: Device -> Host
        ::alpaka::memcpy(queue, bufHost_counter, bufAcc_counter);
        ::alpaka::wait(queue);

        // Set the number of input parameters
        const unsigned int n_in_params = (step == 0)
                                             ? in_params_buffer.size()
                                             : pBufHost_counter->n_out_params;

        // Terminate if there is no parameter to process.
        if (n_in_params == 0) {
            break;
        }

        // Reset the global counter
        ::alpaka::memset(queue, bufAcc_counter, 0);

        /*****************************************************************
         * Kernel2: Apply material interaction
         ****************************************************************/

        blocksPerGrid = (n_in_params + threadsPerBlock - 1) / threadsPerBlock;
        workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

        ::alpaka::exec<Acc>(queue, workDiv,
                            ApplyInteractionKernel<detector_type>{}, det_view,
                            n_in_params, vecmem::get_data(in_params_buffer));
        ::alpaka::wait(queue);

        /*****************************************************************
         * Kernel3: Count the number of measurements per parameter
         ****************************************************************/

        vecmem::data::vector_buffer<unsigned int> n_measurements_buffer(
            n_in_params, m_mr.main);
        vecmem::device_vector<unsigned int> n_measurements_device(
            n_measurements_buffer);
        thrust::fill(thrustExecPolicy, n_measurements_device.begin(),
                     n_measurements_device.end(), 0u);

        // Create a buffer for the first measurement index of parameter
        vecmem::data::vector_buffer<unsigned int> ref_meas_idx_buffer(
            n_in_params, m_mr.main);

        blocksPerGrid = (n_in_params + threadsPerBlock - 1) / threadsPerBlock;
        workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

        ::alpaka::exec<Acc>(queue, workDiv, CountMeasurementsKernel{},
                            vecmem::get_data(in_params_buffer),
                            vecmem::get_data(barcodes_buffer),
                            vecmem::get_data(upper_bounds_buffer), n_in_params,
                            vecmem::get_data(n_measurements_buffer),
                            vecmem::get_data(ref_meas_idx_buffer),
                            ::alpaka::getPtrNative(bufAcc_counter));
        ::alpaka::wait(queue);

        // Global counter object: Device -> Host
        ::alpaka::memcpy(queue, bufHost_counter, bufAcc_counter);
        ::alpaka::wait(queue);

        // Create the buffer for the prefix sum of the number of
        // measurements per parameter
        vecmem::data::vector_buffer<unsigned int>
            n_measurements_prefix_sum_buffer(n_in_params, m_mr.main);
        vecmem::device_vector<unsigned int> n_measurements_prefix_sum(
            n_measurements_prefix_sum_buffer);
        thrust::inclusive_scan(thrustExecPolicy, n_measurements_device.begin(),
                               n_measurements_device.end(),
                               n_measurements_prefix_sum.begin());

        /*****************************************************************
         * Kernel4: Find valid tracks
         *****************************************************************/

        // Buffer for kalman-updated parameters spawned by the measurement
        // candidates
        const unsigned int n_max_candidates =
            n_in_params * m_cfg.max_num_branches_per_surface;

        vecmem::data::vector_buffer<unsigned int> n_candidates_buffer{
            n_in_params, m_mr.main};
        vecmem::device_vector<unsigned int> n_candidates_device(
            n_candidates_buffer);
        thrust::fill(thrustExecPolicy, n_candidates_device.begin(),
                     n_candidates_device.end(), 0u);

        bound_track_parameters_collection_types::buffer updated_params_buffer(
            n_in_params * m_cfg.max_num_branches_per_surface, m_mr.main);

        // Create the link map
        link_map[step] = {n_in_params * m_cfg.max_num_branches_per_surface,
                          m_mr.main};
        m_copy.setup(link_map[step]);
        blocksPerGrid =
            (pBufHost_counter->n_measurements_sum +
             threadsPerBlock * m_cfg.n_measurements_per_thread - 1) /
            (threadsPerBlock * m_cfg.n_measurements_per_thread);
        workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

        if (blocksPerGrid > 0) {
            ::alpaka::exec<Acc>(
                queue, workDiv, FindTracksKernel<detector_type, config_type>{},
                m_cfg, det_view, measurements,
                vecmem::get_data(in_params_buffer),
                vecmem::get_data(n_measurements_prefix_sum_buffer),
                vecmem::get_data(ref_meas_idx_buffer),
                vecmem::get_data(link_map[prev_step]),
                vecmem::get_data(param_to_link_map[prev_step]), step,
                n_max_candidates, vecmem::get_data(updated_params_buffer),
                vecmem::get_data(n_candidates_buffer),
                vecmem::get_data(link_map[step]),
                ::alpaka::getPtrNative(bufAcc_counter));
            ::alpaka::wait(queue);
        }

        /*****************************************************************
         * Kernel5: Add a dummy links in case of no branches
         *****************************************************************/

        blocksPerGrid = (n_in_params + threadsPerBlock - 1) / threadsPerBlock;
        workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

        if (blocksPerGrid > 0) {
            ::alpaka::exec<Acc>(queue, workDiv, AddLinksForHolesKernel{},
                                vecmem::get_data(n_candidates_buffer),
                                vecmem::get_data(in_params_buffer),
                                vecmem::get_data(link_map[prev_step]),
                                vecmem::get_data(param_to_link_map[prev_step]),
                                step, n_max_candidates,
                                vecmem::get_data(updated_params_buffer),
                                vecmem::get_data(link_map[step]),
                                ::alpaka::getPtrNative(bufAcc_counter));
            ::alpaka::wait(queue);
        }

        // Global counter object: Device -> Host
        ::alpaka::memcpy(queue, bufHost_counter, bufAcc_counter);
        ::alpaka::wait(queue);

        /*****************************************************************
         * Kernel6: Propagate to the next surface
         *****************************************************************/

        // Buffer for out parameters for the next step
        bound_track_parameters_collection_types::buffer out_params_buffer(
            pBufHost_counter->n_candidates, m_mr.main);

        // Create the param to link ID map
        param_to_link_map[step] = {pBufHost_counter->n_candidates, m_mr.main};
        m_copy.setup(param_to_link_map[step]);

        // Create the tip map
        tips_map[step] = {pBufHost_counter->n_candidates, m_mr.main,
                          vecmem::data::buffer_type::resizable};
        m_copy.setup(tips_map[step]);

        if (pBufHost_counter->n_candidates > 0) {

            blocksPerGrid =
                (pBufHost_counter->n_candidates + threadsPerBlock - 1) /
                threadsPerBlock;
            workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

            ::alpaka::exec<Acc>(
                queue, workDiv,
                PropagateToNextSurfaceKernel<propagator_type, bfield_type,
                                             config_type>{},
                m_cfg, det_view, field_view, navigation_buffer,
                vecmem::get_data(updated_params_buffer),
                vecmem::get_data(link_map[step]), step,
                vecmem::get_data(out_params_buffer),
                vecmem::get_data(param_to_link_map[step]),
                vecmem::get_data(tips_map[step]),
                vecmem::get_data(n_tracks_per_seed_buffer),
                ::alpaka::getPtrNative(bufAcc_counter));
            ::alpaka::wait(queue);
        }

        // Global counter object: Device -> Host
        ::alpaka::memcpy(queue, bufHost_counter, bufAcc_counter);
        ::alpaka::wait(queue);

        // Fill the candidate size vector
        n_candidates_per_step.push_back(pBufHost_counter->n_candidates);
        n_parameters_per_step.push_back(pBufHost_counter->n_out_params);

        // Swap parameter buffer for the next step
        in_params_buffer = std::move(out_params_buffer);
    }

    // Create link buffer
    vecmem::data::jagged_vector_buffer<candidate_link> links_buffer(
        n_candidates_per_step, m_mr.main, m_mr.host);
    m_copy.setup(links_buffer);

    // Copy link map to link buffer
    const auto n_steps = n_candidates_per_step.size();
    for (unsigned int it = 0; it < n_steps; it++) {

        vecmem::device_vector<candidate_link> in(link_map[it]);
        vecmem::device_vector<candidate_link> out(
            *(links_buffer.host_ptr() + it));

        thrust::copy(thrustExecPolicy, in.begin(),
                     in.begin() + n_candidates_per_step[it], out.begin());
    }

    // Create param_to_link
    vecmem::data::jagged_vector_buffer<unsigned int> param_to_link_buffer(
        n_parameters_per_step, m_mr.main, m_mr.host);
    m_copy.setup(param_to_link_buffer);

    // Copy param_to_link map to param_to_link buffer
    for (unsigned int it = 0; it < n_steps; it++) {

        vecmem::device_vector<unsigned int> in(param_to_link_map[it]);
        vecmem::device_vector<unsigned int> out(
            *(param_to_link_buffer.host_ptr() + it));

        thrust::copy(thrustExecPolicy, in.begin(),
                     in.begin() + n_parameters_per_step[it], out.begin());
    }

    // Get the number of tips per step
    std::vector<unsigned int> n_tips_per_step;
    n_tips_per_step.reserve(n_steps);
    for (unsigned int it = 0; it < n_steps; it++) {
        n_tips_per_step.push_back(m_copy.get_size(tips_map[it]));
    }

    // Copy tips_map into the tips vector (D->D)
    unsigned int n_tips_total =
        std::accumulate(n_tips_per_step.begin(), n_tips_per_step.end(), 0);
    vecmem::data::vector_buffer<typename candidate_link::link_index_type>
        tips_buffer{n_tips_total, m_mr.main};
    m_copy.setup(tips_buffer);

    vecmem::device_vector<typename candidate_link::link_index_type> tips(
        tips_buffer);

    unsigned int prefix_sum = 0;

    for (unsigned int it = 0; it < n_steps; it++) {
        vecmem::device_vector<typename candidate_link::link_index_type> in(
            tips_map[it]);

        const unsigned int n_tips = n_tips_per_step[it];
        if (n_tips > 0) {
            thrust::copy(thrustExecPolicy, in.begin(), in.begin() + n_tips,
                         tips.begin() + prefix_sum);
            prefix_sum += n_tips;
        }
    }

    /*****************************************************************
     * Kernel7: Build tracks
     *****************************************************************/

    // Create track candidate buffer
    track_candidate_container_types::buffer track_candidates_buffer{
        {n_tips_total, m_mr.main},
        {std::vector<std::size_t>(n_tips_total,
                                  m_cfg.max_track_candidates_per_track),
         m_mr.main, m_mr.host, vecmem::data::buffer_type::resizable}};
    track_candidate_container_types::view track_candidates(
        track_candidates_buffer);
    track_candidate_container_types::const_view track_candidates_const(
        track_candidates_buffer);

    m_copy.setup(track_candidates_buffer.headers);
    m_copy.setup(track_candidates_buffer.items);

    // Wrap the track candidate buffer in an alpaka buffer
    auto bufAcc_candidates =
        ::alpaka::allocBuf<track_candidate_container_types::view, Idx>(devAcc,
                                                                       1u);
    auto bufAcc_const_candidates =
        ::alpaka::allocBuf<track_candidate_container_types::const_view, Idx>(
            devAcc, 1u);

    auto bufHost_candidates =
        ::alpaka::allocBuf<track_candidate_container_types::view, Idx>(devHost,
                                                                       1u);
    auto bufHost_const_candidates =
        ::alpaka::allocBuf<track_candidate_container_types::const_view, Idx>(
            devHost, 1u);

    track_candidate_container_types::view* pBufHost_candidates =
        ::alpaka::getPtrNative(bufHost_candidates);
    pBufHost_candidates = &track_candidates;
    track_candidate_container_types::const_view* pBufHost_const_candidates =
        ::alpaka::getPtrNative(bufHost_const_candidates);
    pBufHost_const_candidates = &track_candidates_const;

    // Copy the track candidate buffer to the device
    ::alpaka::memcpy(queue, bufAcc_candidates, bufHost_candidates);
    ::alpaka::wait(queue);

    // Create buffer for valid indices
    vecmem::data::vector_buffer<unsigned int> valid_indices_buffer(n_tips_total,
                                                                   m_mr.main);

    // @Note: blocksPerGrid can be zero in case there is no tip. This happens
    // when chi2_max config is set tightly and no tips are found
    if (n_tips_total > 0) {
        blocksPerGrid = (n_tips_total + threadsPerBlock - 1) / threadsPerBlock;
        workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

        // ::alpaka::exec<Acc>(queue, workDiv, TestKernel{},
        // ::alpaka::getPtrNative(bufAcc_candidates));

        ::alpaka::exec<Acc>(queue, workDiv, BuildTracksKernel<config_type>{},
                            m_cfg, measurements, vecmem::get_data(seeds_buffer),
                            vecmem::get_data(links_buffer),
                            vecmem::get_data(param_to_link_buffer),
                            vecmem::get_data(tips_buffer),
                            ::alpaka::getPtrNative(bufAcc_candidates),
                            vecmem::get_data(valid_indices_buffer),
                            ::alpaka::getPtrNative(bufAcc_counter));
        ::alpaka::wait(queue);
    }

    // Global counter object: Device -> Host
    ::alpaka::memcpy(queue, bufHost_counter, bufAcc_counter);
    ::alpaka::wait(queue);

    // Create pruned candidate buffer
    track_candidate_container_types::buffer prune_candidates_buffer{
        {pBufHost_counter->n_valid_tracks, m_mr.main},
        {std::vector<std::size_t>(pBufHost_counter->n_valid_tracks,
                                  m_cfg.max_track_candidates_per_track),
         m_mr.main, m_mr.host, vecmem::data::buffer_type::resizable}};
    track_candidate_container_types::view prune_candidates(
        prune_candidates_buffer);

    m_copy.setup(prune_candidates_buffer.headers);
    m_copy.setup(prune_candidates_buffer.items);

    // Wrap the pruned candidate buffer in an alpaka buffer
    auto bufAcc_prune_candidates =
        ::alpaka::allocBuf<track_candidate_container_types::view, Idx>(devAcc,
                                                                       1u);
    auto bufHost_prune_candidates =
        ::alpaka::allocBuf<track_candidate_container_types::view, Idx>(devHost,
                                                                       1u);
    track_candidate_container_types::view* pBufHost_prune_candidates =
        ::alpaka::getPtrNative(bufHost_prune_candidates);
    pBufHost_prune_candidates = &prune_candidates;

    // Copy the pruned candidate buffer to the device
    ::alpaka::memcpy(queue, bufAcc_prune_candidates, bufHost_prune_candidates);
    ::alpaka::wait(queue);

    if (pBufHost_counter->n_valid_tracks > 0) {
        blocksPerGrid =
            (pBufHost_counter->n_valid_tracks + threadsPerBlock - 1) /
            threadsPerBlock;
        workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

        ::alpaka::exec<Acc>(queue, workDiv, PruneTracksKernel{},
                            ::alpaka::getPtrNative(bufAcc_const_candidates),
                            vecmem::get_data(valid_indices_buffer),
                            ::alpaka::getPtrNative(bufAcc_prune_candidates));
        ::alpaka::wait(queue);
    }

    return prune_candidates_buffer;
}

// Explicit template instantiation
using default_detector_type =
    detray::detector<detray::default_metadata, detray::device_container_types>;
using default_stepper_type =
    detray::rk_stepper<covfie::field<detray::bfield::const_bknd_t>::view_t,
                       traccc::default_algebra, detray::constrained_step<>>;
using default_navigator_type = detray::navigator<const default_detector_type>;
template class finding_algorithm<default_stepper_type, default_navigator_type>;

}  // namespace traccc::alpaka

// Add an Alpaka trait that the measurement_collection_types::const_device type
// is trivially copyable
namespace alpaka {

template <>
struct IsKernelArgumentTriviallyCopyable<
    traccc::measurement_collection_types::const_device> : std::true_type {};

}  // namespace alpaka