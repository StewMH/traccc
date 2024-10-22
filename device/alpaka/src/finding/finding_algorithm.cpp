/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/alpaka/finding/finding_algorithm.hpp"

#include "../utils/barrier.hpp"
#include "../utils/utils.hpp"
#include "traccc/alpaka/utils/thread_id.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/device/sort_key.hpp"
#include "traccc/finding/candidate_link.hpp"
#include "traccc/finding/device/apply_interaction.hpp"
#include "traccc/finding/device/build_tracks.hpp"
#include "traccc/finding/device/fill_sort_keys.hpp"
#include "traccc/finding/device/find_tracks.hpp"
#include "traccc/finding/device/make_barcode_sequence.hpp"
#include "traccc/finding/device/propagate_to_next_surface.hpp"
#include "traccc/finding/device/prune_tracks.hpp"
#include "traccc/utils/projections.hpp"

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
#include <cassert>
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
        const finding_config& cfg, const int n_params,
        bound_track_parameters_collection_types::view params_view,
        vecmem::data::vector_view<const unsigned int> params_liveness_view)
        const {

        int globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];

        device::apply_interaction<detector_t>(globalThreadIdx, cfg, det_data,
                                              n_params, params_view,
                                              params_liveness_view);
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
        vecmem::data::vector_view<const unsigned int> in_params_liveness_view,
        const unsigned int n_in_params,
        vecmem::data::vector_view<const detray::geometry::barcode>
            barcodes_view,
        vecmem::data::vector_view<const unsigned int> upper_bounds_view,
        vecmem::data::vector_view<const candidate_link> prev_links_view,
        const unsigned int step, const unsigned int n_max_candidates,
        bound_track_parameters_collection_types::view out_params_view,
        vecmem::data::vector_view<unsigned int> out_params_liveness_view,
        vecmem::data::vector_view<candidate_link> links_view,
        unsigned int* n_candidates) {

        auto& shared_candidates_size =
            ::alpaka::declareSharedVar<unsigned int, __COUNTER__>(acc);
        unsigned int* const s = ::alpaka::getDynSharedMem<unsigned int>(acc);
        unsigned int* shared_num_candidates = s;

        alpaka::barrier barrier(acc);
        alpaka::thread_id1 thread_id(acc);

        int blockDimX = thread_id.getBlockDimX();
        std::pair<unsigned int, unsigned int>* shared_candidates =
            reinterpret_cast<std::pair<unsigned int, unsigned int>*>(
                &shared_num_candidates[blockDimX]);

        device::find_tracks<alpaka::thread_id1, alpaka::barrier, detector_t,
                            config_t>(
            thread_id, barrier, cfg, det_data, measurements_view,
            in_params_view, in_params_liveness_view, n_in_params, barcodes_view,
            upper_bounds_view, prev_links_view, step, n_max_candidates,
            out_params_view, out_params_liveness_view, links_view,
            *n_candidates, shared_num_candidates, shared_candidates,
            shared_candidates_size);
    }
};

struct FillSortKeysKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        bound_track_parameters_collection_types::const_view params_view,
        vecmem::data::vector_view<device::sort_key> keys_view,
        vecmem::data::vector_view<unsigned int> ids_view) {

        int globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];

        device::fill_sort_keys(globalThreadIdx, params_view, keys_view,
                               ids_view);
    }
};

template <typename propagator_t, typename bfield_t, typename config_t>
struct PropagateToNextSurfaceKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const config_t cfg,
        typename propagator_t::detector_type::view_type det_data,
        bfield_t field_data,
        bound_track_parameters_collection_types::view params_view,
        vecmem::data::vector_view<unsigned int> params_liveness_view,
        vecmem::data::vector_view<const unsigned int> param_ids_view,
        vecmem::data::vector_view<const candidate_link> links_view,
        const unsigned int step, const unsigned int n_candidates,
        vecmem::data::vector_view<typename candidate_link::link_index_type>
            tips_view,
        vecmem::data::vector_view<unsigned int> n_tracks_per_seed_view) {

        int globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];

        device::propagate_to_next_surface<propagator_t, bfield_t, config_t>(
            globalThreadIdx, cfg, det_data, field_data, params_view,
            params_liveness_view, param_ids_view, links_view, step,
            n_candidates, tips_view, n_tracks_per_seed_view);
    }
};

template <typename config_t>
struct BuildTracksKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, const config_t cfg,
        measurement_collection_types::const_view measurements_view,
        bound_track_parameters_collection_types::const_view seeds_view,
        vecmem::data::jagged_vector_view<const candidate_link> links_view,
        vecmem::data::vector_view<
            const typename candidate_link::link_index_type>
            tips_view,
        track_candidate_container_types::view track_candidates_view,
        vecmem::data::vector_view<unsigned int> valid_indices_view,
        unsigned int* n_valid_tracks) {

        int globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];

        device::build_tracks(globalThreadIdx, cfg, measurements_view,
                             seeds_view, links_view, tips_view,
                             track_candidates_view, valid_indices_view,
                             *n_valid_tracks);
    }
};

struct PruneTracksKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        track_candidate_container_types::const_view track_candidates_view,
        vecmem::data::vector_view<const unsigned int> valid_indices_view,
        track_candidate_container_types::view prune_candidates_view) const {

        int globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];

        device::prune_tracks(globalThreadIdx, track_candidates_view,
                             valid_indices_view, prune_candidates_view);
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
    const typename measurement_collection_types::view& measurements,
    const bound_track_parameters_collection_types::buffer& seeds_buffer) const {

    // Setup alpaka
    auto devHost = ::alpaka::getDevByIdx(::alpaka::Platform<Host>{}, 0u);
    auto devAcc = ::alpaka::getDevByIdx(::alpaka::Platform<Acc>{}, 0u);
    auto queue = Queue{devAcc};
    auto threadsPerBlock = warpSize * 2;

    // Copy setup
    m_copy.setup(seeds_buffer)->ignore();

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    auto thrustExecPolicy = thrust::device;
#else
    auto thrustExecPolicy = thrust::host;
#endif

    /*****************************************************************
     * Measurement Operations
     *****************************************************************/

    unsigned int n_modules;
    measurement_collection_types::const_view::size_type n_measurements =
        m_copy.get_size(measurements);

    // Get copy of barcode uniques
    measurement_collection_types::buffer uniques_buffer{n_measurements,
                                                        m_mr.main};
    m_copy.setup(uniques_buffer)->ignore();

    {
        measurement_collection_types::device uniques(uniques_buffer);

        measurement* uniques_end =
            thrust::unique_copy(thrustExecPolicy, measurements.ptr(),
                                measurements.ptr() + n_measurements,
                                uniques.begin(), measurement_equal_comp());
        n_modules = uniques_end - uniques.begin();
    }

    // Get upper bounds of unique elements
    vecmem::data::vector_buffer<unsigned int> upper_bounds_buffer{n_modules,
                                                                  m_mr.main};
    m_copy.setup(upper_bounds_buffer)->ignore();

    {
        vecmem::device_vector<unsigned int> upper_bounds(upper_bounds_buffer);

        measurement_collection_types::device uniques(uniques_buffer);

        thrust::upper_bound(thrustExecPolicy, measurements.ptr(),
                            measurements.ptr() + n_measurements,
                            uniques.begin(), uniques.begin() + n_modules,
                            upper_bounds.begin(), measurement_sort_comp());
    }

    /*****************************************************************
     * Kernel1: Create barcode sequence
     *****************************************************************/

    vecmem::data::vector_buffer<detray::geometry::barcode> barcodes_buffer{
        n_modules, m_mr.main};
    m_copy.setup(barcodes_buffer)->ignore();

    {
        auto blocksPerGrid =
            (barcodes_buffer.size() + threadsPerBlock - 1) / threadsPerBlock;
        auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

        ::alpaka::exec<Acc>(queue, workDiv, MakeBarcodeSequenceKernel{},
                            vecmem::get_data(uniques_buffer),
                            vecmem::get_data(barcodes_buffer));
        ::alpaka::wait(queue);
    }

    const unsigned int n_seeds = m_copy.get_size(seeds_buffer);

    // Prepare input parameters with seeds
    bound_track_parameters_collection_types::buffer in_params_buffer(n_seeds,
                                                                     m_mr.main);
    m_copy.setup(in_params_buffer)->ignore();
    m_copy(vecmem::get_data(seeds_buffer), vecmem::get_data(in_params_buffer))
        ->ignore();
    vecmem::data::vector_buffer<unsigned int> param_liveness_buffer(n_seeds,
                                                                    m_mr.main);
    m_copy.setup(param_liveness_buffer)->ignore();
    m_copy.memset(param_liveness_buffer, 1)->ignore();

    // Number of tracks per seed
    vecmem::data::vector_buffer<unsigned int> n_tracks_per_seed_buffer(
        n_seeds, m_mr.main);
    m_copy.setup(n_tracks_per_seed_buffer)->ignore();

    // Create a map for links
    std::map<unsigned int, vecmem::data::vector_buffer<candidate_link>>
        link_map;

    // Create a buffer of tip links
    vecmem::data::vector_buffer<typename candidate_link::link_index_type>
        tips_buffer{m_cfg.max_num_branches_per_seed * n_seeds, m_mr.main,
                    vecmem::data::buffer_type::resizable};
    m_copy.setup(tips_buffer)->wait();

    // Link size
    std::vector<std::size_t> n_candidates_per_step;
    n_candidates_per_step.reserve(m_cfg.max_track_candidates_per_track);

    unsigned int n_in_params = n_seeds;

    for (unsigned int step = 0;
         step < m_cfg.max_track_candidates_per_track && n_in_params > 0;
         step++) {

        /*****************************************************************
         * Kernel2: Apply material interaction
         ****************************************************************/

        {
            auto blocksPerGrid =
                (n_in_params + threadsPerBlock - 1) / threadsPerBlock;
            auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

            ::alpaka::exec<Acc>(queue, workDiv,
                                ApplyInteractionKernel<detector_type>{},
                                det_view, m_cfg, n_in_params,
                                vecmem::get_data(in_params_buffer),
                                vecmem::get_data(param_liveness_buffer));
            ::alpaka::wait(queue);
        }

        /*****************************************************************
         * Kernel3: Count the number of measurements per parameter
         ****************************************************************/

        unsigned int n_candidates = 0;

        {
            // Previous step
            const unsigned int prev_step = (step == 0 ? 0 : step - 1);

            // Buffer for kalman-updated parameters spawned by the measurement
            // candidates
            const unsigned int n_max_candidates =
                n_in_params * m_cfg.max_num_branches_per_surface;

            bound_track_parameters_collection_types::buffer
                updated_params_buffer(
                    n_in_params * m_cfg.max_num_branches_per_surface,
                    m_mr.main);
            m_copy.setup(updated_params_buffer)->ignore();

            vecmem::data::vector_buffer<unsigned int> updated_liveness_buffer(
                n_in_params * m_cfg.max_num_branches_per_surface, m_mr.main);
            m_copy.setup(updated_liveness_buffer)->ignore();

            // Create the link map
            link_map[step] = {n_in_params * m_cfg.max_num_branches_per_surface,
                              m_mr.main};
            m_copy.setup(link_map[step])->ignore();

            auto blocksPerGrid =
                (n_in_params + threadsPerBlock - 1) / threadsPerBlock;
            auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

            vecmem::unique_alloc_ptr<unsigned int> n_candidates_device =
                vecmem::make_unique_alloc<unsigned int>(m_mr.main);
            // TRACCC_CUDA_ERROR_CHECK(cudaMemsetAsync(
            //     n_candidates_device.get(), 0, sizeof(unsigned int), stream));

            ::alpaka::exec<Acc>(
                queue, workDiv, FindTracksKernel<detector_type, config_type>{},
                m_cfg, det_view, measurements,
                vecmem::get_data(in_params_buffer),
                vecmem::get_data(param_liveness_buffer), n_in_params,
                vecmem::get_data(barcodes_buffer),
                vecmem::get_data(upper_bounds_buffer),
                vecmem::get_data(link_map[prev_step]), step, n_max_candidates,
                vecmem::get_data(updated_params_buffer),
                vecmem::get_data(updated_liveness_buffer),
                vecmem::get_data(link_map[step]), n_candidates_device.get());
            ::alpaka::wait(queue);

            std::swap(in_params_buffer, updated_params_buffer);
            std::swap(param_liveness_buffer, updated_liveness_buffer);

            // TRACCC_CUDA_ERROR_CHECK(cudaMemcpyAsync(
            //     &n_candidates, n_candidates_device.get(), sizeof(unsigned
            //     int), cudaMemcpyDeviceToHost, stream));
        }

        if (n_candidates > 0) {
            /*****************************************************************
             * Kernel4: Get key and value for parameter sorting
             *****************************************************************/

            vecmem::data::vector_buffer<unsigned int> param_ids_buffer(
                n_candidates, m_mr.main);
            m_copy.setup(param_ids_buffer)->ignore();

            {
                vecmem::data::vector_buffer<device::sort_key> keys_buffer(
                    n_candidates, m_mr.main);
                m_copy.setup(keys_buffer)->ignore();

                auto blocksPerGrid =
                    (n_candidates + threadsPerBlock - 1) / threadsPerBlock;
                auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

                ::alpaka::exec<Acc>(queue, workDiv, FillSortKeysKernel{},
                                    vecmem::get_data(in_params_buffer),
                                    vecmem::get_data(keys_buffer),
                                    vecmem::get_data(param_ids_buffer));
                ::alpaka::wait(queue);

                // Sort the key and values
                vecmem::device_vector<device::sort_key> keys_device(
                    keys_buffer);
                vecmem::device_vector<unsigned int> param_ids_device(
                    param_ids_buffer);
                thrust::sort_by_key(thrustExecPolicy, keys_device.begin(),
                                    keys_device.end(),
                                    param_ids_device.begin());
            }

            /*****************************************************************
             * Kernel5: Propagate to the next surface
             *****************************************************************/

            {
                // Reset the number of tracks per seed
                m_copy.memset(n_tracks_per_seed_buffer, 0)->ignore();

                auto blocksPerGrid =
                    (n_candidates + threadsPerBlock - 1) / threadsPerBlock;
                auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

                ::alpaka::exec<Acc>(
                    queue, workDiv,
                    PropagateToNextSurfaceKernel<propagator_type, bfield_type,
                                                 config_type>{},
                    m_cfg, det_view, field_view,
                    vecmem::get_data(in_params_buffer),
                    vecmem::get_data(param_liveness_buffer),
                    vecmem::get_data(param_ids_buffer),
                    vecmem::get_data(link_map[step]), step, n_candidates,
                    vecmem::get_data(tips_buffer),
                    vecmem::get_data(n_tracks_per_seed_buffer));
                ::alpaka::wait(queue);
            }
        }

        // Fill the candidate size vector
        n_candidates_per_step.push_back(n_candidates);

        n_in_params = n_candidates;
    }

    // Create link buffer
    vecmem::data::jagged_vector_buffer<candidate_link> links_buffer(
        n_candidates_per_step, m_mr.main, m_mr.host);
    m_copy.setup(links_buffer)->ignore();

    // Copy link map to link buffer
    const auto n_steps = n_candidates_per_step.size();
    for (unsigned int it = 0; it < n_steps; it++) {

        vecmem::device_vector<candidate_link> in(link_map[it]);
        vecmem::device_vector<candidate_link> out(
            *(links_buffer.host_ptr() + it));

        thrust::copy(thrustExecPolicy, in.begin(),
                     in.begin() + n_candidates_per_step[it], out.begin());
    }

    /*****************************************************************
     * Kernel6: Build tracks
     *****************************************************************/

    // Get the number of tips
    auto n_tips_total = m_copy.get_size(tips_buffer);

    // Create track candidate buffer
    track_candidate_container_types::buffer track_candidates_buffer{
        {n_tips_total, m_mr.main},
        {std::vector<std::size_t>(n_tips_total,
                                  m_cfg.max_track_candidates_per_track),
         m_mr.main, m_mr.host, vecmem::data::buffer_type::resizable}};

    m_copy.setup(track_candidates_buffer.headers)->ignore();
    m_copy.setup(track_candidates_buffer.items)->ignore();

    // Create buffer for valid indices
    vecmem::data::vector_buffer<unsigned int> valid_indices_buffer(n_tips_total,
                                                                   m_mr.main);

    unsigned int n_valid_tracks;

    // @Note: nBlocks can be zero in case there is no tip. This happens when
    // chi2_max config is set tightly and no tips are found
    if (n_tips_total > 0) {
        vecmem::unique_alloc_ptr<unsigned int> n_valid_tracks_device =
            vecmem::make_unique_alloc<unsigned int>(m_mr.main);
        // TRACCC_CUDA_ERROR_CHECK(cudaMemsetAsync(n_valid_tracks_device.get(),
        // 0,
        //                                         sizeof(unsigned int),
        //                                         stream));

        auto blocksPerGrid =
            (n_tips_total + threadsPerBlock - 1) / threadsPerBlock;
        auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

        ::alpaka::exec<Acc>(
            queue, workDiv, BuildTracksKernel<config_type>{}, m_cfg,
            measurements, vecmem::get_data(seeds_buffer),
            vecmem::get_data(links_buffer), vecmem::get_data(tips_buffer),
            track_candidates_buffer, vecmem::get_data(valid_indices_buffer),
            n_valid_tracks_device.get());
        ::alpaka::wait(queue);

        // // Global counter object: Device -> Host
        // TRACCC_CUDA_ERROR_CHECK(cudaMemcpyAsync(
        //     &n_valid_tracks, n_valid_tracks_device.get(), sizeof(unsigned
        //     int), cudaMemcpyDeviceToHost, stream));
    }

    // Create pruned candidate buffer
    track_candidate_container_types::buffer prune_candidates_buffer{
        {n_valid_tracks, m_mr.main},
        {std::vector<std::size_t>(n_valid_tracks,
                                  m_cfg.max_track_candidates_per_track),
         m_mr.main, m_mr.host, vecmem::data::buffer_type::resizable}};

    m_copy.setup(prune_candidates_buffer.headers)->ignore();
    m_copy.setup(prune_candidates_buffer.items)->ignore();

    if (n_valid_tracks > 0) {
        auto blocksPerGrid =
            (n_valid_tracks + threadsPerBlock - 1) / threadsPerBlock;
        auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

        ::alpaka::exec<Acc>(queue, workDiv, PruneTracksKernel{},
                          track_candidates_buffer,
                          vecmem::get_data(valid_indices_buffer),
                          prune_candidates_buffer);
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

// Also need to add an Alpaka trait for the dynamic shared memory
namespace alpaka::trait {

template <typename TAcc, typename detector_t, typename config_t>
struct BlockSharedMemDynSizeBytes<
    traccc::alpaka::FindTracksKernel<detector_t, config_t>, TAcc> {
    template <typename TVec, typename... TArgs>
    ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
        traccc::alpaka::FindTracksKernel<detector_t,
                                         config_t> const& /* kernel */,
        TVec const& /* blockThreadExtent */, TVec const& threadElemExtent,
        TArgs const&... /* args */
        ) -> std::size_t {
        return static_cast<std::size_t>(threadElemExtent.prod()) *
               sizeof(unsigned int) * 2 *
               sizeof(std::pair<unsigned int, unsigned int>);
    }
};

}  // namespace alpaka::trait