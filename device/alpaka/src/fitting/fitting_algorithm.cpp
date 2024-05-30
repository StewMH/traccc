/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/alpaka/fitting/fitting_algorithm.hpp"

#include "../utils/utils.hpp"
#include "traccc/fitting/device/fit.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"

// detray include(s).
#include "detray/core/detector_metadata.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/propagator/rk_stepper.hpp"

// System include(s).
#include <vector>

namespace traccc::alpaka {

template <typename fitter_t, typename detector_view_t>
struct FitTrackKernel {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc, detector_view_t det_data,
        const typename fitter_t::bfield_type field_data,
        const typename fitter_t::config_type cfg,
        vecmem::data::jagged_vector_view<typename fitter_t::intersection_type>
            nav_candidates_buffer,
        track_candidate_container_types::const_view* track_candidates_view,
        track_state_container_types::view* track_states_view) const {

        int globalThreadIdx =
            ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Threads>(acc)[0];
        device::fit<fitter_t>(globalThreadIdx, det_data, field_data, cfg,
                              nav_candidates_buffer, *track_candidates_view,
                              *track_states_view);
    }
};

template <typename fitter_t>
fitting_algorithm<fitter_t>::fitting_algorithm(
    const config_type& cfg, const traccc::memory_resource& mr,
    vecmem::copy& copy)
    : m_cfg(cfg), m_mr(mr), m_copy(copy) {}

template <typename fitter_t>
track_state_container_types::buffer fitting_algorithm<fitter_t>::operator()(
    const typename fitter_t::detector_type::view_type& det_view,
    const typename fitter_t::bfield_type& field_view,
    const vecmem::data::jagged_vector_view<
        typename fitter_t::intersection_type>& navigation_buffer,
    const typename track_candidate_container_types::const_view&
        track_candidates_view) const {

    // Setup alpaka
    auto devHost = ::alpaka::getDevByIdx(::alpaka::Platform<Host>{}, 0u);
    auto devAcc = ::alpaka::getDevByIdx(::alpaka::Platform<Acc>{}, 0u);
    auto queue = Queue{devAcc};
    auto threadsPerBlock = warpSize * 2;

    // Number of tracks
    const track_candidate_container_types::const_device::header_vector::
        size_type n_tracks = m_copy.get_size(track_candidates_view.headers);

    // Get the sizes of the track candidates in each track
    const std::vector<track_candidate_container_types::const_device::
                          item_vector::value_type::size_type>
        candidate_sizes = m_copy.get_sizes(track_candidates_view.items);

    track_state_container_types::buffer track_states_buffer{
        {n_tracks, m_mr.main},
        {candidate_sizes, m_mr.main, m_mr.host,
         vecmem::data::buffer_type::resizable}};
    track_state_container_types::view track_states_view(track_states_buffer);

    m_copy.setup(track_states_buffer.headers);
    m_copy.setup(track_states_buffer.items);
    m_copy.setup(navigation_buffer);

    // Wrap the buffers in alpaka buffers
    auto bufAcc_const_candidates =
        ::alpaka::allocBuf<track_candidate_container_types::const_view, Idx>(
            devAcc, 1u);
    auto bufHost_const_candidates =
        ::alpaka::allocBuf<track_candidate_container_types::const_view, Idx>(
            devHost, 1u);
    const track_candidate_container_types::const_view* pBufHost_const_candidates =
        ::alpaka::getPtrNative(bufHost_const_candidates);
    pBufHost_const_candidates = &track_candidates_view;

    auto bufAcc_states =
        ::alpaka::allocBuf<track_state_container_types::view, Idx>(devAcc, 1u);
    auto bufHost_states =
        ::alpaka::allocBuf<track_state_container_types::view, Idx>(devHost, 1u);
    track_state_container_types::view* pBufHost_states =
        ::alpaka::getPtrNative(bufHost_states);
    pBufHost_states = &track_states_view;

    ::alpaka::memcpy(queue, bufAcc_const_candidates, bufHost_const_candidates);
    ::alpaka::memcpy(queue, bufAcc_states, bufHost_states);
    ::alpaka::wait(queue);

    // Calculate the number of threads and thread blocks to run the track
    // fitting
    if (n_tracks > 0) {
        const auto blocksPerGrid =
            (n_tracks + threadsPerBlock - 1) / threadsPerBlock;
        auto workDiv = makeWorkDiv<Acc>(blocksPerGrid, threadsPerBlock);

        // Run the track fitting
        ::alpaka::exec<Acc>(queue, workDiv, FitTrackKernel<fitter_t, typename fitter_t::detector_type::view_type>{},
                            det_view, field_view, m_cfg,
                            navigation_buffer,
                            ::alpaka::getPtrNative(bufAcc_const_candidates),
                            ::alpaka::getPtrNative(bufAcc_states));
        ::alpaka::wait(queue);
    }

    // Copy the results back to the host
    ::alpaka::memcpy(queue, bufHost_states, bufAcc_states);

    return track_states_buffer;
}

// Explicit template instantiation
using default_detector_type =
    detray::detector<detray::default_metadata, detray::device_container_types>;
using default_stepper_type =
    detray::rk_stepper<covfie::field<detray::bfield::const_bknd_t>::view_t,
                       default_algebra, detray::constrained_step<>>;
using default_navigator_type = detray::navigator<const default_detector_type>;
using default_fitter_type =
    kalman_fitter<default_stepper_type, default_navigator_type>;
template class fitting_algorithm<default_fitter_type>;

}  // namespace traccc::alpaka
