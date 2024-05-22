/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate.hpp"
#include "traccc/finding/ckf_aborter.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/finding/interaction_register.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// detray include(s).
#include "detray/propagator/actor_chain.hpp"
#include "detray/propagator/actors/aborters.hpp"
#include "detray/propagator/actors/parameter_resetter.hpp"
#include "detray/propagator/actors/parameter_transporter.hpp"
#include "detray/propagator/actors/pointwise_material_interactor.hpp"
#include "detray/propagator/propagator.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// Thrust Library
#include <thrust/pair.h>

namespace traccc::alpaka {

/// Track Finding algorithm for a set of tracks
template <typename stepper_t, typename navigator_t>
class finding_algorithm
    : public algorithm<track_candidate_container_types::buffer(
          const typename navigator_t::detector_type::view_type&,
          const typename stepper_t::magnetic_field_type&,
          const vecmem::data::jagged_vector_view<
              typename navigator_t::intersection_type>&,
          const typename measurement_collection_types::view&,
          const bound_track_parameters_collection_types::buffer&)> {

    /// Detector type
    using detector_type = typename navigator_t::detector_type;

    /// algebra type
    using algebra_type = typename detector_type::algebra_type;

    /// scalar type
    using scalar_type = detray::dscalar<algebra_type>;

    /// Field type
    using bfield_type = typename stepper_t::magnetic_field_type;

    /// Actor types
    using interactor = detray::pointwise_material_interactor<algebra_type>;

    /// Actor chain for propagate to the next surface and its propagator type
    using actor_type =
        detray::actor_chain<std::tuple, detray::pathlimit_aborter,
                            detray::parameter_transporter<algebra_type>,
                            interaction_register<interactor>, interactor,
                            ckf_aborter>;

    using propagator_type =
        detray::propagator<stepper_t, navigator_t, actor_type>;

    public:
    /// Configuration type
    using config_type = finding_config<scalar_type>;

    /// Constructor for the finding algorithm
    ///
    /// @param cfg  Configuration object
    /// @param mr   The memory resource to use
    /// @param copy Copy object
    finding_algorithm(const config_type& cfg, const traccc::memory_resource& mr,
                      vecmem::copy& copy);

    /// Get config object (const access)
    const finding_config<scalar_type>& get_config() const { return m_cfg; }

    /// Run the algorithm
    ///
    /// @param det_view  Detector view object
    /// @param navigation_buffer  Buffer for navigation candidates
    /// @param seeds     Input seeds
    track_candidate_container_types::buffer operator()(
        const typename detector_type::view_type& det_view,
        const bfield_type& field_view,
        const vecmem::data::jagged_vector_view<
            typename navigator_t::intersection_type>& navigation_buffer,
        const typename measurement_collection_types::view& measurements,
        const bound_track_parameters_collection_types::buffer& seeds)
        const override;

    private:
    /// Config object
    config_type m_cfg;
    /// Memory resource used by the algorithm
    traccc::memory_resource m_mr;
    /// The copy object to use
    vecmem::copy& m_copy;
};

}  // namespace traccc::alpaka