/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/alpaka/clusterization/clusterization_algorithm.hpp"
#include "traccc/alpaka/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/alpaka/clusterization/spacepoint_formation_algorithm.hpp"
#include "traccc/alpaka/finding/finding_algorithm.hpp"
#include "traccc/alpaka/fitting/fitting_algorithm.hpp"
#include "traccc/alpaka/seeding/seeding_algorithm.hpp"
#include "traccc/alpaka/seeding/track_params_estimation.hpp"
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/clusterization/spacepoint_formation_algorithm.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/finding/finding_algorithm.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/accelerator.hpp"
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_seeding.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/performance/container_comparator.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/io/frontend/detector_reader.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>
#endif

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
#include <vecmem/memory/hip/device_memory_resource.hpp>
#include <vecmem/memory/hip/host_memory_resource.hpp>
#include <vecmem/utils/hip/copy.hpp>
#endif

#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>

namespace po = boost::program_options;

int seq_run(const traccc::opts::detector& detector_opts,
            const traccc::opts::input_data& input_opts,
            const traccc::opts::clusterization& clusterization_opts,
            const traccc::opts::track_seeding& seeding_opts,
            const traccc::opts::track_finding& finding_opts,
            const traccc::opts::track_propagation& propagation_opts,
            const traccc::opts::performance& performance_opts,
            const traccc::opts::accelerator& accelerator_opts) {

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    vecmem::cuda::copy copy;
    vecmem::cuda::host_memory_resource cuda_host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &cuda_host_mr};
#elif ALPAKA_ACC_GPU_HIP_ENABLED
    vecmem::hip::copy copy;
    vecmem::hip::host_memory_resource hip_host_mr;
    vecmem::hip::device_memory_resource hip_device_mr;
    traccc::memory_resource mr{hip_device_mr, &hip_host_mr};
#else
    vecmem::copy copy;
    traccc::memory_resource mr{host_mr, &host_mr};
#endif

    // Read in the geometry.
    auto [surface_transforms, barcode_map] = traccc::io::read_geometry(
        detector_opts.detector_file,
        (detector_opts.use_detray_detector ? traccc::data_format::json
                                           : traccc::data_format::csv));

    using host_detector_type = detray::detector<detray::default_metadata,
                                                detray::host_container_types>;
    using device_detector_type =
        detray::detector<detray::default_metadata,
                         detray::device_container_types>;

    host_detector_type host_detector{host_mr};
    host_detector_type::buffer_type device_detector;
    host_detector_type::view_type device_detector_view;
    if (detector_opts.use_detray_detector) {
        // Set up the detector reader configuration.
        detray::io::detector_reader_config cfg;
        cfg.add_file(traccc::io::data_directory() +
                     detector_opts.detector_file);
        if (detector_opts.material_file.empty() == false) {
            cfg.add_file(traccc::io::data_directory() +
                         detector_opts.material_file);
        }
        if (detector_opts.grid_file.empty() == false) {
            cfg.add_file(traccc::io::data_directory() +
                         detector_opts.grid_file);
        }

        // Read the detector.
        auto det = detray::io::read_detector<host_detector_type>(host_mr, cfg);
        host_detector = std::move(det.first);

        // Copy it to the device.
        device_detector = detray::get_buffer(detray::get_data(host_detector),
                                             device_mr, copy);
        device_detector_view = detray::get_data(device_detector);
    }

    // Read the digitization configuration file
    auto digi_cfg =
        traccc::io::read_digitization_config(detector_opts.digitization_file);

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_modules = 0;
    uint64_t n_measurements = 0;
    uint64_t n_measurements_alpaka = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_spacepoints_alpaka = 0;
    uint64_t n_seeds = 0;
    uint64_t n_seeds_alpaka = 0;
    uint64_t n_found_tracks = 0;
    uint64_t n_found_tracks_alpaka = 0;
    uint64_t n_fitted_tracks = 0;
    uint64_t n_fitted_tracks_alpaka = 0;

    // Type definitions
    using stepper_type =
        detray::rk_stepper<detray::bfield::const_field_t::view_t,
                           host_detector_type::algebra_type,
                           detray::constrained_step<>>;
    using host_navigator_type = detray::navigator<const host_detector_type>;
    using device_navigator_type = detray::navigator<const device_detector_type>;

    using host_finding_algorithm =
        traccc::finding_algorithm<stepper_type, host_navigator_type>;
    using device_finding_algorithm =
        traccc::alpaka::finding_algorithm<stepper_type, device_navigator_type>;

    using host_fitting_algorithm = traccc::fitting_algorithm<
        traccc::kalman_fitter<stepper_type, host_navigator_type>>;
    using device_fitting_algorithm = traccc::alpaka::fitting_algorithm<
        traccc::kalman_fitter<stepper_type, device_navigator_type>>;

    // Algorithm configuration(s).
    host_finding_algorithm::config_type finding_cfg;
    finding_cfg.min_track_candidates_per_track =
        finding_opts.track_candidates_range[0];
    finding_cfg.max_track_candidates_per_track =
        finding_opts.track_candidates_range[1];
    finding_cfg.min_step_length_for_next_surface =
        finding_opts.min_step_length_for_next_surface;
    finding_cfg.max_step_counts_for_next_surface =
        finding_opts.max_step_counts_for_next_surface;
    finding_cfg.chi2_max = finding_opts.chi2_max;
    finding_cfg.max_num_branches_per_seed = finding_opts.nmax_per_seed;
    finding_cfg.max_num_skipping_per_cand =
        finding_opts.max_num_skipping_per_cand;
    finding_cfg.propagation = propagation_opts.config;

    host_fitting_algorithm::config_type fitting_cfg;
    fitting_cfg.propagation = propagation_opts.config;

    // Constant B field for the track finding and fitting
    const traccc::vector3 field_vec = {0.f, 0.f,
                                       seeding_opts.seedfinder.bFieldInZ};
    const detray::bfield::const_field_t field =
        detray::bfield::create_const_field(field_vec);

    traccc::host::clusterization_algorithm ca(host_mr);
    traccc::host::spacepoint_formation_algorithm sf(host_mr);
    traccc::seeding_algorithm sa(seeding_opts.seedfinder,
                                 {seeding_opts.seedfinder},
                                 seeding_opts.seedfilter, host_mr);
    traccc::track_params_estimation tp(host_mr);
    host_finding_algorithm finding_alg(finding_cfg);
    host_fitting_algorithm fitting_alg(fitting_cfg);

    traccc::alpaka::clusterization_algorithm ca_alpaka(
        mr, copy, clusterization_opts.target_cells_per_partition);
    traccc::alpaka::measurement_sorting_algorithm ms_alpaka(copy);
    traccc::alpaka::spacepoint_formation_algorithm sf_alpaka(mr, copy);
    traccc::alpaka::seeding_algorithm sa_alpaka(
        seeding_opts.seedfinder, {seeding_opts.seedfinder},
        seeding_opts.seedfilter, mr, copy);
    traccc::alpaka::track_params_estimation tp_alpaka(mr, copy);
    device_finding_algorithm finding_alg_alpaka(finding_cfg, mr, copy);
    device_fitting_algorithm fitting_alg_alpaka(fitting_cfg, mr, copy);

    traccc::device::container_d2h_copy_alg<
        traccc::track_candidate_container_types>
        copy_track_candidates(mr, copy);
    traccc::device::container_d2h_copy_alg<traccc::track_state_container_types>
        copy_track_states(mr, copy);

    // performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});

    traccc::performance::timing_info elapsedTimes;

    // Loop over events
    for (unsigned int event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        // Instantiate host containers/collections
        traccc::io::cell_reader_output read_out_per_event(mr.host);
        traccc::host::clusterization_algorithm::output_type
            measurements_per_event;
        traccc::host::spacepoint_formation_algorithm::output_type
            spacepoints_per_event;
        traccc::seeding_algorithm::output_type seeds;
        traccc::track_params_estimation::output_type params;
        host_finding_algorithm::output_type track_candidates;
        host_fitting_algorithm::output_type track_states;

        // Instantiate alpaka containers/collections
        traccc::measurement_collection_types::buffer measurements_alpaka_buffer(
            0, *mr.host);
        traccc::spacepoint_collection_types::buffer spacepoints_alpaka_buffer(
            0, *mr.host);
        traccc::seed_collection_types::buffer seeds_alpaka_buffer(0, *mr.host);
        traccc::bound_track_parameters_collection_types::buffer
            params_alpaka_buffer(0, *mr.host);
        traccc::track_candidate_container_types::buffer track_candidates_buffer;
        traccc::track_state_container_types::buffer track_states_buffer;

        {
            traccc::performance::timer wall_t("Wall time", elapsedTimes);

            {
                traccc::performance::timer t("File reading  (cpu)",
                                             elapsedTimes);
                // Read the cells from the relevant event file into host memory.
                traccc::io::read_cells(read_out_per_event, event,
                                       input_opts.directory, input_opts.format,
                                       &surface_transforms, &digi_cfg,
                                       barcode_map.get());
            }  // stop measuring file reading timer

            const traccc::cell_collection_types::host& cells_per_event =
                read_out_per_event.cells;
            const traccc::cell_module_collection_types::host&
                modules_per_event = read_out_per_event.modules;

            /*-----------------------------
                Clusterization and Spacepoint Creation (alpaka)
            -----------------------------*/
            // Create device copy of input collections
            traccc::cell_collection_types::buffer cells_buffer(
                cells_per_event.size(), mr.main);
            copy(vecmem::get_data(cells_per_event), cells_buffer);
            traccc::cell_module_collection_types::buffer modules_buffer(
                modules_per_event.size(), mr.main);
            copy(vecmem::get_data(modules_per_event), modules_buffer);

            {
                traccc::performance::timer t("Clusterization (alpaka)",
                                             elapsedTimes);
                // Reconstruct it into spacepoints on the device.
                measurements_alpaka_buffer =
                    ca_alpaka(cells_buffer, modules_buffer);
                ms_alpaka(measurements_alpaka_buffer);
            }  // stop measuring clusterization alpaka timer

            {
                traccc::performance::timer t("Spacepoint formation (alpaka)",
                                             elapsedTimes);
                spacepoints_alpaka_buffer =
                    sf_alpaka(measurements_alpaka_buffer, modules_buffer);
            }  // stop measuring spacepoint formation alpaka timer

            if (accelerator_opts.compare_with_cpu) {

                /*-----------------------------
                    Clusterization (cpu)
                -----------------------------*/

                {
                    traccc::performance::timer t("Clusterization  (cpu)",
                                                 elapsedTimes);
                    measurements_per_event =
                        ca(vecmem::get_data(cells_per_event),
                           vecmem::get_data(modules_per_event));
                }  // stop measuring clusterization cpu timer

                /*---------------------------------
                    Spacepoint formation (cpu)
                ---------------------------------*/

                {
                    traccc::performance::timer t("Spacepoint formation  (cpu)",
                                                 elapsedTimes);
                    spacepoints_per_event =
                        sf(vecmem::get_data(measurements_per_event),
                           vecmem::get_data(modules_per_event));
                }  // stop measuring spacepoint formation cpu timer
            }

            /*----------------------------
                Seeding algorithm
            ----------------------------*/

            // Alpaka

            {
                traccc::performance::timer t("Seeding (alpaka)", elapsedTimes);
                seeds_alpaka_buffer = sa_alpaka(spacepoints_alpaka_buffer);
            }  // stop measuring seeding alpaka timer

            // CPU

            if (accelerator_opts.compare_with_cpu) {
                traccc::performance::timer t("Seeding  (cpu)", elapsedTimes);
                seeds = sa(spacepoints_per_event);
            }  // stop measuring seeding cpu timer

            /*----------------------------
            Track params estimation
            ----------------------------*/

            // Alpaka

            {
                traccc::performance::timer t("Track params (alpaka)",
                                             elapsedTimes);
                params_alpaka_buffer = tp_alpaka(
                    spacepoints_alpaka_buffer, seeds_alpaka_buffer, field_vec);
            }  // stop measuring track params timer

            // CPU

            if (accelerator_opts.compare_with_cpu) {
                traccc::performance::timer t("Track params  (cpu)",
                                             elapsedTimes);
                params = tp(spacepoints_per_event, seeds, field_vec);
            }  // stop measuring track params cpu timer

            // Perform track finding and fitting only when using a Detray
            // geometry.
            if (detector_opts.use_detray_detector) {
                // Alpaka
                auto navigation_buffer = detray::create_candidates_buffer(
                    host_detector,
                    finding_cfg.navigation_buffer_size_scaler *
                        copy.get_size(seeds_alpaka_buffer),
                    mr.main, mr.host);
                {
                    traccc::performance::timer timer{"Track finding (alpaka)",
                                                     elapsedTimes};
                    track_candidates_buffer = finding_alg_alpaka(
                        device_detector_view, field, navigation_buffer,
                        measurements_alpaka_buffer, params_alpaka_buffer);
                }
                // CPU
                if (accelerator_opts.compare_with_cpu) {
                    traccc::performance::timer timer{"Track finding (cpu)",
                                                     elapsedTimes};
                    track_candidates = finding_alg(
                        host_detector, field, measurements_per_event, params);
                }
                // Alpaka
                {
                    traccc::performance::timer timer{"Track fitting (alpaka)",
                                                     elapsedTimes};
                    track_states_buffer = fitting_alg_alpaka(
                        device_detector_view, field, navigation_buffer,
                        track_candidates_buffer);
                }
                // CPU
                if (accelerator_opts.compare_with_cpu) {
                    traccc::performance::timer timer{"Track fitting (cpu)",
                                                     elapsedTimes};
                    track_states =
                        fitting_alg(host_detector, field, track_candidates);
                }
            }

        }  // Stop measuring wall time

        /*----------------------------------
          compare cpu and alpaka result
          ----------------------------------*/

        traccc::measurement_collection_types::host
            measurements_per_event_alpaka;
        traccc::spacepoint_collection_types::host spacepoints_per_event_alpaka;
        traccc::seed_collection_types::host seeds_alpaka;
        traccc::bound_track_parameters_collection_types::host params_alpaka;

        copy(measurements_alpaka_buffer, measurements_per_event_alpaka)->wait();
        copy(spacepoints_alpaka_buffer, spacepoints_per_event_alpaka)->wait();
        copy(seeds_alpaka_buffer, seeds_alpaka)->wait();
        copy(params_alpaka_buffer, params_alpaka)->wait();
        auto track_candidates_alpaka =
            copy_track_candidates(track_candidates_buffer);
        auto track_states_alpaka = copy_track_states(track_states_buffer);

        if (accelerator_opts.compare_with_cpu) {

            // Show which event we are currently presenting the results for.
            std::cout << "===>>> Event " << event << " <<<===" << std::endl;

            // Compare the measurements made on the host and on the device.
            traccc::collection_comparator<traccc::measurement>
                compare_measurements{"measurements"};
            compare_measurements(
                vecmem::get_data(measurements_per_event),
                vecmem::get_data(measurements_per_event_alpaka));

            // Compare the spacepoints made on the host and on the device.
            traccc::collection_comparator<traccc::spacepoint>
                compare_spacepoints{"spacepoints"};
            compare_spacepoints(vecmem::get_data(spacepoints_per_event),
                                vecmem::get_data(spacepoints_per_event_alpaka));

            // Compare the seeds made on the host and on the device
            traccc::collection_comparator<traccc::seed> compare_seeds{
                "seeds", traccc::details::comparator_factory<traccc::seed>{
                             vecmem::get_data(spacepoints_per_event),
                             vecmem::get_data(spacepoints_per_event_alpaka)}};
            compare_seeds(vecmem::get_data(seeds),
                          vecmem::get_data(seeds_alpaka));

            // Compare the track parameters made on the host and on the device.
            traccc::collection_comparator<traccc::bound_track_parameters>
                compare_track_parameters{"track parameters"};
            compare_track_parameters(vecmem::get_data(params),
                                     vecmem::get_data(params_alpaka));

            // Compare tracks found on the host and on the device.
            traccc::collection_comparator<
                traccc::track_candidate_container_types::host::header_type>
                compare_track_candidates{"track candidates (header)"};
            compare_track_candidates(
                vecmem::get_data(track_candidates.get_headers()),
                vecmem::get_data(track_candidates_alpaka.get_headers()));

            unsigned int n_matches = 0;
            for (unsigned int i = 0; i < track_candidates.size(); i++) {
                auto iso = traccc::details::is_same_object(
                    track_candidates.at(i).items);

                for (unsigned int j = 0; j < track_candidates_alpaka.size();
                     j++) {
                    if (iso(track_candidates_alpaka.at(j).items)) {
                        n_matches++;
                        break;
                    }
                }
            }

            std::cout << "  Track candidates (item) matching rate: "
                      << 100. * float(n_matches) /
                             std::max(track_candidates.size(),
                                      track_candidates_alpaka.size())
                      << "%" << std::endl;

            // Compare tracks fitted on the host and on the device.
            traccc::collection_comparator<
                traccc::track_state_container_types::host::header_type>
                compare_track_states{"track states"};
            compare_track_states(
                vecmem::get_data(track_states.get_headers()),
                vecmem::get_data(track_states_alpaka.get_headers()));
        }
        /// Statistics
        n_modules += read_out_per_event.modules.size();
        n_cells += read_out_per_event.cells.size();
        n_measurements += measurements_per_event.size();
        n_spacepoints += spacepoints_per_event.size();
        n_seeds += seeds.size();
        n_measurements_alpaka += measurements_per_event_alpaka.size();
        n_spacepoints_alpaka += spacepoints_per_event_alpaka.size();
        n_seeds_alpaka += seeds_alpaka.size();
        n_found_tracks += track_candidates.size();
        n_found_tracks_alpaka += track_candidates_alpaka.size();
        n_fitted_tracks += track_states.size();
        n_fitted_tracks_alpaka += track_states_alpaka.size();

        if (performance_opts.run) {

            traccc::event_map evt_map(
                event, detector_opts.detector_file,
                detector_opts.digitization_file, input_opts.directory,
                input_opts.directory, input_opts.directory, host_mr);
            sd_performance_writer.write(
                vecmem::get_data(seeds_alpaka),
                vecmem::get_data(spacepoints_per_event_alpaka), evt_map);
        }
    }

    if (performance_opts.run) {
        sd_performance_writer.finalize();
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_cells << " cells from " << n_modules
              << " modules" << std::endl;
    std::cout << "- created (cpu)  " << n_measurements << " measurements     "
              << std::endl;
    std::cout << "- created (alpaka)  " << n_measurements_alpaka
              << " measurements     " << std::endl;
    std::cout << "- created (cpu)  " << n_spacepoints << " spacepoints     "
              << std::endl;
    std::cout << "- created (alpaka) " << n_spacepoints_alpaka
              << " spacepoints     " << std::endl;

    std::cout << "- created  (cpu) " << n_seeds << " seeds" << std::endl;
    std::cout << "- created (alpaka) " << n_seeds_alpaka << " seeds"
              << std::endl;
    std::cout << "- found (cpu)    " << n_found_tracks << " tracks"
              << std::endl;
    std::cout << "- found (alpaka)   " << n_found_tracks_alpaka << " tracks"
              << std::endl;
    std::cout << "- fitted (cpu)   " << n_fitted_tracks << " tracks"
              << std::endl;
    std::cout << "- fitted (alpaka)  " << n_fitted_tracks_alpaka << " tracks"
              << std::endl;
    std::cout << "==>Elapsed times...\n" << elapsedTimes << std::endl;

    return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::clusterization clusterization_opts;
    traccc::opts::track_seeding seeding_opts;
    traccc::opts::track_finding finding_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::accelerator accelerator_opts;
    traccc::opts::program_options program_opts{
        "Full Tracking Chain Using Alpaka",
        {detector_opts, input_opts, clusterization_opts, seeding_opts,
         finding_opts, propagation_opts, performance_opts, accelerator_opts},
        argc,
        argv};

    // Run the application.
    return seq_run(detector_opts, input_opts, clusterization_opts, seeding_opts,
                   finding_opts, propagation_opts, performance_opts,
                   accelerator_opts);
}
