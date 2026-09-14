"""
Passive shimming top-level workflow
===================================

It should:
    - define experiment inputs,
    - instantiate reusable workflow objects,
    - execute the passive-shimming pipeline step-by-step,
    - save outputs,
    - provide clear checkpoints for debugging.

All implementation details live in reusable modules/classes.

Expected modules
----------------
make_shim_rings.py
    external_stl_tray
    shim_ring

shimming_io.py
    ShimmingRun
    FieldMap

shimming_geometry.py
    TrayGeometry
    ShimMagnet
    CandidateTray

shimming_basis.py
    ShimBasis

shimming_optimization.py
    ShimOptimizationProblem
    ShimOptimizer

shimming_reporting.py
    ShimmingReporter

shimming_export.py
    ShimCollectionExporter
    ShimTrayExporter

These classes can be implemented/refactored incrementally while preserving
this top-level API.
"""

from pathlib import Path

import numpy as np

from shimming_io import ShimmingRun, FieldMap
from shimming_geometry import TrayGeometry, ShimMagnet, CandidateTray
from shimming_basis import ShimBasis
from shimming_optimization import ShimOptimizationProblem, ShimOptimizer

import shimming_optimization
import inspect
import pymoo
from shimming_reporting import ShimmingReporter
from shimming_export import ShimCollectionExporter, ShimTrayExporter


# ============================================================================
# RUN CONTROL
# ============================================================================

STOP_AFTER_STEP = None

RANDOM_SEED = 42

SHOW_INPUT_FIELD = True
SHOW_TRAY_GEOMETRY = True
SHOW_CANDIDATES = True
SHOW_FINAL_COLLECTIONS = True
SHOW_EXPORTED_STLS = True
PERTURBATION_SIZES = (2, 3, 5, 8, 10)

# ============================================================================
# STEP 0 — EXPERIMENT INPUTS
# ============================================================================

FMR_DATA_FILE = Path(
    "./data/Exp_1044_2026831.npy"
    # "./data/Exp_1043_2026913.npy"
    
)

OUTPUT_ROOT = Path(
    "./data/shimming_runs"
)

GAMMABAR_HZ_PER_T = 42.577478518e6

MEASURED_POSITION_SCALE_TO_M = 1e-3
MEASURED_FIELD_SCALE_TO_T = 1e-3


# ============================================================================
# STEP 1 — TRAY GEOMETRY INPUTS
# ============================================================================

USE_EXTERNAL_STL = False

EXTERNAL_STL_FILE = Path(
    "./data/delta1_first_layer_shim_tray.STL"
)

STL_UNITS = "mm"

TRAY_INNER_SURFACE_SEPARATION_M = (
    97e-3
)

STL_MARGIN_M = 0.0

# Used only when USE_EXTERNAL_STL=False.
CIRCULAR_TRAY_DIAMETER_M = (
    203.3e-3
)

CIRCULAR_TRAY_HEIGHTS_M = np.array(
    [
        # -36.10e-3,
        # +36.10e-3,
        -48.5e-3,
        +48.5e-3,

    ]
)


# ============================================================================
# STEP 2 — MAGNET INPUTS
# ============================================================================

MAGNET_MATERIAL = "N45"

MAGNET_DIMS_M = np.array(
    [
        6.35e-3,
        6.35e-3,
        3.18e-3,
    ]
)

POLARIZATION_T = 1.2

RADIAL_SPACING_FACTOR = 1.0
AZIMUTHAL_SPACING_FACTOR = 1.25

USE_SYMMETRY = False
SKIP_CENTER_MAGNETS = False

MAX_CANDIDATES_PER_PLANE = None
REQUIRE_FULL_MAGNET_INSIDE = True


# ============================================================================
# STEP 4 — OPTIMIZATION INPUTS
# ============================================================================

OPTIMIZATION_STATES = (
    -1,
    0,
    +1,
)

POPULATION_SIZE = 500
NUM_GENERATIONS = 100

STD_WEIGHT = 0.75
P01_P99_WEIGHT = 0.25

LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0


# ============================================================================
# STEP 5 — BASIS VALIDATION
# ============================================================================

NUM_BASIS_VALIDATION_TESTS = 5
BASIS_VALIDATION_ATOL_T = 1e-10
BASIS_VALIDATION_RTOL = 1e-8


# ============================================================================
# MAIN
# ============================================================================

def main():
    # ----------------------------------------------------------------------
    # STEP 0 — Initialize experiment and load measured field
    # ----------------------------------------------------------------------

    run = ShimmingRun(
        output_root=OUTPUT_ROOT,
        random_seed=RANDOM_SEED,
        stop_after_step=STOP_AFTER_STEP,
    )

    field_map = FieldMap.from_npy(
        filename=FMR_DATA_FILE,
        position_scale_to_m=
            MEASURED_POSITION_SCALE_TO_M,
        field_scale_to_T=
            MEASURED_FIELD_SCALE_TO_T,
        gammabar_Hz_per_T=
            GAMMABAR_HZ_PER_T,
    )

    field_map.validate()

    if SHOW_INPUT_FIELD:
        field_map.show(
            title="Measured B field"
        )

    run.save_config(
        "field_map",
        field_map
    )

    if run.stop_after(
        0
    ):
        return

    # ----------------------------------------------------------------------
    # STEP 1 — Define shim tray geometry
    # ----------------------------------------------------------------------

    if USE_EXTERNAL_STL:
        tray_geometry = (
            TrayGeometry.from_external_stl(
                stl_file=
                    EXTERNAL_STL_FILE,
                separation_m=
                    TRAY_INNER_SURFACE_SEPARATION_M,
                stl_units=
                    STL_UNITS,
                margin_m=
                    STL_MARGIN_M,
            )
        )

    else:
        tray_geometry = (
            TrayGeometry.circular(
                diameter_m=
                    CIRCULAR_TRAY_DIAMETER_M,
                heights_m=
                    CIRCULAR_TRAY_HEIGHTS_M,
            )
        )

    tray_geometry.validate()

    if SHOW_TRAY_GEOMETRY:
        tray_geometry.show()

    run.save_config(
        "tray_geometry",
        tray_geometry
    )

    if run.stop_after(
        1
    ):
        return

    # ----------------------------------------------------------------------
    # STEP 2 — Define shim magnet
    # ----------------------------------------------------------------------

    shim_magnet = ShimMagnet(
        material=
            MAGNET_MATERIAL,
        dimensions_m=
            MAGNET_DIMS_M,
        polarization_T=
            POLARIZATION_T,
    )

    shim_magnet.validate()

    run.save_config(
        "shim_magnet",
        shim_magnet
    )

    if run.stop_after(
        2
    ):
        return

    # ----------------------------------------------------------------------
    # STEP 3 — Construct candidate shim trays
    # ----------------------------------------------------------------------

    candidate_tray = CandidateTray(
        tray_geometry=
            tray_geometry,
        shim_magnet=
            shim_magnet,
        radial_spacing_factor=
            RADIAL_SPACING_FACTOR,
        azimuthal_spacing_factor=
            AZIMUTHAL_SPACING_FACTOR,
        symmetry=
            USE_SYMMETRY,
        skip_center_magnets=
            SKIP_CENTER_MAGNETS,
        max_candidates_per_plane=
            MAX_CANDIDATES_PER_PLANE,
        require_full_magnet_inside=
            REQUIRE_FULL_MAGNET_INSIDE,
    )

    candidate_tray.build()

    candidate_tray.print_statistics()

    run.save_report(
        "candidate_geometry",
        candidate_tray.statistics()
    )

    if SHOW_CANDIDATES:
        candidate_tray.show()

    if run.stop_after(
        3
    ):
        return

    # ----------------------------------------------------------------------
    # STEP 4 — Define optimization problem
    # ----------------------------------------------------------------------

    optimization_problem = (
        ShimOptimizationProblem(
            field_map=
                field_map,
            candidate_tray=
                candidate_tray,
            allowed_states=
                OPTIMIZATION_STATES,
            std_weight=
                STD_WEIGHT,
            percentile_range_weight=
                P01_P99_WEIGHT,
            low_percentile=
                LOW_PERCENTILE,
            high_percentile=
                HIGH_PERCENTILE,
        )
    )

    optimization_problem.validate()

    run.save_config(
        "optimization_problem",
        optimization_problem
    )

    if run.stop_after(
        4
    ):
        return

    # ----------------------------------------------------------------------
    # STEP 5 — Precompute shim basis
    # ----------------------------------------------------------------------

    shim_basis = ShimBasis(
        field_map=
            field_map,
        candidate_tray=
            candidate_tray,
    )

    shim_basis.compute()

    shim_basis.validate(
        n_tests=
            NUM_BASIS_VALIDATION_TESTS,
        atol_T=
            BASIS_VALIDATION_ATOL_T,
        rtol=
            BASIS_VALIDATION_RTOL,
        random_seed=
            RANDOM_SEED,
    )

    shim_basis.save(
        run.path(
            "shim_basis.npz"
        )
    )

    run.save_report(
        "shim_basis",
        shim_basis.statistics()
    )

    optimization_problem.attach_basis(
        shim_basis
    )

    if run.stop_after(
        5
    ):
        return

    # ----------------------------------------------------------------------
    # STEP 6 — Pre-shimming report
    # ----------------------------------------------------------------------

    reporter = ShimmingReporter(
        field_map=
            field_map,
        optimization_problem=
            optimization_problem,
    )

    pre_report = reporter.evaluate(
        B_total_T=
            field_map.B_T,
        label=
            "pre_shim",
    )

    reporter.print_report(
        pre_report
    )

    reporter.save_report(
        pre_report,
        run.path(
            "pre_shim_report.json"
        ),
    )

    if run.stop_after(
        6
    ):
        return

    # ----------------------------------------------------------------------
    # STEP 7 — Run optimizer
    # ----------------------------------------------------------------------

    optimizer = ShimOptimizer(
        problem=optimization_problem,
        population_size=POPULATION_SIZE,
        n_generations=NUM_GENERATIONS,
        random_seed=RANDOM_SEED,
        perturbation_sizes=PERTURBATION_SIZES,
    )

    diagnostic = optimizer.diagnose_search_landscape(
        max_greedy_steps=100,
        verbose=True,
    )

    optimizer.initial_states = (
        diagnostic["greedy_states"].copy()
    )

    result = optimizer.run()

    optimizer.save_history(
        run.path(
            "ga_history.json"
        )
    )

    optimizer.save_best_solution(
        run.path(
            "best_solution.json"
        )
    )

    if run.stop_after(
        7
    ):
        return

    # ----------------------------------------------------------------------
    # STEP 8 — Post-shimming report
    # ----------------------------------------------------------------------

    best_states = (
        optimizer.best_states
    )

    B_shim_T = (
        shim_basis.field_from_states(
            best_states
        )
    )

    B_total_T = (
        field_map.B_T
        + B_shim_T
    )

    field_map_shims_only = FieldMap(
        positions_m=
            field_map.positions_m,
        B_T=
            B_shim_T,
        gammabar_Hz_per_T=
            GAMMABAR_HZ_PER_T,
    )
    field_map_shims_only.show(title="Predicted B0 shim field")

    field_map_shimmed = FieldMap(
        positions_m=
            field_map.positions_m,
        B_T=
            B_total_T,
        gammabar_Hz_per_T=
            GAMMABAR_HZ_PER_T,
    )

    field_map_shimmed.show(title="Predicted B0 after passive shimming")
    post_report = reporter.evaluate(
        B_total_T=
            B_total_T,
        states=
            best_states,
        label=
            "post_shim",
    )

    post_report = (
        reporter.add_improvement(
            pre_report=
                pre_report,
            post_report=
                post_report,
        )
    )

    reporter.print_report(
        post_report
    )

    reporter.save_report(
        post_report,
        run.path(
            "post_shim_report.json"
        ),
    )

    if run.stop_after(
        8
    ):
        return

    # ----------------------------------------------------------------------
    # STEP 9 — Build and save optimized Magpylib collections
    # ----------------------------------------------------------------------

    collection_exporter = (
        ShimCollectionExporter(
            candidate_tray=
                candidate_tray,
            states=
                best_states,
        )
    )

    optimized_collections = (
        collection_exporter.build()
    )

    collection_exporter.save(
        output_directory=
            run.run_dir
    )

    if SHOW_FINAL_COLLECTIONS:
        collection_exporter.show()

    if run.stop_after(
        9
    ):
        return

    # ----------------------------------------------------------------------
    # STEP 10 — Export physical top and bottom STL trays
    # ----------------------------------------------------------------------

    tray_exporter = ShimTrayExporter(
        tray_geometry=
            tray_geometry,
        optimized_collections=
            optimized_collections,
    )

    tray_exporter.export(
        output_directory=
            run.run_dir,
        notch=False,
        orientation_marks=True,
        positive_polarity_mark="+",
        negative_polarity_mark=None,
    )

    if run.stop_after(
        10
    ):
        return

    # ----------------------------------------------------------------------
    # STEP 11 — Reload and verify exported STL trays
    # ----------------------------------------------------------------------

    verification = (
        tray_exporter.verify_exports()
    )

    run.save_report(
        "stl_verification",
        verification
    )

    if SHOW_EXPORTED_STLS:
        tray_exporter.show_exports()

    print(
        "\nPassive shimming workflow complete."
    )

    print(
        f"Run directory: "
        f"{run.run_dir}"
    )


if __name__ == "__main__":
    main()
