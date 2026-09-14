"""
Diagnostic numerical validation for ShimBasis.

This test intentionally decomposes the superposition check into controlled
cases so that failures can be localized to:

1. one positive-polarity magnet,
2. all positive-polarity magnets,
3. one negative-polarity magnet,
4. a random {-1, 0, +1} configuration,
5. ShimBasis.validate() over several random configurations.

The direct Magpylib comparison constructs each selected magnet explicitly
from the candidate geometry instead of mutating the polarization attribute of
an already-created object. This avoids ambiguity in negative-polarity tests.

Run directly
------------
    python test_shim_basis.py

or with pytest
--------------
    pytest -q test_shim_basis.py
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import magpylib as magpy
import numpy as np

from shimming_basis import ShimBasis
from shimming_io import FieldMap
from utils import get_magnetic_field


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

RANDOM_SEED = 1234

N_MAGNETS = 24
N_DSV_POINTS = 250

TRAY_RADIUS_M = 80e-3
DSV_RADIUS_M = 40e-3

PLANE_HEIGHTS_M = (
    -48.5e-3,
    +48.5e-3,
)

MAGNET_DIMS_M = np.array(
    [
        6.35e-3,
        6.35e-3,
        3.18e-3,
    ],
    dtype=float,
)

POLARIZATION_T = 1.2

GAMMABAR_HZ_PER_T = 42.577478518e6

ATOL_T = 1e-10
RTOL = 1e-8


# ============================================================================
# RANDOM TEST GEOMETRY
# ============================================================================

def random_points_in_sphere(
    rng,
    n_points,
    radius_m,
):
    """Generate uniformly distributed random points inside a sphere."""
    directions = rng.normal(
        size=(
            n_points,
            3,
        )
    )

    directions /= np.linalg.norm(
        directions,
        axis=1,
        keepdims=True,
    )

    radii = (
        radius_m
        * rng.random(
            n_points
        ) ** (
            1.0 / 3.0
        )
    )

    return (
        directions
        * radii[:, None]
    )


def random_magnet_candidates(
    rng,
    n_magnets,
):
    """
    Create random positive-polarity Cuboid candidates in two shim planes.

    All candidate basis magnets begin with +Z polarization.
    """
    magnets = []
    candidate_ids = []

    safe_radius = (
        TRAY_RADIUS_M
        - 0.5
        * np.hypot(
            MAGNET_DIMS_M[0],
            MAGNET_DIMS_M[1],
        )
    )

    for index in range(
        n_magnets
    ):
        radius = (
            safe_radius
            * np.sqrt(
                rng.random()
            )
        )

        angle = (
            2.0
            * np.pi
            * rng.random()
        )

        z = float(
            rng.choice(
                PLANE_HEIGHTS_M
            )
        )

        magnet = magpy.magnet.Cuboid(
            dimension=
                MAGNET_DIMS_M,
            position=(
                radius
                * np.cos(
                    angle
                ),
                radius
                * np.sin(
                    angle
                ),
                z,
            ),
            polarization=(
                0.0,
                0.0,
                POLARIZATION_T,
            ),
        )

        # In-plane physical orientation only.
        magnet.rotate_from_angax(
            np.degrees(
                angle
            ),
            "z",
        )

        magnets.append(
            magnet
        )

        candidate_ids.append(
            f"candidate_{index:04d}"
        )

    return (
        magnets,
        candidate_ids,
    )


def build_test_problem():
    """Create DSV, candidates, basis-compatible container, and random states."""
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    positions_m = (
        random_points_in_sphere(
            rng,
            n_points=
                N_DSV_POINTS,
            radius_m=
                DSV_RADIUS_M,
        )
    )

    # FieldMap only needs a valid measured field for this basis test.
    B_measured_T = np.full(
        N_DSV_POINTS,
        0.25,
        dtype=float,
    )

    field_map = FieldMap(
        positions_m=
            positions_m,
        B_T=
            B_measured_T,
        gammabar_Hz_per_T=
            GAMMABAR_HZ_PER_T,
    )

    (
        candidates,
        candidate_ids,
    ) = random_magnet_candidates(
        rng,
        N_MAGNETS,
    )

    candidate_tray = SimpleNamespace(
        candidates=
            candidates,
        candidate_ids=
            candidate_ids,
    )

    states = rng.choice(
        np.array(
            [
                -1,
                0,
                1,
            ],
            dtype=int,
        ),
        size=
            N_MAGNETS,
        replace=
            True,
    )

    return (
        field_map,
        candidate_tray,
        states,
    )


# ============================================================================
# DIRECT MAGPYLIB REFERENCE CALCULATION
# ============================================================================

def explicit_magnet_from_candidate(
    candidate,
    state,
):
    """
    Create a fresh Cuboid from candidate geometry with explicit polarity.

    This avoids changing the polarization of an existing oriented object.
    """
    state = int(
        state
    )

    if state not in (
        -1,
        1,
    ):
        raise ValueError(
            "state must be -1 or +1 for an explicit magnet."
        )

    return magpy.magnet.Cuboid(
        dimension=np.asarray(
            candidate.dimension,
            dtype=float,
        ).copy(),
        position=np.asarray(
            candidate.position,
            dtype=float,
        ).copy(),
        polarization=np.asarray(
            [
                0.0,
                0.0,
                state * POLARIZATION_T,
            ],
            dtype=float,
        ),
        orientation=copy.deepcopy(
            candidate.orientation
        ),
    )


def direct_field_for_states(
    candidates,
    states,
    sensors,
):
    """
    Compute complete Bz shim field directly with one Magpylib Collection.
    """
    states = np.asarray(
        states,
        dtype=int,
    ).reshape(
        -1
    )

    collection = magpy.Collection()

    for candidate, state in zip(
        candidates,
        states,
    ):
        if state == 0:
            continue

        collection.add(
            explicit_magnet_from_candidate(
                candidate,
                state,
            )
        )

    if np.count_nonzero(
        states
    ) == 0:
        n_points = len(
            sensors[0].position
        )

        return np.zeros(
            n_points,
            dtype=float,
        )

    return np.asarray(
        get_magnetic_field(
            magnets=
                collection,
            sensors=
                sensors,
            axis=
                2,
        ),
        dtype=float,
    ).reshape(
        -1
    )


def direct_field_single_magnet(
    candidate,
    state,
    sensors,
):
    """Direct Bz field of one explicitly constructed +/- candidate."""
    magnet = explicit_magnet_from_candidate(
        candidate,
        state,
    )

    return np.asarray(
        get_magnetic_field(
            magnets=
                magnet,
            sensors=
                sensors,
            axis=
                2,
        ),
        dtype=float,
    ).reshape(
        -1
    )


# ============================================================================
# COMPARISON UTILITIES
# ============================================================================

def comparison_statistics(
    expected_T,
    actual_T,
):
    """Return numerical error statistics between two field arrays."""
    expected_T = np.asarray(
        expected_T,
        dtype=float,
    ).reshape(
        -1
    )

    actual_T = np.asarray(
        actual_T,
        dtype=float,
    ).reshape(
        -1
    )

    error_T = (
        actual_T
        - expected_T
    )

    max_abs_error_T = float(
        np.max(
            np.abs(
                error_T
            )
        )
    )

    rms_error_T = float(
        np.sqrt(
            np.mean(
                error_T ** 2
            )
        )
    )

    scale_T = max(
        float(
            np.max(
                np.abs(
                    expected_T
                )
            )
        ),
        np.finfo(
            float
        ).eps,
    )

    return {
        "max_abs_error_T":
            max_abs_error_T,
        "rms_error_T":
            rms_error_T,
        "max_relative_error":
            max_abs_error_T
            / scale_T,
        "passed":
            bool(
                np.allclose(
                    actual_T,
                    expected_T,
                    atol=
                        ATOL_T,
                    rtol=
                        RTOL,
                )
            ),
    }


def print_comparison(
    title,
    expected_T,
    actual_T,
):
    """Print a compact diagnostic comparison and return its statistics."""
    stats = comparison_statistics(
        expected_T,
        actual_T,
    )

    print(
        f"\n{title}"
    )

    print(
        f"  Max |reference field|: "
        f"{np.max(np.abs(expected_T)):.6e} T"
    )

    print(
        f"  Max absolute error: "
        f"{stats['max_abs_error_T']:.6e} T"
    )

    print(
        f"  RMS error: "
        f"{stats['rms_error_T']:.6e} T"
    )

    print(
        f"  Max relative error: "
        f"{stats['max_relative_error']:.6e}"
    )

    print(
        f"  Passed: "
        f"{stats['passed']}"
    )

    return stats


# ============================================================================
# CONTROLLED DIAGNOSTIC TESTS
# ============================================================================

def test_single_positive_magnet():
    """
    TEST A:
    one + magnet direct Magpylib vs one ShimBasis column.
    """
    (
        field_map,
        candidate_tray,
        _,
    ) = build_test_problem()

    basis = ShimBasis(
        field_map=
            field_map,
        candidate_tray=
            candidate_tray,
        debug=
            False,
    )

    basis.compute()

    index = 0

    B_basis_T = (
        basis.basis_T[
            :,
            index,
        ]
    )

    B_direct_T = (
        direct_field_single_magnet(
            candidate_tray.candidates[
                index
            ],
            +1,
            field_map.sensors,
        )
    )

    stats = print_comparison(
        "TEST A — single positive-polarity magnet",
        B_direct_T,
        B_basis_T,
    )

    assert stats[
        "passed"
    ], (
        "One positive candidate does not match "
        "its ShimBasis column."
    )


def test_all_positive_magnets():
    """
    TEST B:
    complete all-positive Magpylib collection vs sum of all basis columns.
    """
    (
        field_map,
        candidate_tray,
        _,
    ) = build_test_problem()

    basis = ShimBasis(
        field_map=
            field_map,
        candidate_tray=
            candidate_tray,
    )

    basis.compute()

    states = np.ones(
        N_MAGNETS,
        dtype=int,
    )

    B_basis_T = (
        basis.field_from_states(
            states
        )
    )

    B_direct_T = (
        direct_field_for_states(
            candidate_tray.candidates,
            states,
            field_map.sensors,
        )
    )

    stats = print_comparison(
        "TEST B — all positive-polarity magnets",
        B_direct_T,
        B_basis_T,
    )

    assert stats[
        "passed"
    ], (
        "Sum of positive basis columns does not match "
        "the direct positive collection."
    )


def test_single_negative_magnet():
    """
    TEST C:
    one explicitly negative direct magnet vs negative of one basis column.

    Also checks the exact sign symmetry:
        B(-P) = -B(+P)
    """
    (
        field_map,
        candidate_tray,
        _,
    ) = build_test_problem()

    basis = ShimBasis(
        field_map=
            field_map,
        candidate_tray=
            candidate_tray,
    )

    basis.compute()

    index = 0

    B_basis_negative_T = (
        -basis.basis_T[
            :,
            index,
        ]
    )

    B_direct_negative_T = (
        direct_field_single_magnet(
            candidate_tray.candidates[
                index
            ],
            -1,
            field_map.sensors,
        )
    )

    stats = print_comparison(
        "TEST C — single negative-polarity magnet",
        B_direct_negative_T,
        B_basis_negative_T,
    )

    B_direct_positive_T = (
        direct_field_single_magnet(
            candidate_tray.candidates[
                index
            ],
            +1,
            field_map.sensors,
        )
    )

    sign_error_stats = (
        print_comparison(
            "TEST C2 — polarity sign symmetry: B(-P) vs -B(+P)",
            B_direct_negative_T,
            -B_direct_positive_T,
        )
    )

    assert stats[
        "passed"
    ], (
        "Negative direct magnet does not equal "
        "negative ShimBasis column."
    )

    assert sign_error_stats[
        "passed"
    ], (
        "Magpylib direct field does not show expected "
        "linear polarity sign reversal."
    )


def test_random_state_superposition():
    """
    TEST D:
    random {-1,0,+1} direct collection vs ShimBasis @ states.
    """
    (
        field_map,
        candidate_tray,
        states,
    ) = build_test_problem()

    basis = ShimBasis(
        field_map=
            field_map,
        candidate_tray=
            candidate_tray,
    )

    basis.compute()

    B_basis_T = (
        basis.field_from_states(
            states
        )
    )

    B_direct_T = (
        direct_field_for_states(
            candidate_tray.candidates,
            states,
            field_map.sensors,
        )
    )

    print(
        "\nRandom configuration"
    )
    print(
        f"  DSV points: "
        f"{len(field_map.B_T)}"
    )
    print(
        f"  Candidate magnets: "
        f"{len(candidate_tray.candidates)}"
    )
    print(
        f"  Active magnets: "
        f"{np.count_nonzero(states)}"
    )
    print(
        f"  Positive magnets: "
        f"{np.sum(states == 1)}"
    )
    print(
        f"  Negative magnets: "
        f"{np.sum(states == -1)}"
    )

    stats = print_comparison(
        "TEST D — random {-1,0,+1} configuration",
        B_direct_T,
        B_basis_T,
    )

    # Additional sign-error diagnostic.
    negative_mask = (
        states
        == -1
    )

    if np.any(
        negative_mask
    ):
        B_negative_positive_basis_T = (
            basis.basis_T[
                :,
                negative_mask,
            ]
            @ np.ones(
                np.sum(
                    negative_mask
                )
            )
        )

        predicted_if_negatives_were_not_flipped_T = (
            2.0
            * B_negative_positive_basis_T
        )

        sign_bug_stats = (
            comparison_statistics(
                predicted_if_negatives_were_not_flipped_T,
                B_basis_T
                - B_direct_T,
            )
        )

        print(
            "\nNegative-polarity mismatch diagnostic"
        )
        print(
            "  Compare actual basis-direct difference with "
            "2*sum(positive basis fields of negative-state magnets)"
        )
        print(
            f"  Max diagnostic error: "
            f"{sign_bug_stats['max_abs_error_T']:.6e} T"
        )

    assert stats[
        "passed"
    ], (
        "ShimBasis random-state superposition does not match "
        "direct Magpylib collection simulation."
    )


def test_internal_random_validation():
    """
    TEST E:
    exercise ShimBasis.validate() over several random chromosomes.
    """
    (
        field_map,
        candidate_tray,
        _,
    ) = build_test_problem()

    basis = ShimBasis(
        field_map=
            field_map,
        candidate_tray=
            candidate_tray,
    )

    basis.compute()

    report = basis.validate(
        n_tests=
            5,
        atol_T=
            ATOL_T,
        rtol=
            RTOL,
        random_seed=
            RANDOM_SEED + 1,
    )

    print(
        "\nTEST E — ShimBasis.validate()"
    )

    for result in report[
        "tests"
    ]:
        print(
            f"  Test {result['test']}: "
            f"active={result['n_active']}, "
            f"max_abs={result['max_abs_error_T']:.6e} T, "
            f"passed={result['passed']}"
        )

    assert report[
        "all_passed"
    ]


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print(
        "\n========================================"
    )
    print(
        "ShimBasis diagnostic validation"
    )
    print(
        "========================================"
    )

    test_single_positive_magnet()
    test_all_positive_magnets()
    test_single_negative_magnet()
    test_random_state_superposition()
    test_internal_random_validation()

    print(
        "\n========================================"
    )
    print(
        "All ShimBasis diagnostic tests passed."
    )
    print(
        "========================================\n"
    )
