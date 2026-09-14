"""
Four-magnet Gauss-meter benchmark jig + Magpylib simulation.

This script is a self-contained fabrication/validation benchmark for the
passive-shimming workflow.

Changes in this version
-----------------------
1. The center Gauss-probe holder is no longer floating.
   It is supported from the bottom plate by a printable vertical stem.

2. Magnet positions are true recessed slots/pockets, analogous to the shim tray:
   - slots open from the INNER faces of the two plates,
   - pocket depth = magnet thickness + clearance,
   - a back wall remains in each plate,
   - magnets are inserted from the central gap.

3. The requested 20 mm separation is interpreted as the CLEAR separation
   between the inward-facing magnet surfaces.

4. Simulation places the physical magnets at their actual seated locations and
   prints the cumulative field at exactly (0, 0, 0) after loading magnets
   1, 2, 3, and 4.

Dependencies
------------
pip install numpy trimesh manifold3d magpylib

Geometry convention
-------------------
Internal units: meters
STL export units: millimeters
"""

from pathlib import Path

import numpy as np
import trimesh
import magpylib as magpy


# ============================================================================
# USER SETTINGS
# ============================================================================

OUTPUT_DIR = Path("./benchmark_jig_output")
OUTPUT_STL = OUTPUT_DIR / "four_magnet_gauss_benchmark_jig.stl"

# ----------------------------------------------------------------------------
# Magnet
# ----------------------------------------------------------------------------

MAGNET_DIMS_M = np.array(
    [6.35e-3, 6.35e-3, 3.18e-3],
    dtype=float,
)

POLARIZATION_T = 1.20

# Loading/polarity configuration.
# +1 = +Z polarization
# -1 = -Z polarization
#  0 = do not load this position
MAGNET_STATES = np.array(
    [+1, +1, +1, +1],
    dtype=int,
)

# ----------------------------------------------------------------------------
# Experimental geometry
# ----------------------------------------------------------------------------

# Clear distance between the inward-facing surfaces of the magnets.
INNER_MAGNET_FACE_SEPARATION_M = 20.0e-3

# Two circular mounting plates.
PLATE_DIAMETER_M = 32.0e-3
PLATE_THICKNESS_M = 4.0e-3

# Two magnets per plate, symmetric about X=0.
SLOT_X_OFFSET_M = 5.0e-3

# Recessed magnet-slot clearances.
POCKET_XY_CLEARANCE_M = 0.15e-3
POCKET_DEPTH_CLEARANCE_M = 0.10e-3

# Two structural side pillars connecting the plates.
SUPPORT_WIDTH_M = 4.0e-3
SUPPORT_DEPTH_M = 5.0e-3
SUPPORT_X_OFFSET_M = 13.0e-3

# ----------------------------------------------------------------------------
# Gauss-meter probe holder
# ----------------------------------------------------------------------------

# Nominal probe opening.
PROBE_DIAMETER_M = 4.0e-3

# Outer diameter of the central sleeve.
PROBE_HOLDER_OD_M = 8.0e-3

# Sleeve length along Y.
PROBE_HOLDER_LENGTH_M = 8.0e-3

# Printable stem which supports the sleeve from the bottom plate.
PROBE_STEM_WIDTH_M = 6.0e-3
PROBE_STEM_DEPTH_M = 6.0e-3

# Extend the stem slightly into the sleeve; the probe bore is subtracted later.
PROBE_STEM_TOP_Z_M = -1.0e-3

# ----------------------------------------------------------------------------
# Boolean / export
# ----------------------------------------------------------------------------

BOOLEAN_ENGINE = "manifold"
STL_SCALE_TO_MM = 1000.0

SHOW_STL_ONLY = True
SHOW_JIG_WITH_MAGNETS = True
SHOW_MAGPY_CONFIGURATION = True


# ============================================================================
# DERIVED GEOMETRY
# ============================================================================

MAGNET_THICKNESS_M = float(MAGNET_DIMS_M[2])

# The two inward-facing physical magnet surfaces are at z = +/-10 mm.
INNER_FACE_HALF_GAP_M = INNER_MAGNET_FACE_SEPARATION_M / 2.0

BOTTOM_PLATE_INNER_Z_M = -INNER_FACE_HALF_GAP_M
TOP_PLATE_INNER_Z_M = +INNER_FACE_HALF_GAP_M

# Plates extend AWAY from the central measurement region.
BOTTOM_PLATE_CENTER_Z_M = (
    BOTTOM_PLATE_INNER_Z_M
    - PLATE_THICKNESS_M / 2.0
)

TOP_PLATE_CENTER_Z_M = (
    TOP_PLATE_INNER_Z_M
    + PLATE_THICKNESS_M / 2.0
)

# Physical magnet centers when seated flush with the inner plate faces.
BOTTOM_MAGNET_CENTER_Z_M = (
    BOTTOM_PLATE_INNER_Z_M
    - MAGNET_THICKNESS_M / 2.0
)

TOP_MAGNET_CENTER_Z_M = (
    TOP_PLATE_INNER_Z_M
    + MAGNET_THICKNESS_M / 2.0
)

MEASUREMENT_POINT_M = np.array(
    [0.0, 0.0, 0.0],
    dtype=float,
)

SLOT_ORDER = [
    (
        "bottom_left",
        np.array(
            [-SLOT_X_OFFSET_M, 0.0, BOTTOM_MAGNET_CENTER_Z_M]
        ),
    ),
    (
        "bottom_right",
        np.array(
            [+SLOT_X_OFFSET_M, 0.0, BOTTOM_MAGNET_CENTER_Z_M]
        ),
    ),
    (
        "top_left",
        np.array(
            [-SLOT_X_OFFSET_M, 0.0, TOP_MAGNET_CENTER_Z_M]
        ),
    ),
    (
        "top_right",
        np.array(
            [+SLOT_X_OFFSET_M, 0.0, TOP_MAGNET_CENTER_Z_M]
        ),
    ),
]


# ============================================================================
# MESH HELPERS
# ============================================================================

def _as_mesh(mesh_like):
    """Convert Trimesh/Scene Boolean output to one Trimesh."""

    if isinstance(mesh_like, trimesh.Trimesh):
        return mesh_like

    if isinstance(mesh_like, trimesh.Scene):
        geometries = [
            geometry
            for geometry in mesh_like.geometry.values()
        ]

        if not geometries:
            raise RuntimeError(
                "Boolean operation returned an empty Scene."
            )

        return trimesh.util.concatenate(
            geometries
        )

    raise TypeError(
        f"Unexpected mesh type: {type(mesh_like)}"
    )


def _clean_mesh(mesh):
    """Conservative cleanup after Boolean operations."""

    mesh = _as_mesh(
        mesh
    )

    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        mesh.merge_vertices()
    except Exception:
        pass

    try:
        trimesh.repair.fix_normals(
            mesh,
            multibody=True,
        )
    except Exception:
        pass

    return mesh


def _boolean_union(
    meshes,
    description,
):
    """Union independent valid solids."""

    result = trimesh.boolean.union(
        meshes,
        engine=BOOLEAN_ENGINE,
    )

    if result is None:
        raise RuntimeError(
            f"Boolean union returned None: {description}"
        )

    result = _clean_mesh(
        result
    )

    if not result.is_watertight:
        raise RuntimeError(
            f"Union produced non-watertight geometry: {description}"
        )

    if not result.is_volume:
        raise RuntimeError(
            f"Union produced non-volume geometry: {description}"
        )

    return result


def _boolean_difference(
    base,
    cutters,
    description,
):
    """
    Subtract all cutters in one Manifold Boolean call.

    The cutters remain independent closed volumes, matching the robust strategy
    used in the main passive-shim tray exporter.
    """

    if not cutters:
        return base

    base = _clean_mesh(
        base
    )

    cutters = [
        _clean_mesh(cutter)
        for cutter in cutters
    ]

    bad = [
        index
        for index, cutter in enumerate(cutters)
        if not cutter.is_volume
    ]

    if bad:
        raise RuntimeError(
            f"{description}: invalid cutter volumes at indices {bad}"
        )

    if not base.is_volume:
        raise RuntimeError(
            f"{description}: input base is not a valid volume."
        )

    result = trimesh.boolean.difference(
        [
            base,
            *cutters,
        ],
        engine=BOOLEAN_ENGINE,
    )

    if result is None:
        raise RuntimeError(
            f"Boolean difference returned None: {description}"
        )

    result = _clean_mesh(
        result
    )

    if not result.is_watertight:
        raise RuntimeError(
            f"{description} produced a non-watertight mesh."
        )

    if not result.is_volume:
        raise RuntimeError(
            f"{description} produced a non-volume mesh."
        )

    return result


def _cylinder_along_y(
    radius_m,
    length_m,
    center=(0.0, 0.0, 0.0),
    sections=96,
):
    """Create a cylinder whose long axis is Y."""

    mesh = trimesh.creation.cylinder(
        radius=float(radius_m),
        height=float(length_m),
        sections=int(sections),
    )

    rotation = (
        trimesh.transformations.rotation_matrix(
            np.pi / 2.0,
            [1.0, 0.0, 0.0],
        )
    )

    mesh.apply_transform(
        rotation
    )

    mesh.apply_translation(
        center
    )

    return mesh


# ============================================================================
# MAGNET SLOT GEOMETRY
# ============================================================================

def _magnet_slot_cutter(
    slot_position_m,
    side,
):
    """
    Create one recessed magnet slot opening toward the central gap.

    The pocket is slightly deeper and wider than the physical magnet.
    A finite back wall remains because PLATE_THICKNESS_M is larger than
    pocket depth.
    """

    depth = (
        MAGNET_THICKNESS_M
        + POCKET_DEPTH_CLEARANCE_M
    )

    if depth >= PLATE_THICKNESS_M:
        raise ValueError(
            "Pocket depth must be smaller than plate thickness so that "
            "a back wall remains."
        )

    dimensions = np.array(
        [
            MAGNET_DIMS_M[0] + POCKET_XY_CLEARANCE_M,
            MAGNET_DIMS_M[1] + POCKET_XY_CLEARANCE_M,
            depth + 0.10e-3,
        ],
        dtype=float,
    )

    cutter = trimesh.creation.box(
        extents=dimensions
    )

    x = float(
        slot_position_m[0]
    )

    y = float(
        slot_position_m[1]
    )

    # Cutter protrudes 0.05 mm through the opening face to avoid coplanar
    # Boolean ambiguity.
    overlap = 0.05e-3

    if side == "bottom":

        center_z = (
            BOTTOM_PLATE_INNER_Z_M
            - depth / 2.0
            + overlap
        )

    elif side == "top":

        center_z = (
            TOP_PLATE_INNER_Z_M
            + depth / 2.0
            - overlap
        )

    else:
        raise ValueError(
            "side must be 'bottom' or 'top'."
        )

    cutter.apply_translation(
        [x, y, center_z]
    )

    return cutter


# ============================================================================
# BUILD JIG
# ============================================================================

def build_jig():
    """
    Build the printable benchmark jig.

    Components:
      - bottom circular plate
      - top circular plate
      - two vertical side pillars
      - central Gauss-probe sleeve
      - vertical support stem from bottom plate to sleeve

    Cutouts:
      - four recessed magnet-loading slots
      - 4 mm diameter Gauss-probe bore
    """

    # ------------------------------------------------------------------------
    # Plates
    # ------------------------------------------------------------------------

    bottom_plate = trimesh.creation.cylinder(
        radius=
            PLATE_DIAMETER_M
            / 2.0,
        height=
            PLATE_THICKNESS_M,
        sections=160,
    )

    bottom_plate.apply_translation(
        [
            0.0,
            0.0,
            BOTTOM_PLATE_CENTER_Z_M,
        ]
    )

    top_plate = trimesh.creation.cylinder(
        radius=
            PLATE_DIAMETER_M
            / 2.0,
        height=
            PLATE_THICKNESS_M,
        sections=160,
    )

    top_plate.apply_translation(
        [
            0.0,
            0.0,
            TOP_PLATE_CENTER_Z_M,
        ]
    )

    # ------------------------------------------------------------------------
    # Side supports
    # ------------------------------------------------------------------------

    outer_bottom_z = (
        BOTTOM_PLATE_INNER_Z_M
        - PLATE_THICKNESS_M
    )

    outer_top_z = (
        TOP_PLATE_INNER_Z_M
        + PLATE_THICKNESS_M
    )

    support_height = (
        outer_top_z
        - outer_bottom_z
    )

    support_center_z = (
        outer_top_z
        + outer_bottom_z
    ) / 2.0

    side_supports = []

    for x in (
        -SUPPORT_X_OFFSET_M,
        +SUPPORT_X_OFFSET_M,
    ):

        support = trimesh.creation.box(
            extents=[
                SUPPORT_WIDTH_M,
                SUPPORT_DEPTH_M,
                support_height,
            ]
        )

        support.apply_translation(
            [
                x,
                0.0,
                support_center_z,
            ]
        )

        side_supports.append(
            support
        )

    # ------------------------------------------------------------------------
    # Central probe sleeve
    # ------------------------------------------------------------------------

    probe_holder_outer = _cylinder_along_y(
        radius_m=
            PROBE_HOLDER_OD_M
            / 2.0,
        length_m=
            PROBE_HOLDER_LENGTH_M,
        center=
            (0.0, 0.0, 0.0),
    )

    # ------------------------------------------------------------------------
    # Printable support stem
    #
    # The stem rises from the bottom plate's inner face and overlaps slightly
    # with the lower half of the cylindrical probe sleeve.
    # ------------------------------------------------------------------------

    stem_bottom_z = (
        BOTTOM_PLATE_INNER_Z_M
    )

    stem_top_z = float(
        PROBE_STEM_TOP_Z_M
    )

    if stem_top_z <= stem_bottom_z:
        raise ValueError(
            "PROBE_STEM_TOP_Z_M must lie above the bottom plate inner face."
        )

    stem_height = (
        stem_top_z
        - stem_bottom_z
    )

    stem_center_z = (
        stem_bottom_z
        + stem_top_z
    ) / 2.0

    probe_stem = trimesh.creation.box(
        extents=[
            PROBE_STEM_WIDTH_M,
            PROBE_STEM_DEPTH_M,
            stem_height,
        ]
    )

    probe_stem.apply_translation(
        [
            0.0,
            0.0,
            stem_center_z,
        ]
    )

    # ------------------------------------------------------------------------
    # Union structural body
    # ------------------------------------------------------------------------

    jig = _boolean_union(
        [
            bottom_plate,
            top_plate,
            *side_supports,
            probe_stem,
            probe_holder_outer,
        ],
        "benchmark jig structural body",
    )

    # ------------------------------------------------------------------------
    # Four recessed magnet slots
    # ------------------------------------------------------------------------

    pocket_cutters = []

    for name, position in SLOT_ORDER:

        side = (
            "bottom"
            if name.startswith("bottom")
            else "top"
        )

        pocket_cutters.append(
            _magnet_slot_cutter(
                slot_position_m=
                    position,
                side=
                    side,
            )
        )

    jig = _boolean_difference(
        base=
            jig,
        cutters=
            pocket_cutters,
        description=
            "four recessed magnet slots",
    )

    # ------------------------------------------------------------------------
    # Probe bore
    #
    # This cuts through both the sleeve and any stem overlap, ensuring an
    # unobstructed 4 mm probe channel centered exactly on (0,0,0).
    # ------------------------------------------------------------------------

    probe_bore = _cylinder_along_y(
        radius_m=
            PROBE_DIAMETER_M
            / 2.0,
        length_m=
            PROBE_HOLDER_LENGTH_M
            + 2.0e-3,
        center=
            (0.0, 0.0, 0.0),
    )

    jig = _boolean_difference(
        base=
            jig,
        cutters=[
            probe_bore,
        ],
        description=
            "Gauss-meter probe bore",
    )

    return _clean_mesh(
        jig
    )


# ============================================================================
# MAGNETIC SIMULATION
# ============================================================================

def _make_magnet(
    position_m,
    state,
):
    """Create one physical benchmark magnet."""

    state = int(
        state
    )

    if state not in (
        -1,
        +1,
    ):
        raise ValueError(
            "Installed magnet state must be -1 or +1."
        )

    return magpy.magnet.Cuboid(
        dimension=
            MAGNET_DIMS_M,
        position=
            np.asarray(
                position_m,
                dtype=float,
            ),
        polarization=
            np.array(
                [
                    0.0,
                    0.0,
                    state
                    * POLARIZATION_T,
                ],
                dtype=float,
            ),
    )


def simulate_loading():
    """
    Load magnets sequentially and evaluate the field at exactly (0,0,0).

    Prints both cumulative field and the incremental contribution of the newly
    loaded magnet.
    """

    active = magpy.Collection()

    previous_B_T = np.zeros(
        3,
        dtype=float,
    )

    rows = []

    print(
        "\n"
        + "=" * 82
    )

    print(
        "FOUR-MAGNET GAUSS-METER BENCHMARK"
    )

    print(
        "=" * 82
    )

    print(
        "Measurement point: "
        "(0.000, 0.000, 0.000) mm"
    )

    print(
        f"Clear magnet-face separation: "
        f"{INNER_MAGNET_FACE_SEPARATION_M*1e3:.3f} mm"
    )

    print(
        f"Bottom magnet center z: "
        f"{BOTTOM_MAGNET_CENTER_Z_M*1e3:.3f} mm"
    )

    print(
        f"Top magnet center z: "
        f"{TOP_MAGNET_CENTER_Z_M*1e3:.3f} mm"
    )

    print(
        f"Polarization magnitude: "
        f"{POLARIZATION_T:.3f} T"
    )

    print(
        "-" * 82
    )

    loaded_count = 0

    for slot_index, (
        (name, position),
        state,
    ) in enumerate(
        zip(
            SLOT_ORDER,
            MAGNET_STATES,
        ),
        start=1,
    ):

        state = int(
            state
        )

        if state == 0:

            print(
                f"Slot {slot_index}: {name} "
                "state=0 -> not loaded"
            )

            continue

        magnet = _make_magnet(
            position_m=
                position,
            state=
                state,
        )

        active.add(
            magnet
        )

        loaded_count += 1

        B_T = np.asarray(
            active.getB(
                MEASUREMENT_POINT_M
            ),
            dtype=float,
        ).reshape(
            3
        )

        delta_B_T = (
            B_T
            - previous_B_T
        )

        B_mag_T = float(
            np.linalg.norm(
                B_T
            )
        )

        row = {
            "loaded_count":
                loaded_count,
            "slot_number":
                slot_index,
            "slot_name":
                name,
            "state":
                state,
            "position_m":
                position.copy(),
            "Bx_T":
                float(
                    B_T[0]
                ),
            "By_T":
                float(
                    B_T[1]
                ),
            "Bz_T":
                float(
                    B_T[2]
                ),
            "Bmag_T":
                B_mag_T,
            "delta_Bx_T":
                float(
                    delta_B_T[0]
                ),
            "delta_By_T":
                float(
                    delta_B_T[1]
                ),
            "delta_Bz_T":
                float(
                    delta_B_T[2]
                ),
        }

        rows.append(
            row
        )

        print(
            f"\nLoad {loaded_count}: "
            f"{name} | polarity={state:+d}"
        )

        print(
            f"  magnet position [mm]: "
            f"{position*1e3}"
        )

        print(
            "  incremental contribution:"
        )

        print(
            f"    dBx = "
            f"{delta_B_T[0]*1e3:+.6f} mT"
        )

        print(
            f"    dBy = "
            f"{delta_B_T[1]*1e3:+.6f} mT"
        )

        print(
            f"    dBz = "
            f"{delta_B_T[2]*1e3:+.6f} mT "
            f"= {delta_B_T[2]*1e4:+.3f} G"
        )

        print(
            "  cumulative field at (0,0,0):"
        )

        print(
            f"    Bx  = "
            f"{B_T[0]*1e3:+.6f} mT"
        )

        print(
            f"    By  = "
            f"{B_T[1]*1e3:+.6f} mT"
        )

        print(
            f"    Bz  = "
            f"{B_T[2]*1e3:+.6f} mT "
            f"= {B_T[2]*1e4:+.3f} G"
        )

        print(
            f"    |B| = "
            f"{B_mag_T*1e3:.6f} mT "
            f"= {B_mag_T*1e4:.3f} G"
        )

        previous_B_T = (
            B_T.copy()
        )

    print(
        "\n"
        + "=" * 82
    )

    return (
        active,
        rows,
    )


# ============================================================================
# VISUALIZATION
# ============================================================================

def show_stl_only(
    jig,
):
    """Visualize only the printable jig, making the actual slots visible."""

    display_mesh = (
        jig.copy()
    )

    display_mesh.visual.face_colors = (
        [180, 180, 200, 255]
    )

    trimesh.Scene(
        {
            "benchmark_jig":
                display_mesh,
        }
    ).show()


def show_jig_with_magnets(
    jig,
):
    """
    Show the jig plus physical seated magnets and the exact measurement point.

    The magnets are visualization overlays only; they are not part of the STL.
    """

    scene = trimesh.Scene()

    jig_display = (
        jig.copy()
    )

    jig_display.visual.face_colors = (
        [180, 180, 200, 210]
    )

    scene.add_geometry(
        jig_display,
        node_name=
            "benchmark_jig",
    )

    for slot_index, (
        (name, position),
        state,
    ) in enumerate(
        zip(
            SLOT_ORDER,
            MAGNET_STATES,
        ),
        start=1,
    ):

        magnet_mesh = trimesh.creation.box(
            extents=
                MAGNET_DIMS_M,
        )

        magnet_mesh.apply_translation(
            position
        )

        if state > 0:
            magnet_mesh.visual.face_colors = (
                [220, 90, 90, 190]
            )
        elif state < 0:
            magnet_mesh.visual.face_colors = (
                [90, 90, 220, 190]
            )
        else:
            magnet_mesh.visual.face_colors = (
                [150, 150, 150, 80]
            )

        scene.add_geometry(
            magnet_mesh,
            node_name=
                f"{slot_index}_{name}",
        )

    center_marker = trimesh.creation.icosphere(
        subdivisions=
            2,
        radius=
            0.7e-3,
    )

    center_marker.visual.face_colors = (
        [50, 220, 50, 255]
    )

    scene.add_geometry(
        center_marker,
        node_name=
            "measurement_point",
    )

    scene.show()


def show_magpy_configuration(
    active_magnets,
):
    """Display the final magnet configuration and center sensor in Magpylib."""

    sensor = magpy.Sensor(
        position=
            MEASUREMENT_POINT_M,
    )

    magpy.show(
        active_magnets,
        sensor,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Fabrication jig
    # ------------------------------------------------------------------------

    jig = build_jig()

    print(
        "\nBenchmark jig geometry"
    )

    print(
        f"  Watertight: "
        f"{jig.is_watertight}"
    )

    print(
        f"  Volume: "
        f"{jig.is_volume}"
    )

    print(
        f"  Extents [mm]: "
        f"{jig.extents*1e3}"
    )

    pocket_depth = (
        MAGNET_THICKNESS_M
        + POCKET_DEPTH_CLEARANCE_M
    )

    back_wall = (
        PLATE_THICKNESS_M
        - pocket_depth
    )

    print(
        f"  Magnet slot depth: "
        f"{pocket_depth*1e3:.3f} mm"
    )

    print(
        f"  Remaining slot back wall: "
        f"{back_wall*1e3:.3f} mm"
    )

    print(
        f"  Probe bore diameter: "
        f"{PROBE_DIAMETER_M*1e3:.3f} mm"
    )

    if not jig.is_watertight:

        raise RuntimeError(
            "Benchmark jig is not watertight. "
            "Refusing STL export."
        )

    if not jig.is_volume:

        raise RuntimeError(
            "Benchmark jig is not a valid volume. "
            "Refusing STL export."
        )

    # Most slicers interpret STL coordinates as millimeters.
    export_mesh = (
        jig.copy()
    )

    export_mesh.apply_scale(
        STL_SCALE_TO_MM
    )

    export_mesh.export(
        OUTPUT_STL
    )

    # Reload exact exported STL as a final fabrication check.
    reloaded = trimesh.load(
        OUTPUT_STL,
        force="mesh",
    )

    print(
        f"  STL written: "
        f"{OUTPUT_STL}"
    )

    print(
        f"  Reloaded STL watertight: "
        f"{reloaded.is_watertight}"
    )

    print(
        f"  Reloaded STL volume: "
        f"{reloaded.is_volume}"
    )

    print(
        f"  Reloaded STL extents [mm]: "
        f"{reloaded.extents}"
    )

    # ------------------------------------------------------------------------
    # Magnetic benchmark
    # ------------------------------------------------------------------------

    (
        active_magnets,
        simulation_rows,
    ) = simulate_loading()

    # ------------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------------

    if SHOW_STL_ONLY:
        show_stl_only(
            jig
        )

    if SHOW_JIG_WITH_MAGNETS:
        show_jig_with_magnets(
            jig
        )

    if SHOW_MAGPY_CONFIGURATION:
        show_magpy_configuration(
            active_magnets
        )

    return {
        "jig":
            jig,
        "magnets":
            active_magnets,
        "simulation_rows":
            simulation_rows,
        "stl_path":
            OUTPUT_STL,
    }


if __name__ == "__main__":
    main()
