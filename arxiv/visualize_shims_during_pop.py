import numpy as np
import magpylib as magpy
from make_shim_rings import make_shim_ring_template
from utils import get_field_pos, display_scatter_3D, get_magnetic_field, load_magnets_with_rot, filter_dsv, cost_fn, write2stl
from colorama import Style, Fore
from target_B0_2_shim_locations_rot import shimming_problem
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
import pickle
import time
import trimesh

# Changelog:
# 2024-08-21 - sairamgeethanath: Reflecting support for new fmr mapping axes - aligned with magpy
# 2026-08-21 - sairamgeethanath: Data variable is now an array of shape (N, 5) with columns x, y, z, B, V. The get_field_pos function has been updated to handle this new format.


#---------------------------------------------------------
# Tray geometry input
#---------------------------------------------------------
tray_shape = input(
    "Enter tray shape ('disc' or 'rectangle') [disc]: "
).strip().lower()

if tray_shape == "":
    tray_shape = "disc"

if tray_shape in ["disk", "circular", "circle"]:
    tray_shape = "disc"

if tray_shape in ["rect", "rectangular", "slab"]:
    tray_shape = "rectangle"

if tray_shape not in ["disc", "rectangle"]:
    raise ValueError("tray_shape must be 'disc' or 'rectangle'.")

#---------------------------------------------------------
# Filenames and tray dimensions
#---------------------------------------------------------
default_disc_dia_mm = 8 * 25.4          # 6 inch tray, in mm
default_tray_thickness_mm = 6.5         # 3.2 in mm

disk_thickness = default_tray_thickness_mm * 1e-3   # meters
# pkl_filename = './data/magnet_collection_shims_20260825.pkl'
pkl_filename = "./data/rectangle_magnet_collection_shims.pkl"

if tray_shape == "disc":
    disk_dia = default_disc_dia_mm * 1e-3           # meters
    tray_size_x = disk_dia
    tray_size_y = disk_dia

else:
    rect_x_mm_str = input(
        f"Enter rectangle X size in mm [{default_disc_dia_mm:.1f}]: "
    ).strip()
    rect_y_mm_str = input(
        f"Enter rectangle Y size in mm [{default_disc_dia_mm:.1f}]: "
    ).strip()

    rect_x_mm = float(rect_x_mm_str) if rect_x_mm_str else default_disc_dia_mm
    rect_y_mm = float(rect_y_mm_str) if rect_y_mm_str else default_disc_dia_mm

    tray_size_x = rect_x_mm * 1e-3
    tray_size_y = rect_y_mm * 1e-3

    # Keep disk_dia defined for compatibility, though unused for rectangle tray creation
    disk_dia = max(tray_size_x, tray_size_y)




#---------------------------------------------------------
# Visualization colors
# RGBA values: [R, G, B, alpha]
# These affect visualization only, not STL geometry.
#---------------------------------------------------------
TOP_TRAY_COLOR = [110, 180, 230, 255]       # light blue
BOTTOM_TRAY_COLOR = [240, 180, 90, 255]     # light orange


with open(pkl_filename, 'rb') as file:
    shim_rings_optimized_read = pickle.load(file)

shim_rings_optimized_read.show()


# Count the number of magnets in the optimized shim tray
num_magnets = len(shim_rings_optimized_read)
print(
    Fore.GREEN
    + 'Number of magnets in the optimized shim tray: '
    + str(num_magnets)
)

# Separate the magnets into two collections based on their z position
shim_rings_top = magpy.Collection(
    style_label='top_shims'
)

shim_rings_bottom = magpy.Collection(
    style_label='bottom_shims'
)

num_magnets = 0

for magnet in shim_rings_optimized_read:

    original_position = np.array(
        magnet.position,
        copy=True
    )

    stored_position = original_position.copy()
    stored_position[2] = 0

    cuboid = magpy.magnet.Cuboid(
        magnetization=np.array(
            magnet.magnetization,
            copy=True
        ),
        dimension=np.array(
            magnet.dimension,
            copy=True
        ),
        position=stored_position,
        orientation=magnet.orientation,
    )

    if original_position[2] > 0:
        shim_rings_top.add(cuboid)
        num_magnets += 1

    else:
        shim_rings_bottom.add(cuboid)
        num_magnets += 1


print(
    Fore.GREEN
    + 'Total number of magnets in the optimized shim tray: '
    + str(num_magnets)
)

print(
    Fore.GREEN
    + 'Number of magnets in the top shim ring: '
    + str(len(shim_rings_top))
)

print(
    Fore.GREEN
    + 'Number of magnets in the bottom shim ring: '
    + str(len(shim_rings_bottom))
)


#---------------------------------------------------------
# Visualize the top and bottom shim rings separately
#---------------------------------------------------------

# Set visualization colors for the magnetic cuboids.
# This does not alter position, orientation, dimensions, or magnetization.
for magnet in shim_rings_top:
    magnet.style.color = '#6EB4E6'

for magnet in shim_rings_bottom:
    magnet.style.color = '#F0B45A'


shim_rings_top.show()
shim_rings_bottom.show()


#---------------------------------------------------------
# Write the two trays as meshes.
#
# The holes are cut from a tray rather than exporting the
# magnetic cuboids themselves, so the resulting STL is the
# tray that has to be manufactured.
#---------------------------------------------------------


def _make_base_tray():
    if tray_shape == "disc":
        return trimesh.creation.cylinder(
            radius=disk_dia / 2,
            height=disk_thickness,
            sections=128
        )
    else:
        return trimesh.creation.box(
            extents=[
                tray_size_x,
                tray_size_y,
                disk_thickness
            ]
        )


def _engraving_z(engraving_depth):
    return (
        disk_thickness / 2
        - engraving_depth / 2
    )


def _make_bar_global(center_xy, angle, length, width, engraving_depth):
    bar = trimesh.creation.box(
        extents=[
            length,
            width,
            engraving_depth * 2
        ]
    )

    rotation = trimesh.transformations.rotation_matrix(
        angle,
        [0, 0, 1]
    )
    bar.apply_transform(rotation)

    bar.apply_translation([
        center_xy[0],
        center_xy[1],
        _engraving_z(engraving_depth)
    ])

    return bar


def _make_bar_local(label_center_xy, local_center_xy, angle_local, length, width, engraving_depth, label_rotation=0.0):
    c = np.cos(label_rotation)
    s = np.sin(label_rotation)

    rot2 = np.array([
        [c, -s],
        [s,  c]
    ])

    global_center_xy = (
        np.asarray(label_center_xy, dtype=float)
        + rot2 @ np.asarray(local_center_xy, dtype=float)
    )

    global_angle = label_rotation + angle_local

    return _make_bar_global(
        global_center_xy,
        global_angle,
        length,
        width,
        engraving_depth
    )


def _symbol_meshes(symbol, label_center_xy, char_center_xy, char_size, stroke_width, engraving_depth, label_rotation=0.0):
    meshes = []

    if symbol == '+':
        meshes.append(
            _make_bar_local(
                label_center_xy,
                char_center_xy,
                0.0,
                0.85 * char_size,
                stroke_width,
                engraving_depth,
                label_rotation
            )
        )
        meshes.append(
            _make_bar_local(
                label_center_xy,
                char_center_xy,
                np.pi / 2,
                0.85 * char_size,
                stroke_width,
                engraving_depth,
                label_rotation
            )
        )

    elif symbol == '-':
        meshes.append(
            _make_bar_local(
                label_center_xy,
                char_center_xy,
                0.0,
                0.85 * char_size,
                stroke_width,
                engraving_depth,
                label_rotation
            )
        )

    elif symbol == 'X':
        meshes.append(
            _make_bar_local(
                label_center_xy,
                char_center_xy,
                np.pi / 4,
                1.00 * char_size,
                stroke_width,
                engraving_depth,
                label_rotation
            )
        )
        meshes.append(
            _make_bar_local(
                label_center_xy,
                char_center_xy,
                -np.pi / 4,
                1.00 * char_size,
                stroke_width,
                engraving_depth,
                label_rotation
            )
        )

    elif symbol == 'Y':
        meshes.append(
            _make_bar_local(
                label_center_xy,
                char_center_xy + np.array([-0.18 * char_size, 0.16 * char_size]),
                -np.pi / 4,
                0.55 * char_size,
                stroke_width,
                engraving_depth,
                label_rotation
            )
        )
        meshes.append(
            _make_bar_local(
                label_center_xy,
                char_center_xy + np.array([0.18 * char_size, 0.16 * char_size]),
                np.pi / 4,
                0.55 * char_size,
                stroke_width,
                engraving_depth,
                label_rotation
            )
        )
        meshes.append(
            _make_bar_local(
                label_center_xy,
                char_center_xy + np.array([0.0, -0.16 * char_size]),
                np.pi / 2,
                0.65 * char_size,
                stroke_width,
                engraving_depth,
                label_rotation
            )
        )

    else:
        raise ValueError(f"Unsupported symbol '{symbol}'.")

    return meshes


def _label_meshes(label_text, center_xy, char_size, stroke_width, engraving_depth, rotation=0.0):
    meshes = []

    n = len(label_text)
    char_pitch = 1.20 * char_size

    x_offsets = (
        np.arange(n) - (n - 1) / 2
    ) * char_pitch

    for char, x_offset in zip(label_text, x_offsets):
        char_center_xy = np.array([x_offset, 0.0])
        meshes.extend(
            _symbol_meshes(
                char,
                np.asarray(center_xy, dtype=float),
                char_center_xy,
                char_size,
                stroke_width,
                engraving_depth,
                rotation
            )
        )

    return meshes


def _label_bbox(center_xy, label_text, char_size):
    """
    Approximate XY bounding box of a horizontal orientation label.
    Used only for collision detection.
    """
    char_pitch = 1.20 * char_size
    label_width = (
        (len(label_text) - 1) * char_pitch
        + char_size
    )
    label_height = char_size

    center_xy = np.asarray(center_xy, dtype=float)

    bbox_min = center_xy - np.array([
        label_width / 2,
        label_height / 2
    ])

    bbox_max = center_xy + np.array([
        label_width / 2,
        label_height / 2
    ])

    return bbox_min, bbox_max


def _bbox_overlap(
    bbox1_min,
    bbox1_max,
    bbox2_min,
    bbox2_max,
    clearance=0.0
):
    """
    Return True when two XY bounding boxes overlap.
    """
    return not (
        bbox1_max[0] + clearance < bbox2_min[0]
        or bbox1_min[0] - clearance > bbox2_max[0]
        or bbox1_max[1] + clearance < bbox2_min[1]
        or bbox1_min[1] - clearance > bbox2_max[1]
    )


def _label_inside_tray(
    center_xy,
    label_text,
    char_size,
    boundary_clearance=0.001
):
    """
    Check whether the entire label lies safely within the tray.
    """
    bbox_min, bbox_max = _label_bbox(
        center_xy,
        label_text,
        char_size
    )

    corners = np.array([
        [bbox_min[0], bbox_min[1]],
        [bbox_min[0], bbox_max[1]],
        [bbox_max[0], bbox_min[1]],
        [bbox_max[0], bbox_max[1]],
    ])

    if tray_shape == "disc":

        safe_radius = (
            disk_dia / 2
            - boundary_clearance
        )

        distances = np.linalg.norm(
            corners,
            axis=1
        )

        return np.all(
            distances <= safe_radius
        )

    else:

        safe_half_x = (
            tray_size_x / 2
            - boundary_clearance
        )

        safe_half_y = (
            tray_size_y / 2
            - boundary_clearance
        )

        return (
            np.all(
                np.abs(corners[:, 0])
                <= safe_half_x
            )
            and
            np.all(
                np.abs(corners[:, 1])
                <= safe_half_y
            )
        )


def _label_hits_magnet(
    center_xy,
    label_text,
    char_size,
    collection,
    clearance=0.0015
):
    """
    Return True if orientation label overlaps any magnet pocket.
    """
    label_min, label_max = _label_bbox(
        center_xy,
        label_text,
        char_size
    )

    for magnet in collection:

        # Use actual STL cutter bounds, including magnet orientation.
        cutter = _cuboid_mesh(magnet)

        magnet_min = cutter.bounds[0, :2]
        magnet_max = cutter.bounds[1, :2]

        if _bbox_overlap(
            label_min,
            label_max,
            magnet_min,
            magnet_max,
            clearance=clearance
        ):
            return True

    return False


def _find_orientation_label_position(
    label_text,
    direction,
    collection,
    char_size
):
    """
    Find a clear position near the requested scanner-axis endpoint.

    Search strategy:
      1. Start close to the physical tray boundary.
      2. Move inward only as needed.
      3. At each inward position, also search tangentially left/right
         so a nearby magnet does not block the label entirely.

    This keeps X+/X-/Y+/Y- near the ends of the tray while avoiding
    magnet pockets.
    """
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)

    # Tangential direction in the XY plane
    tangent = np.array([
        -direction[1],
         direction[0]
    ])

    if tray_shape == "disc":
        boundary_distance = disk_dia / 2
    else:
        if abs(direction[0]) > 0.5:
            boundary_distance = tray_size_x / 2
        else:
            boundary_distance = tray_size_y / 2

    # Search parameters
    initial_inset = 0.003     # start 3 mm from edge
    inward_step = 0.001       # 1 mm inward increments
    max_inset = 0.050         # allow up to 50 mm inward

    tangent_step = 0.002      # 2 mm sideways increments
    max_tangent = 0.030       # search up to +/-30 mm sideways

    inward_values = np.arange(
        initial_inset,
        max_inset + inward_step / 2,
        inward_step
    )

    tangent_magnitudes = np.arange(
        0.0,
        max_tangent + tangent_step / 2,
        tangent_step
    )

    for inset in inward_values:

        axis_center = (
            direction
            * (
                boundary_distance
                - inset
            )
        )

        # Try no tangential shift first, then +/- shifts.
        offsets = [0.0]

        for mag in tangent_magnitudes[1:]:
            offsets.extend([mag, -mag])

        for tangent_offset in offsets:

            candidate = (
                axis_center
                + tangent
                * tangent_offset
            )

            inside = _label_inside_tray(
                candidate,
                label_text,
                char_size
            )

            if not inside:
                continue

            collision = _label_hits_magnet(
                candidate,
                label_text,
                char_size,
                collection
            )

            if not collision:
                return candidate

    raise RuntimeError(
        f"Could not find a clear location for orientation label "
        f"{label_text} after searching inward and tangentially."
    )


def _orientation_marks(collection):
    """
    Create all four scanner-orientation engravings:

        +X end -> X+
        -X end -> X-
        +Y end -> Y+
        -Y end -> Y-

    Each label is placed as close as practical to its tray edge while
    avoiding overlap with magnet holes.
    """

    char_size = 0.0045          # 4.5 mm
    stroke_width = 0.0008       # 0.8 mm
    engraving_depth = 0.00085   # 0.85 mm

    marks = []

    labels = {
        "X+": np.array([+1.0,  0.0]),
        "X-": np.array([-1.0,  0.0]),
        "Y+": np.array([ 0.0, +1.0]),
        "Y-": np.array([ 0.0, -1.0]),
    }

    print("\nScanner orientation marks:")

    for label_text, direction in labels.items():

        center_xy = _find_orientation_label_position(
            label_text,
            direction,
            collection,
            char_size
        )

        print(
            f"  {label_text}: "
            f"x = {center_xy[0] * 1000:.1f} mm, "
            f"y = {center_xy[1] * 1000:.1f} mm"
        )

        marks.extend(
            _label_meshes(
                label_text,
                center_xy,
                char_size,
                stroke_width,
                engraving_depth,
                rotation=0.0
            )
        )

    return marks


def _cuboid_mesh(magnet):

    position = np.asarray(
        magnet.position,
        dtype=float
    )

    dimension = np.asarray(
        magnet.dimension,
        dtype=float
    ).copy()

    # Make cutter extend completely through tray.
    # Add some margin to avoid coplanar Boolean faces.
    dimension[2] = max(
        dimension[2],
        disk_thickness + 0.002
    )

    # Box is initially centered at (0,0,0)
    mesh = trimesh.creation.box(
        extents=dimension
    )

    # Rotate about its own center
    if magnet.orientation is not None:

        transform = np.eye(4)

        transform[:3, :3] = (
            magnet.orientation.as_matrix()
        )

        mesh.apply_transform(
            transform
        )

    # THEN move it to the magnet XY location
    mesh.apply_translation(
        [
            position[0],
            position[1],
            0.0
        ]
    )

    return mesh


def _polarity_marks(magnet, tray_side):
    """
    Return an engraved '+' marker ONLY for magnets whose
    magnetization arrow points in the global +Z direction.

    Convention:
        arrow points UP   (+Z) -> engrave "+"
        arrow points DOWN (-Z) -> no mark
    """

    position = np.asarray(
        magnet.position,
        dtype=float
    )

    # ---------------------------------------------------------
    # Determine GLOBAL magnetization direction
    # ---------------------------------------------------------
    magnetization_local = np.asarray(
        magnet.magnetization,
        dtype=float
    )

    if magnet.orientation is not None:
        magnetization_global = magnet.orientation.apply(
            magnetization_local
        )
    else:
        magnetization_global = magnetization_local.copy()

    is_positive = magnetization_global[2] > 0

    # Negative magnet -> no polarity engraving
    if not is_positive:
        return []

    # Determine radial direction from center of tray
    radial = position[:2].copy()
    radial_norm = np.linalg.norm(radial)

    if radial_norm > 0:
        radial /= radial_norm
    else:
        radial = np.array([1.0, 0.0])

    # Tangential direction, perpendicular to radial
    tangent = np.array([
        -radial[1],
        radial[0]
    ])

    # Place "+" just outside the magnet pocket
    magnet_dims = np.asarray(
        magnet.dimension,
        dtype=float
    )

    magnet_half_width = (
        max(magnet_dims[:2]) / 2
    )

    symbol_gap = 0.0020  # 2 mm

    symbol_center_xy = (
        position[:2]
        + radial
        * (
            magnet_half_width
            + symbol_gap
        )
    )

    # ---------------------------------------------------------
    # Deeper polarity mark: 0.60 mm + 0.25 mm = 0.85 mm
    # ---------------------------------------------------------
    symbol_length = 0.0030      # 3.0 mm
    symbol_width = 0.0007       # 0.7 mm
    engraving_depth = 0.00085   # 0.85 mm

    marks = []

    # Tangential bar
    marks.append(
        _make_bar_global(
            symbol_center_xy,
            np.arctan2(tangent[1], tangent[0]),
            symbol_length,
            symbol_width,
            engraving_depth
        )
    )

    # Radial bar
    marks.append(
        _make_bar_global(
            symbol_center_xy,
            np.arctan2(radial[1], radial[0]),
            symbol_length,
            symbol_width,
            engraving_depth
        )
    )

    return marks



def _alignment_notch():
    """
    Create a through-notch centered at the -Y edge of the tray.

    The notch is centered at:
        x = 0
        y = -ymax

    This provides a simple mechanical orientation reference so the
    tray can be aligned to the scanner with X centered and the notch
    identifying the -Y end.

    Geometry:
        semicircular edge notch
        diameter = 1.5 mm
        through the full tray thickness
    """

    notch_radius = 0.00075  # 0.75 mm radius = 1.5 mm diameter

    # Make the cutter taller than the tray to ensure a clean through-cut.
    notch_height = disk_thickness + 0.004

    if tray_shape == "disc":
        y_edge = -disk_dia / 2
    else:
        y_edge = -tray_size_y / 2

    notch = trimesh.creation.cylinder(
        radius=notch_radius,
        height=notch_height,
        sections=64
    )

    # Center the circular cutter exactly on the -Y boundary.
    # Half of the cutter lies inside the tray and half outside,
    # leaving a clear semicircular notch in the edge.
    notch.apply_translation([
        0.0,
        y_edge,
        0.0
    ])

    return notch


def _write_tray(
    collection,
    filename,
    tray_side
):

    tray = _make_base_tray()

    cutters = []

    # Add a mechanical alignment notch at x=0, y=-ymax.
    # This identifies the centered -Y end of the tray.
    cutters.append(
        _alignment_notch()
    )

    # Add orientation labels once per tray
    cutters.extend(
        _orientation_marks(collection)
    )

    for magnet in collection:

        cutters.append(
            _cuboid_mesh(
                magnet
            )
        )

        cutters.extend(
            _polarity_marks(
                magnet,
                tray_side
            )
        )

    for cutter in cutters:

        result = (
            trimesh.boolean.difference(
                [
                    tray,
                    cutter
                ],
                engine=None
            )
        )

        if result is not None:
            tray = result

    tray.export(
        filename
    )


#---------------------------------------------------------
# Write STL files
#---------------------------------------------------------

_write_tray(
    shim_rings_top,
    'shim_tray_top.stl',
    1
)

_write_tray(
    shim_rings_bottom,
    'shim_tray_bottom.stl',
    -1
)


#---------------------------------------------------------
# Reload the exported trays and display them
# for visual inspection.
#---------------------------------------------------------

top_tray_mesh = trimesh.load_mesh(
    'shim_tray_top.stl'
)

bottom_tray_mesh = trimesh.load_mesh(
    'shim_tray_bottom.stl'
)


#---------------------------------------------------------
# Apply visualization colors.
#
# These colors exist only in the Trimesh viewer.
# They do NOT change the STL geometry.
#---------------------------------------------------------

top_tray_mesh.visual.face_colors = (
    TOP_TRAY_COLOR
)

bottom_tray_mesh.visual.face_colors = (
    BOTTOM_TRAY_COLOR
)


#---------------------------------------------------------
# Display trays
#---------------------------------------------------------

trimesh.Scene(
    {
        'top_tray':
            top_tray_mesh
    }
).show()


trimesh.Scene(
    {
        'bottom_tray':
            bottom_tray_mesh
    }
).show()