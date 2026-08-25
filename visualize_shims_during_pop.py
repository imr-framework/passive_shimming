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
# Filenames
#---------------------------------------------------------
disk_dia = 6 * 2.54 * 1e-2 # in meter
disk_thickness = 3.2 * 1e-3 # in meter
pkl_filename = './data/magnet_collection_shims_20260825.pkl'


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
# The holes are cut from a disk rather than exporting the
# magnetic cuboids themselves, so the resulting STL is the
# tray that has to be manufactured.
#---------------------------------------------------------


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

    Note:
        magnet.magnetization is defined in the magnet's LOCAL
        coordinate system. Magpylib's displayed arrow includes
        magnet.orientation, so the magnetization must first be
        transformed into GLOBAL coordinates before testing its
        Z direction.

    tray_side is retained in the function signature so that the
    rest of the code can remain unchanged, but it is intentionally
    NOT used to determine polarity.
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

    # Arrow pointing upward in Magpylib means global +Z
    is_positive = magnetization_global[2] > 0

    # ---------------------------------------------------------
    # NEGATIVE magnet:
    # no polarity engraving at all
    # ---------------------------------------------------------
    if not is_positive:
        return []

    # ---------------------------------------------------------
    # Positive magnet from here onward:
    # create a "+" engraving
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Place "+" just outside the magnet pocket
    # ---------------------------------------------------------
    magnet_dims = np.asarray(
        magnet.dimension,
        dtype=float
    )

    magnet_half_width = (
        max(magnet_dims[:2]) / 2
    )

    # Distance between magnet edge and '+' symbol
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
    # Symbol dimensions
    # ---------------------------------------------------------
    symbol_length = 0.0030      # 3.0 mm
    symbol_width = 0.0007       # 0.7 mm
    engraving_depth = 0.0006    # 0.6 mm

    # Put cutter into upper surface of tray.
    #
    # Tray extends:
    #   -disk_thickness/2 ... +disk_thickness/2
    #
    # The cutter overlaps the top surface by engraving_depth.
    cutter_z = (
        disk_thickness / 2
        - engraving_depth / 2
    )

    marks = []

    # ---------------------------------------------------------
    # Helper: make one rectangular engraving bar
    # ---------------------------------------------------------
    def make_bar(center_xy, direction, length):

        # Rectangle initially lies along local X axis
        bar = trimesh.creation.box(
            extents=[
                length,
                symbol_width,
                engraving_depth * 2
            ]
        )

        # Rotate bar into requested XY direction
        angle = np.arctan2(
            direction[1],
            direction[0]
        )

        rotation = (
            trimesh.transformations.rotation_matrix(
                angle,
                [0, 0, 1]
            )
        )

        bar.apply_transform(rotation)

        bar.apply_translation([
            center_xy[0],
            center_xy[1],
            cutter_z
        ])

        return bar

    # ---------------------------------------------------------
    # Positive polarity:
    # create BOTH bars of the "+"
    # ---------------------------------------------------------

    # Tangential bar
    marks.append(
        make_bar(
            symbol_center_xy,
            tangent,
            symbol_length
        )
    )

    # Radial bar
    marks.append(
        make_bar(
            symbol_center_xy,
            radial,
            symbol_length
        )
    )

    return marks


def _write_tray(
    collection,
    filename,
    tray_side
):

    tray = trimesh.creation.cylinder(
        radius=disk_dia / 2,
        height=disk_thickness,
        sections=128
    )

    cutters = []

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