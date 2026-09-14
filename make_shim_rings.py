"""
Passive-shim candidate geometry.

Exports
-------
external_stl_tray
    Loads and standardizes one external shim-tray STL, mirrors it for the
    opposite pole piece, and provides an XY footprint for candidate clipping.

shim_ring
    Generates concentric-ring candidate Cuboids using Magpylib polarization.

Units
-----
All program-side geometry is SI (meters, tesla). Raw STL units are converted
according to ``stl_units``.

Key conventions
---------------
- Magpylib ``polarization`` only; no magnetization pathway.
- Default shim polarization is 1.2 T in ``shim_ring``.
- Full-ring candidate count uses ``2*pi*r``.
- In external-STL mode, ``diameter=None`` lets the STL footprint determine
  the radial search extent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import magpylib as magpy
import trimesh

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


class external_stl_tray:
    """
    Prepare one externally designed physical shim-tray STL.

    The supplied STL is interpreted as ONE physical tray. It is converted to
    meters, reoriented so the tray footprint lies in XY and thickness lies
    along Z, centered in XY, and mirrored to produce the opposite tray.

    The standardized reference/bottom tray has its INNER surface at z=0 and
    its body extends toward negative Z. The mirrored top tray has its body
    extending toward positive Z. Their inner faces are then placed at
    ``-separation/2`` and ``+separation/2``.

    Parameters
    ----------
    stl_file : str, pathlib.Path, or trimesh.Trimesh
        Physical shim-tray STL.
    separation : float
        Distance between bottom and top INNER tray surfaces, in meters.
    stl_units : {'mm', 'm', 'inch'}, optional
        Units of raw STL coordinates. Default 'mm'.
    margin : float, optional
        Optional inward margin from the usable XY boundary, in meters.
    show : bool, optional
        Display the mirrored physical tray pair when True.
    debug : bool, optional
        Print detailed geometry diagnostics when True.
    """

    _UNIT_SCALE = {
        "mm": 1e-3,
        "millimeter": 1e-3,
        "millimeters": 1e-3,
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "in": 0.0254,
        "inch": 0.0254,
        "inches": 0.0254,
    }

    def __init__(
        self,
        stl_file,
        separation,
        stl_units="mm",
        margin=0.0,
        show=False,
        debug=False,
    ):
        self.stl_file = stl_file
        self.separation = float(separation)
        self.stl_units = str(stl_units)
        self.margin = float(margin)
        self.debug = bool(debug)

        if self.separation <= 0:
            raise ValueError("separation must be positive and in meters.")
        if self.margin < 0:
            raise ValueError("margin must be >= 0 and in meters.")

        self.original_mesh = None
        self.reference_tray = None
        self.bottom_tray = None
        self.top_tray = None
        self.allowed_xy_region = None
        self.original_thickness_axis = None
        self.thickness = None
        self.heights = None

        self._prepare()

        if show:
            self.show()

    @classmethod
    def _scale_to_meters(cls, units):
        key = str(units).strip().lower()
        if key not in cls._UNIT_SCALE:
            raise ValueError("stl_units must be 'mm', 'm', or 'inch'.")
        return cls._UNIT_SCALE[key]

    def _load_mesh(self):
        if isinstance(self.stl_file, trimesh.Trimesh):
            mesh = self.stl_file.copy()
        else:
            path = Path(self.stl_file)
            if not path.exists():
                raise FileNotFoundError(f"STL file not found: {path}")
            mesh = trimesh.load(path, force="mesh", process=True)

        if isinstance(mesh, trimesh.Scene):
            if not mesh.geometry:
                raise ValueError("The supplied STL contains no geometry.")
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Could not interpret stl_file as a Trimesh mesh.")

        if len(mesh.vertices) == 0:
            raise ValueError("The supplied STL contains no vertices.")

        mesh.apply_scale(self._scale_to_meters(self.stl_units))
        self.original_mesh = mesh.copy()

        if self.debug:
            print("\nExternal STL loaded")
            print(f"  Extents [mm]: {mesh.extents * 1e3}")
            print(f"  Bounds [mm]:\n{mesh.bounds * 1e3}")
            print(f"  Watertight: {mesh.is_watertight}")

        return mesh

    @staticmethod
    def _rotation_axis_to_z(axis_index):
        """Return transform mapping X, Y, or Z thickness axis onto +Z."""
        if axis_index == 2:
            return np.eye(4)
        if axis_index == 1:
            return trimesh.transformations.rotation_matrix(
                np.pi / 2.0, [1.0, 0.0, 0.0]
            )
        if axis_index == 0:
            return trimesh.transformations.rotation_matrix(
                -np.pi / 2.0, [0.0, 1.0, 0.0]
            )
        raise ValueError("axis_index must be 0, 1, or 2.")

    def _standardize_orientation(self, mesh):
        """
        Detect the thinnest STL extent, map it to Z, center XY, and put the
        reference tray's inner (+Z-most) surface at z=0.
        """
        extents = np.asarray(mesh.extents, dtype=float)
        axis = int(np.argmin(extents))
        self.original_thickness_axis = axis

        mesh.apply_transform(self._rotation_axis_to_z(axis))

        center = mesh.bounds.mean(axis=0)
        mesh.apply_translation([-center[0], -center[1], 0.0])

        z_inner = float(mesh.bounds[1, 2])
        mesh.apply_translation([0.0, 0.0, -z_inner])

        self.thickness = float(mesh.bounds[1, 2] - mesh.bounds[0, 2])

        if self.debug:
            axis_name = ("X", "Y", "Z")[axis]
            print("\nExternal STL standardized")
            print(f"  Original thickness axis: {axis_name}")
            print(f"  Standardized extents [mm]: {mesh.extents * 1e3}")
            print(f"  Thickness [mm]: {self.thickness * 1e3:.3f}")
            print(f"  Reference z bounds [mm]: {mesh.bounds[:, 2] * 1e3}")

        return mesh

    def _extract_xy_region(self, mesh):
        """
        Extract usable XY footprint.

        A mid-thickness section is preferred because it preserves concavity.
        If section reconstruction fails, use the projected XY convex hull.
        """
        z_mid = 0.5 * (mesh.bounds[0, 2] + mesh.bounds[1, 2])

        section = mesh.section(
            plane_origin=[0.0, 0.0, z_mid],
            plane_normal=[0.0, 0.0, 1.0],
        )

        polygons = []

        if section is not None:
            try:
                for loop in section.discrete:
                    loop = np.asarray(loop, dtype=float)
                    if loop.ndim == 2 and loop.shape[0] >= 3:
                        polygon = Polygon(loop[:, :2])
                        if polygon.is_valid and polygon.area > 0:
                            polygons.append(polygon)
            except Exception:
                polygons = []

        if polygons:
            region = unary_union(polygons)
            if not region.is_empty:
                return region

        if self.debug:
            print(
                "  WARNING: clean STL section unavailable; "
                "using projected XY convex hull."
            )

        return Polygon(mesh.vertices[:, :2]).convex_hull

    def _create_mirrored_pair(self, reference_tray):
        bottom = reference_tray.copy()
        top = reference_tray.copy()

        mirror_z = np.eye(4)
        mirror_z[2, 2] = -1.0
        top.apply_transform(mirror_z)

        try:
            top.fix_normals()
        except Exception:
            pass

        bottom.apply_translation([0.0, 0.0, -self.separation / 2.0])
        top.apply_translation([0.0, 0.0, +self.separation / 2.0])

        bottom.visual.face_colors = [200, 200, 200, 255]
        top.visual.face_colors = [170, 170, 170, 255]

        return bottom, top

    def _prepare(self):
        mesh = self._load_mesh()
        self.reference_tray = self._standardize_orientation(mesh)

        self.allowed_xy_region = self._extract_xy_region(
            self.reference_tray
        )

        if self.margin > 0:
            self.allowed_xy_region = self.allowed_xy_region.buffer(
                -self.margin
            )
            if self.allowed_xy_region.is_empty:
                raise ValueError(
                    "margin is too large; usable STL footprint became empty."
                )

        self.bottom_tray, self.top_tray = self._create_mirrored_pair(
            self.reference_tray
        )

        self.heights = [
            -self.separation / 2.0,
            +self.separation / 2.0,
        ]

        if self.debug:
            print("\nExternal STL tray prepared")
            print(
                f"  Candidate heights [mm]: "
                f"{np.asarray(self.heights) * 1e3}"
            )
            print(
                f"  Usable XY bounds [mm]: "
                f"{np.asarray(self.allowed_xy_region.bounds) * 1e3}"
            )
            print(
                f"  Usable area [mm^2]: "
                f"{self.allowed_xy_region.area * 1e6:.3f}"
            )

    @staticmethod
    def _candidate_xy_footprint(cube):
        dims = np.asarray(cube.dimension, dtype=float)
        hx = dims[0] / 2.0
        hy = dims[1] / 2.0

        corners = np.array(
            [
                [-hx, -hy, 0.0],
                [+hx, -hy, 0.0],
                [+hx, +hy, 0.0],
                [-hx, +hy, 0.0],
            ]
        )

        if cube.orientation is not None:
            corners = cube.orientation.apply(corners)

        corners = corners + np.asarray(cube.position, dtype=float)

        return Polygon(corners[:, :2])

    def contains_candidate(
        self,
        cube,
        require_full_magnet_inside=True,
    ):
        """Return True if the candidate lies inside the usable STL footprint."""
        if require_full_magnet_inside:
            return bool(
                self.allowed_xy_region.covers(
                    self._candidate_xy_footprint(cube)
                )
            )

        center = Point(
            float(cube.position[0]),
            float(cube.position[1]),
        )
        return bool(self.allowed_xy_region.covers(center))

    def statistics(self):
        """Return lightweight external-tray geometry diagnostics."""
        return {
            "stl_file": str(self.stl_file),
            "stl_units": self.stl_units,
            "separation_m": self.separation,
            "margin_m": self.margin,
            "heights_m": list(self.heights),
            "thickness_m": self.thickness,
            "original_thickness_axis": self.original_thickness_axis,
            "allowed_xy_bounds_m": list(self.allowed_xy_region.bounds),
            "allowed_xy_area_m2": float(self.allowed_xy_region.area),
        }

    def show(self):
        """Display standardized bottom/top physical tray pair."""
        trimesh.Scene(
            {
                "bottom_shim_tray": self.bottom_tray,
                "top_shim_tray": self.top_tray,
            }
        ).show()




class shim_ring:
    """
    Generate candidate permanent-magnet shim locations on concentric rings.

    Parameters
    ----------
    diameter : float or None
        Circular tray diameter in meters. Required when no external STL tray
        is supplied. In external-STL mode, set ``diameter=None`` to let the
        STL footprint determine the radial search extent.
    magnet_dims : array-like of float
        Magnet dimensions ``(dx, dy, dz)`` in meters.
    height : float
        Z coordinate of this shim plane in meters.
    num_magnets : int or None, optional
        Optional hard cap on accepted candidates. ``None`` means no cap.
    symmetry : bool, optional
        If True, retain only candidates in the +X,+Y quadrant. Candidate
        density is still computed from the full 360-degree circumference.
    skip_center_magnets : bool, optional
        Skip the central +/-dx/2 pair when True.
    style_color : str, optional
        Magpylib visualization color.
    polarization_T : float, optional
        Remanent polarization magnitude in tesla. Default is 1.2 T, a useful
        nominal value for N45 NdFeB. Can be changed after construction with
        ``ring.polarization_T = new_value``.
    radial_spacing_factor : float, optional
        Radial ring spacing multiplier applied to the XY magnet diagonal.
        Default 1.0.
    azimuthal_spacing_factor : float, optional
        Arc-length spacing multiplier applied to ``max(dx, dy)``. Default
        1.25, preserving the earlier spacing choice while fixing the former
        half-circumference bug.
    external_tray : external_stl_tray or None, optional
        Prepared external STL tray object.
    external_stl_file : str or None, optional
        Convenience STL path. If used, ``external_stl_separation`` is required.
    external_stl_separation : float or None, optional
        Inner-surface separation in meters when constructing an external tray.
    stl_units : {'mm', 'm', 'inch'}, optional
        STL coordinate units. Program-side geometry remains SI meters.
    stl_margin : float, optional
        Optional inward external-STL margin in meters.
    require_full_magnet_inside : bool, optional
        If True, all four rotated XY magnet corners must lie inside the tray.

    Notes
    -----
    * Magpylib ``polarization`` is used exclusively; there is no
      ``magnetization`` pathway in this class.
    * Nonzero ring population uses the full circumference ``2*pi*r``.
    * ``compute_area_metrics`` reports tray area, magnet footprint area,
      accepted candidate count, a simple area-only upper bound, and candidate
      footprint utilization.
    * All program-side dimensions are SI meters.
    """

    DEFAULT_POLARIZATION_T = 1.2

    def __init__(
        self,
        diameter,
        magnet_dims,
        height,
        num_magnets=None,
        symmetry=True,
        skip_center_magnets=False,
        style_color='red',
        polarization_T=DEFAULT_POLARIZATION_T,
        radial_spacing_factor=1.0,
        azimuthal_spacing_factor=1.25,
        external_tray=None,
        external_stl_file=None,
        external_stl_separation=None,
        stl_units='mm',
        stl_margin=0.0,
        require_full_magnet_inside=True,
    ):
        self.diameter = None if diameter is None else float(diameter)
        self.magnet_dims = np.asarray(magnet_dims, dtype=float)
        if self.magnet_dims.shape != (3,):
            raise ValueError("magnet_dims must be (dx, dy, dz) in meters.")
        if np.any(self.magnet_dims <= 0):
            raise ValueError("All magnet dimensions must be positive.")
        if self.diameter is not None and self.diameter <= 0:
            raise ValueError("diameter must be positive or None.")

        self.height = float(height)
        self.num_magnets = None if num_magnets is None else int(num_magnets)
        if self.num_magnets is not None and self.num_magnets <= 0:
            raise ValueError("num_magnets must be positive or None.")

        self.symmetry = bool(symmetry)
        self.skip_center_magnets = bool(skip_center_magnets)
        self.style_color = style_color
        self.radial_spacing_factor = float(radial_spacing_factor)
        self.azimuthal_spacing_factor = float(azimuthal_spacing_factor)
        if self.radial_spacing_factor <= 0 or self.azimuthal_spacing_factor <= 0:
            raise ValueError("Spacing factors must be > 0.")

        self.require_full_magnet_inside = bool(require_full_magnet_inside)
        self.display_collection = False
        self.collection = None
        self.area_report = None
        self.num_rings = None
        self.max_center_radius = None

        self._polarization_T = None
        self.polarization_T = polarization_T

        self.external_tray = external_tray
        if self.external_tray is None and external_stl_file is not None:
            if external_stl_separation is None:
                raise ValueError(
                    "external_stl_separation is required with external_stl_file."
                )
            self.external_stl(
                external_stl_file,
                separation=external_stl_separation,
                stl_units=stl_units,
                stl_margin=stl_margin,
            )

        if self.external_tray is None and self.diameter is None:
            raise ValueError("Supply either diameter or an external tray/STL.")

    @property
    def polarization_T(self):
        """Remanent polarization magnitude in tesla."""
        return self._polarization_T

    @polarization_T.setter
    def polarization_T(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError("polarization_T must be positive.")
        self._polarization_T = value

    @property
    def polarization(self):
        """Magpylib polarization vector ``[0, 0, polarization_T]`` in tesla."""
        return np.array([0.0, 0.0, self.polarization_T], dtype=float)

    def external_stl(
        self,
        stl_file,
        separation,
        stl_units='mm',
        stl_margin=0.0,
        show=False,
    ):
        """Attach an ``external_stl_tray`` prepared from one shim-tray STL."""
        self.external_tray = external_stl_tray(
            stl_file=stl_file,
            separation=separation,
            stl_units=stl_units,
            margin=stl_margin,
            show=show,
        )
        return self.external_tray

    @property
    def magnet_footprint_area(self):
        """XY footprint area of one magnet in square meters."""
        return float(self.magnet_dims[0] * self.magnet_dims[1])

    @property
    def magnet_in_plane_diagonal(self):
        """XY diagonal of one magnet in meters."""
        return float(np.hypot(self.magnet_dims[0], self.magnet_dims[1]))

    @property
    def radial_pitch(self):
        """Center-to-center radial ring pitch in meters."""
        return float(self.radial_spacing_factor * self.magnet_in_plane_diagonal)

    @property
    def azimuthal_pitch(self):
        """Target center-to-center arc pitch along a ring in meters."""
        return float(
            self.azimuthal_spacing_factor * max(self.magnet_dims[0], self.magnet_dims[1])
        )

    def _disc_max_center_radius(self):
        if self.diameter is None:
            return np.inf
        return max(
            0.0,
            self.diameter / 2.0 - self.magnet_in_plane_diagonal / 2.0,
        )

    def _external_max_center_radius(self):
        if self.external_tray is None:
            return np.inf
        min_x, min_y, max_x, max_y = self.external_tray.allowed_xy_region.bounds
        return float(
            np.hypot(
                max(abs(min_x), abs(max_x)),
                max(abs(min_y), abs(max_y)),
            )
        )

    def get_geometry_params(self):
        """
        Compute the usable radial extent and number of concentric rings.

        In external-STL mode, the STL footprint determines the search extent.
        If ``diameter`` is also supplied, the smaller limit is used.
        """
        self.max_center_radius = float(
            min(self._disc_max_center_radius(), self._external_max_center_radius())
        )
        if not np.isfinite(self.max_center_radius):
            raise RuntimeError("Could not determine a finite search radius.")

        self.num_rings = (
            int(np.floor(self.max_center_radius / self.radial_pitch)) + 1
            if self.max_center_radius > 0
            else 1
        )

        print("\nShim-ring geometry:")
        print(f"  Magnet dimensions [mm]: {self.magnet_dims * 1e3}")
        print(f"  Polarization: {self.polarization_T:.3f} T")
        print(f"  Radial pitch: {self.radial_pitch * 1e3:.3f} mm")
        print(f"  Azimuthal target pitch: {self.azimuthal_pitch * 1e3:.3f} mm")
        print(f"  Maximum candidate-center radius: {self.max_center_radius * 1e3:.3f} mm")
        print(f"  Radial levels including center: {self.num_rings}")

        return {
            'diameter_m': self.diameter,
            'magnet_dims_m': self.magnet_dims.copy(),
            'polarization_T': self.polarization_T,
            'radial_pitch_m': self.radial_pitch,
            'azimuthal_pitch_m': self.azimuthal_pitch,
            'max_center_radius_m': self.max_center_radius,
            'num_rings': self.num_rings,
        }

    def _passes_symmetry(self, cube):
        if not self.symmetry:
            return True
        p = np.asarray(cube.position, dtype=float)
        return p[0] >= 0 and p[1] >= 0

    def _candidate_xy_corners(self, cube):
        dims = np.asarray(cube.dimension, dtype=float)
        hx, hy = dims[0] / 2.0, dims[1] / 2.0
        corners = np.array([
            [-hx, -hy, 0.0],
            [+hx, -hy, 0.0],
            [+hx, +hy, 0.0],
            [-hx, +hy, 0.0],
        ])
        if cube.orientation is not None:
            corners = cube.orientation.apply(corners)
        return corners + np.asarray(cube.position, dtype=float)

    def _candidate_inside_disc(self, cube):
        if self.diameter is None:
            return True
        radius = self.diameter / 2.0
        corners = self._candidate_xy_corners(cube)
        return bool(np.all(np.linalg.norm(corners[:, :2], axis=1) <= radius))

    def _inside_external_tray(self, cube):
        if self.external_tray is None:
            return True
        return bool(
            self.external_tray.contains_candidate(
                cube,
                require_full_magnet_inside=self.require_full_magnet_inside,
            )
        )

    def _candidate_is_allowed(self, cube):
        if not self._passes_symmetry(cube):
            return False, 'symmetry'
        if not self._candidate_inside_disc(cube):
            return False, 'disc'
        if not self._inside_external_tray(cube):
            return False, 'external_stl'
        return True, None

    def _new_cube(self, position, angle_deg=0.0, rotate_around_origin=False):
        cube = magpy.magnet.Cuboid(
            dimension=self.magnet_dims,
            position=position,
            polarization=self.polarization,
            style_color=self.style_color,
        )
        if rotate_around_origin:
            cube.rotate_from_angax(angle_deg, 'z', anchor=0)
        if angle_deg != 0:
            cube.rotate_from_angax(angle_deg, 'z')
        return cube

    def _ring_candidate_count(self, radius):
        """Return the full-360-degree candidate count using ``2*pi*r``."""
        if radius <= 0:
            return 0
        circumference = 2.0 * np.pi * radius
        return max(1, int(np.floor(circumference / self.azimuthal_pitch)))

    def tray_area(self):
        """Usable tray footprint area in square meters."""
        if self.external_tray is not None:
            return float(self.external_tray.allowed_xy_region.area)
        if self.diameter is None:
            raise RuntimeError("Tray area requires diameter or external tray.")
        return float(np.pi * (self.diameter / 2.0) ** 2)

    def compute_area_metrics(self, accepted_count=None):
        """
        Report simple area-efficiency diagnostics for the candidate lattice.

        The area-limited capacity is ``floor(tray_area/magnet_area)`` and is
        only an upper bound; it ignores packing inefficiency and edge effects.
        """
        if accepted_count is None:
            if self.collection is None:
                raise RuntimeError("Build the collection first or pass accepted_count.")
            accepted_count = len(self.collection)

        tray_area = self.tray_area()
        magnet_area = self.magnet_footprint_area
        area_limited_capacity = int(np.floor(tray_area / magnet_area))
        area_fraction = accepted_count * magnet_area / tray_area
        capacity_fraction = (
            accepted_count / area_limited_capacity if area_limited_capacity > 0 else np.nan
        )

        self.area_report = {
            'tray_area_m2': tray_area,
            'tray_area_mm2': tray_area * 1e6,
            'magnet_area_m2': magnet_area,
            'magnet_area_mm2': magnet_area * 1e6,
            'accepted_candidates': int(accepted_count),
            'area_limited_capacity': area_limited_capacity,
            'candidate_area_fraction': area_fraction,
            'candidate_area_percent': 100.0 * area_fraction,
            'candidate_vs_area_limit_percent': 100.0 * capacity_fraction,
        }

        print("\nShim-tray area / candidate-density report:")
        print(f"  Usable tray area: {self.area_report['tray_area_mm2']:.1f} mm^2")
        print(f"  Magnet footprint area: {self.area_report['magnet_area_mm2']:.3f} mm^2")
        print(f"  Area-only theoretical capacity: {area_limited_capacity}")
        print(f"  Accepted candidate magnets: {accepted_count}")
        print(f"  Candidate footprint / tray area: {self.area_report['candidate_area_percent']:.1f}%")
        print(
            "  Candidate count / area-only upper bound: "
            f"{self.area_report['candidate_vs_area_limit_percent']:.1f}%"
        )
        return self.area_report

    def make_magnet_collection(self):
        """Generate and return the Magpylib candidate collection."""
        if self.num_rings is None:
            self.get_geometry_params()

        collection = magpy.Collection()
        stats = {
            'generated': 0,
            'accepted': 0,
            'rejected_symmetry': 0,
            'rejected_disc': 0,
            'rejected_external_stl': 0,
            'stopped_by_num_magnets_cap': False,
        }

        def try_add(cube):
            stats['generated'] += 1
            allowed, reason = self._candidate_is_allowed(cube)
            if not allowed:
                if reason == 'symmetry':
                    stats['rejected_symmetry'] += 1
                elif reason == 'disc':
                    stats['rejected_disc'] += 1
                elif reason == 'external_stl':
                    stats['rejected_external_stl'] += 1
                return False

            if self.num_magnets is not None and stats['accepted'] >= self.num_magnets:
                stats['stopped_by_num_magnets_cap'] = True
                return False

            collection.add(cube)
            stats['accepted'] += 1
            return True

        if not self.skip_center_magnets:
            for x_position in (+0.5 * self.magnet_dims[0], -0.5 * self.magnet_dims[0]):
                try_add(
                    self._new_cube(
                        position=(x_position, 0.0, self.height)
                    )
                )

        for ring_index in range(1, self.num_rings):
            if self.num_magnets is not None and stats['accepted'] >= self.num_magnets:
                stats['stopped_by_num_magnets_cap'] = True
                break

            radius = ring_index * self.radial_pitch
            if radius > self.max_center_radius + 1e-12:
                break

            n_ring = self._ring_candidate_count(radius)
            angles = np.linspace(0.0, 360.0, n_ring, endpoint=False)
            accepted_before = stats['accepted']

            for angle in angles:
                if self.num_magnets is not None and stats['accepted'] >= self.num_magnets:
                    stats['stopped_by_num_magnets_cap'] = True
                    break

                cube = self._new_cube(
                    position=(radius, 0.0, self.height),
                    angle_deg=float(angle),
                    rotate_around_origin=True,
                )
                try_add(cube)

            print(
                f"Ring {ring_index}: radius={radius*1e3:.2f} mm | "
                f"full-circle candidates={n_ring} | "
                f"accepted={stats['accepted']-accepted_before}"
            )

        self.collection = collection

        print("\nShim candidate summary:")
        print(f"  Generated: {stats['generated']}")
        print(f"  Accepted: {stats['accepted']}")
        print(f"  Rejected by symmetry: {stats['rejected_symmetry']}")
        print(f"  Rejected by circular boundary: {stats['rejected_disc']}")
        print(f"  Rejected outside external STL: {stats['rejected_external_stl']}")
        if stats['stopped_by_num_magnets_cap']:
            print(f"  NOTE: generation stopped at num_magnets={self.num_magnets}")

        self.compute_area_metrics(accepted_count=stats['accepted'])

        if self.display_collection:
            self.collection.show(backend='matplotlib')

        return self.collection
