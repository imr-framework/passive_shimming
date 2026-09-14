"""
Passive-shimming export utilities.

Classes
-------
ShimCollectionExporter
    Convert the optimized {-1, 0, +1} state vector into fresh Magpylib
    collections for the top and bottom shim planes, save them, and display
    them.

ShimTrayExporter
    Produce fabrication-ready top and bottom STL shim trays with:
        - magnet pockets cut into the tray INNER face,
        - the original external STL plate used as the tray body when one was
          ingested by TrayGeometry,
        - 1.5 mm alignment notch at the -Y edge,
        - X+/X-/Y+/Y- orientation engravings,
        - "+" polarity engravings for magnets with global +Z polarization,
        - no negative-polarity engraving by default,
        - verification/reload utilities and visualization colors.

Design conventions
------------------
1. Internal geometry is in meters.
2. STL files are exported in the same internal meter coordinates used by the
   current workflow. If your slicer assumes millimeters, configure the slicer
   accordingly or use ``stl_scale=1000`` when constructing ShimTrayExporter.
3. For an external STL workflow, the standardized ``external_tray.bottom_tray``
   and ``external_tray.top_tray`` meshes are copied directly. No generic disc
   is substituted.
4. Each standalone fabrication tray is translated only along Z so its body is
   centered near Z=0. XY geometry from the standardized external STL is
   preserved.
5. Pockets and engravings are cut from the INNER surface:
       bottom tray -> +Z face
       top tray    -> -Z face
6. State/polarity convention:
       +1 -> global +Z polarization -> engrave "+"
       -1 -> global -Z polarization -> no mark unless explicitly requested
7. Visualization colors do not alter STL geometry.
"""

from __future__ import annotations

import copy
import json
import pickle
import warnings
from pathlib import Path

import magpylib as magpy
import numpy as np
import trimesh

try:
    from shapely.geometry import Point, Polygon, LineString, box as shapely_box
    from shapely.ops import nearest_points
except Exception:  # pragma: no cover - graceful fallback if Shapely unavailable
    Point = None
    Polygon = None
    LineString = None
    shapely_box = None
    nearest_points = None


# ---------------------------------------------------------------------------
# Visualization-only colors
# ---------------------------------------------------------------------------

TOP_TRAY_COLOR = np.array(
    [110, 180, 230, 255],
    dtype=np.uint8,
)

BOTTOM_TRAY_COLOR = np.array(
    [240, 180, 90, 255],
    dtype=np.uint8,
)

TOP_MAGNET_COLOR = "#6EB4E6"
BOTTOM_MAGNET_COLOR = "#F0B45A"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _json_ready(value):
    if value is None:
        return None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, dict):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_ready(item)
            for item in value
        ]

    return value


def _copy_orientation(orientation):
    if orientation is None:
        return None
    return copy.deepcopy(orientation)


def _source_polarization(source):
    """
    Return a local polarization-like vector in tesla.

    The new workflow uses ``polarization``. ``magnetization`` is supported only
    as a compatibility fallback for older serialized Magpylib objects.
    """
    if hasattr(source, "polarization"):
        value = getattr(
            source,
            "polarization",
        )

        if value is not None:
            return np.asarray(
                value,
                dtype=float,
            ).copy()

    if hasattr(source, "magnetization"):
        value = getattr(
            source,
            "magnetization",
        )

        if value is not None:
            return np.asarray(
                value,
                dtype=float,
            ).copy()

    raise AttributeError(
        "Magnet source provides neither polarization nor magnetization."
    )


def _global_vector(
    vector,
    orientation,
):
    vector = np.asarray(
        vector,
        dtype=float,
    )

    if orientation is None:
        return vector.copy()

    return np.asarray(
        orientation.apply(
            vector
        ),
        dtype=float,
    )


def _fresh_cuboid(
    candidate,
    state,
):
    """
    Construct an independent Magpylib Cuboid.

    Reconstructing rather than deepcopying avoids retaining parent Collection
    relationships.
    """
    state = int(
        state
    )

    if state not in (-1, 1):
        raise ValueError(
            "_fresh_cuboid requires state -1 or +1."
        )

    polarization = _source_polarization(
        candidate
    )

    # Candidate objects are generated using +polarization. The optimized
    # state controls physical polarity.
    polarization = (
        polarization
        * state
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
        polarization=polarization,
        orientation=_copy_orientation(
            candidate.orientation
        ),
    )


def _flatten_collection(collection):
    """
    Return leaf magnetic sources from a Magpylib Collection-like object.
    """
    leaves = []

    def visit(item):
        if isinstance(
            item,
            magpy.Collection,
        ):
            for child in item:
                visit(
                    child
                )
        else:
            leaves.append(
                item
            )

    visit(
        collection
    )

    return leaves


def _mesh_from_any(mesh_like):
    """
    Return one Trimesh from Trimesh/Scene/path-like input.
    """
    if isinstance(
        mesh_like,
        trimesh.Trimesh,
    ):
        return mesh_like.copy()

    if isinstance(
        mesh_like,
        trimesh.Scene,
    ):
        geometries = [
            geometry.copy()
            for geometry
            in mesh_like.geometry.values()
        ]

        if not geometries:
            raise ValueError(
                "Trimesh Scene contains no geometry."
            )

        return trimesh.util.concatenate(
            geometries
        )

    loaded = trimesh.load(
        mesh_like,
        force=None,
    )

    return _mesh_from_any(
        loaded
    )


# ===========================================================================
# Optimized Magpylib collection export
# ===========================================================================

class ShimCollectionExporter:
    """
    Build independent optimized Magpylib collections from candidate states.

    Parameters
    ----------
    candidate_tray : CandidateTray
        Must provide ``candidates`` and ``candidate_ids``.

    states : array-like
        Ordered discrete states aligned with ``candidate_tray.candidates``.

    Notes
    -----
    The returned dictionary has keys:
        ``combined``
        ``top``
        ``bottom``

    Fresh Cuboids are reconstructed to avoid Magpylib parent conflicts.
    """

    def __init__(
        self,
        candidate_tray,
        states,
        debug=False,
    ):
        self.candidate_tray = (
            candidate_tray
        )

        self.states = np.asarray(
            states,
            dtype=int,
        ).reshape(
            -1
        )

        self.debug = bool(
            debug
        )

        self.optimized_collections = None
        self.saved_paths = {}

        self.validate()

    @property
    def candidates(self):
        return list(
            self.candidate_tray.candidates
        )

    @property
    def candidate_ids(self):
        ids = getattr(
            self.candidate_tray,
            "candidate_ids",
            None,
        )

        if ids is None:
            return [
                f"candidate_{index:04d}"
                for index
                in range(
                    len(
                        self.candidates
                    )
                )
            ]

        return list(
            ids
        )

    def validate(self):
        if len(
            self.states
        ) != len(
            self.candidates
        ):
            raise ValueError(
                "State-vector length does not match candidate count."
            )

        unique_states = set(
            np.unique(
                self.states
            ).tolist()
        )

        if not unique_states.issubset(
            {-1, 0, 1}
        ):
            raise ValueError(
                "Shim states must contain only -1, 0, and +1."
            )

        return True

    def build(self):
        """
        Build top, bottom, and combined optimized collections.
        """
        top = magpy.Collection(
            style_label="top_shims"
        )

        bottom = magpy.Collection(
            style_label="bottom_shims"
        )

        combined = magpy.Collection(
            style_label="optimized_shims"
        )

        manifest = []

        for (
            candidate_id,
            candidate,
            state,
        ) in zip(
            self.candidate_ids,
            self.candidates,
            self.states,
        ):
            state = int(
                state
            )

            if state == 0:
                continue

            position = np.asarray(
                candidate.position,
                dtype=float,
            )

            independent = _fresh_cuboid(
                candidate,
                state,
            )

            # Collections cannot share one child object, so build one fresh
            # source for the plane collection and another for combined.
            combined_source = _fresh_cuboid(
                candidate,
                state,
            )

            if position[2] >= 0:
                independent.style.color = (
                    TOP_MAGNET_COLOR
                )

                top.add(
                    independent
                )

                plane = "top"

            else:
                independent.style.color = (
                    BOTTOM_MAGNET_COLOR
                )

                bottom.add(
                    independent
                )

                plane = "bottom"

            combined.add(
                combined_source
            )

            manifest.append(
                {
                    "candidate_id":
                        candidate_id,
                    "plane":
                        plane,
                    "state":
                        state,
                    "position_m":
                        position,
                    "dimension_m":
                        np.asarray(
                            candidate.dimension,
                            dtype=float,
                        ),
                    "polarization_local_T":
                        _source_polarization(
                            independent
                        ),
                    "polarization_global_T":
                        _global_vector(
                            _source_polarization(
                                independent
                            ),
                            independent.orientation,
                        ),
                }
            )

        self.optimized_collections = {
            "combined":
                combined,
            "top":
                top,
            "bottom":
                bottom,
            "manifest":
                manifest,
        }

        print(
            "\nOptimized shim collections"
        )
        print(
            f"  Active magnets: "
            f"{len(manifest)}"
        )
        print(
            f"  Top: "
            f"{len(_flatten_collection(top))}"
        )
        print(
            f"  Bottom: "
            f"{len(_flatten_collection(bottom))}"
        )
        print(
            f"  Positive: "
            f"{int(np.sum(self.states == 1))}"
        )
        print(
            f"  Negative: "
            f"{int(np.sum(self.states == -1))}"
        )

        return self.optimized_collections

    def _require_build(self):
        if self.optimized_collections is None:
            raise RuntimeError(
                "Call ShimCollectionExporter.build() first."
            )

    def save(
        self,
        output_directory,
        combined_filename="magnet_collection_shims.pkl",
        top_filename="magnet_collection_shims_top.pkl",
        bottom_filename="magnet_collection_shims_bottom.pkl",
        manifest_filename="shim_collection_manifest.json",
    ):
        """
        Save optimized Magpylib collections and a JSON manifest.
        """
        self._require_build()

        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        paths = {
            "combined":
                output_directory
                / combined_filename,
            "top":
                output_directory
                / top_filename,
            "bottom":
                output_directory
                / bottom_filename,
            "manifest":
                output_directory
                / manifest_filename,
        }

        for key in (
            "combined",
            "top",
            "bottom",
        ):
            with paths[
                key
            ].open(
                "wb"
            ) as file:
                pickle.dump(
                    self.optimized_collections[
                        key
                    ],
                    file,
                )

        with paths[
            "manifest"
        ].open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                _json_ready(
                    {
                        "n_candidates":
                            len(
                                self.states
                            ),
                        "n_active":
                            int(
                                np.count_nonzero(
                                    self.states
                                )
                            ),
                        "n_positive":
                            int(
                                np.sum(
                                    self.states
                                    == 1
                                )
                            ),
                        "n_negative":
                            int(
                                np.sum(
                                    self.states
                                    == -1
                                )
                            ),
                        "n_absent":
                            int(
                                np.sum(
                                    self.states
                                    == 0
                                )
                            ),
                        "magnets":
                            self.optimized_collections[
                                "manifest"
                            ],
                    }
                ),
                file,
                indent=2,
            )

        self.saved_paths = (
            paths
        )

        return paths

    def show(self):
        """
        Display top and bottom optimized magnetic collections.
        """
        self._require_build()

        top = self.optimized_collections[
            "top"
        ]

        bottom = self.optimized_collections[
            "bottom"
        ]

        if len(
            _flatten_collection(
                top
            )
        ):
            top.show()

        if len(
            _flatten_collection(
                bottom
            )
        ):
            bottom.show()

    def statistics(self):
        self._require_build()

        return {
            "n_candidates":
                len(
                    self.states
                ),
            "n_active":
                int(
                    np.count_nonzero(
                        self.states
                    )
                ),
            "n_positive":
                int(
                    np.sum(
                        self.states
                        == 1
                    )
                ),
            "n_negative":
                int(
                    np.sum(
                        self.states
                        == -1
                    )
                ),
            "n_absent":
                int(
                    np.sum(
                        self.states
                        == 0
                    )
                ),
            "n_top":
                len(
                    _flatten_collection(
                        self.optimized_collections[
                            "top"
                        ]
                    )
                ),
            "n_bottom":
                len(
                    _flatten_collection(
                        self.optimized_collections[
                            "bottom"
                        ]
                    )
                ),
        }


# ===========================================================================
# Fabrication tray export
# ===========================================================================

class ShimTrayExporter:
    """
    Fabrication STL exporter for top and bottom passive-shim trays.

    Parameters
    ----------
    tray_geometry : TrayGeometry
        Current modular tray geometry. External-STL mode is detected through
        ``tray_geometry.mode == 'external_stl'`` and uses the prepared
        ``external_tray.top_tray`` / ``bottom_tray`` meshes directly.

    optimized_collections : dict
        Output of :class:`ShimCollectionExporter.build`.

    pocket_clearance_m : float
        Added pocket depth clearance. Default 0.10 mm.

    pocket_xy_clearance_m : float
        Optional XY clearance around magnet slots. Default 0.10 mm total on
        each dimension.

    notch_diameter_m : float
        Alignment-notch diameter. Default 1.5 mm.

    polarity_engraving_depth_m : float
        Default polarity-mark depth. 0.85 mm.

    orientation_engraving_depth_m : float
        Orientation-mark depth. Default 0.85 mm.

    generated_tray_thickness_m : float or None
        Thickness used only when no external STL exists. If None, inferred as
        max active magnet thickness + pocket_clearance_m.

    stl_scale : float
        Coordinate multiplier applied only at final STL export. Keep 1.0 for
        meter-coordinate STL, use 1000.0 for millimeter-coordinate STL.
    """

    def __init__(
        self,
        tray_geometry,
        optimized_collections,
        pocket_clearance_m=0.10e-3,
        pocket_xy_clearance_m=0.10e-3,
        notch_diameter_m=1.50e-3,
        polarity_engraving_depth_m=0.85e-3,
        orientation_engraving_depth_m=0.85e-3,
        generated_tray_thickness_m=None,
        generated_tray_margin_m=0.0,
        stl_scale=1.0,
        boolean_engine=None,
        debug=False,
    ):
        self.tray_geometry = (
            tray_geometry
        )

        self.optimized_collections = (
            optimized_collections
        )

        self.pocket_clearance_m = float(
            pocket_clearance_m
        )

        self.pocket_xy_clearance_m = float(
            pocket_xy_clearance_m
        )

        self.notch_diameter_m = float(
            notch_diameter_m
        )

        self.polarity_engraving_depth_m = float(
            polarity_engraving_depth_m
        )

        self.orientation_engraving_depth_m = float(
            orientation_engraving_depth_m
        )

        self.generated_tray_thickness_m = (
            None
            if generated_tray_thickness_m is None
            else float(
                generated_tray_thickness_m
            )
        )

        self.generated_tray_margin_m = float(
            generated_tray_margin_m
        )

        self.stl_scale = float(
            stl_scale
        )

        self.boolean_engine = (
            boolean_engine
        )

        self.debug = bool(
            debug
        )

        self.exported_paths = {}
        self.exported_meshes = {}
        self.export_report = {}

        self.validate()

    # ------------------------------------------------------------------
    # Input access
    # ------------------------------------------------------------------

    @property
    def mode(self):
        return getattr(
            self.tray_geometry,
            "mode",
            "circular",
        )

    @property
    def external_tray(self):
        return getattr(
            self.tray_geometry,
            "external_tray",
            None,
        )

    def _collection(
        self,
        side,
    ):
        if not isinstance(
            self.optimized_collections,
            dict,
        ):
            raise TypeError(
                "optimized_collections must be the dictionary returned by "
                "ShimCollectionExporter.build()."
            )

        if side not in self.optimized_collections:
            raise KeyError(
                f"optimized_collections does not contain '{side}'."
            )

        return self.optimized_collections[
            side
        ]

    def _magnets(
        self,
        side,
    ):
        return _flatten_collection(
            self._collection(
                side
            )
        )

    def validate(self):
        for side in (
            "top",
            "bottom",
        ):
            self._collection(
                side
            )

        if self.pocket_clearance_m < 0:
            raise ValueError(
                "pocket_clearance_m must be >= 0."
            )

        if self.pocket_xy_clearance_m < 0:
            raise ValueError(
                "pocket_xy_clearance_m must be >= 0."
            )

        if self.notch_diameter_m <= 0:
            raise ValueError(
                "notch_diameter_m must be positive."
            )

        if self.stl_scale <= 0:
            raise ValueError(
                "stl_scale must be positive."
            )

        if self.mode == "external_stl":
            if self.external_tray is None:
                raise ValueError(
                    "TrayGeometry is external_stl mode but external_tray "
                    "is unavailable."
                )

            for attr in (
                "top_tray",
                "bottom_tray",
            ):
                if not hasattr(
                    self.external_tray,
                    attr,
                ):
                    raise AttributeError(
                        f"external_tray does not provide {attr}."
                    )

        return True

    # ------------------------------------------------------------------
    # Base tray geometry
    # ------------------------------------------------------------------

    def _active_magnet_thickness(
        self,
    ):
        values = []

        for side in (
            "top",
            "bottom",
        ):
            for magnet in self._magnets(
                side
            ):
                dimensions = np.asarray(
                    magnet.dimension,
                    dtype=float,
                )

                # Candidates currently use Z as tray normal.
                values.append(
                    float(
                        dimensions[2]
                    )
                )

        if values:
            return max(
                values
            )

        # No active magnets: fall back to candidate-tray magnet if reachable.
        candidate_tray = getattr(
            self.tray_geometry,
            "candidate_tray",
            None,
        )

        if candidate_tray is not None:
            pass

        return 3.18e-3

    def _inferred_generated_thickness(
        self,
    ):
        if self.generated_tray_thickness_m is not None:
            return self.generated_tray_thickness_m

        return (
            self._active_magnet_thickness()
            + self.pocket_clearance_m
        )

    def _circular_diameter(
        self,
    ):
        for attr in (
            "diameter_m",
            "candidate_diameter_m",
        ):
            value = getattr(
                self.tray_geometry,
                attr,
                None,
            )

            if value is not None:
                return float(
                    value
                )

        raise AttributeError(
            "Circular TrayGeometry does not provide a diameter."
        )

    def _external_base(
        self,
        side,
    ):
        attr = (
            "top_tray"
            if side == "top"
            else "bottom_tray"
        )

        mesh = _mesh_from_any(
            getattr(
                self.external_tray,
                attr,
            )
        )

        # External tray preparation already standardized the plate to XY/Z and
        # positioned the physical pair. For fabrication, retain XY exactly and
        # simply remove the global pole-piece Z offset.
        z_center = float(
            np.mean(
                mesh.bounds[
                    :,
                    2
                ]
            )
        )

        mesh.apply_translation(
            [
                0.0,
                0.0,
                -z_center,
            ]
        )

        return mesh

    def _generated_base(
        self,
        side,
    ):
        thickness = (
            self._inferred_generated_thickness()
        )

        diameter = (
            self._circular_diameter()
        )

        mesh = trimesh.creation.cylinder(
            radius=(
                diameter
                / 2
                + self.generated_tray_margin_m
            ),
            height=
                thickness,
            sections=192,
        )

        return mesh

    def _base_mesh(
        self,
        side,
    ):
        if self.mode == "external_stl":
            mesh = self._external_base(
                side
            )

            source = "external_stl"

        else:
            mesh = self._generated_base(
                side
            )

            source = "generated_circular"

        if self.debug:
            print(
                f"\n{side.title()} base tray"
            )
            print(
                f"  source: {source}"
            )
            print(
                f"  extents: "
                f"{mesh.extents*1e3} mm"
            )
            print(
                f"  bounds: "
                f"{mesh.bounds*1e3} mm"
            )

        return mesh

    # ------------------------------------------------------------------
    # Inner-face convention
    # ------------------------------------------------------------------

    @staticmethod
    def _inner_face_z(
        mesh,
        side,
    ):
        if side == "bottom":
            return float(
                mesh.bounds[
                    1,
                    2
                ]
            )

        if side == "top":
            return float(
                mesh.bounds[
                    0,
                    2
                ]
            )

        raise ValueError(
            "side must be 'top' or 'bottom'."
        )

    @staticmethod
    def _inward_sign(
        side,
    ):
        # Direction from the inner surface INTO tray material.
        if side == "bottom":
            return -1.0

        if side == "top":
            return +1.0

        raise ValueError(
            "side must be 'top' or 'bottom'."
        )

    @staticmethod
    def _tray_thickness(
        mesh,
    ):
        return float(
            mesh.bounds[
                1,
                2
            ]
            - mesh.bounds[
                0,
                2
            ]
        )

    # ------------------------------------------------------------------
    # XY footprint helpers
    # ------------------------------------------------------------------

    def _allowed_region(
        self,
        base_mesh,
    ):
        if (
            self.mode == "external_stl"
            and self.external_tray is not None
        ):
            region = getattr(
                self.external_tray,
                "allowed_xy_region",
                None,
            )

            if region is not None:
                return region

        if Point is not None:
            if self.mode != "external_stl":
                diameter = (
                    self._circular_diameter()
                )

                return Point(
                    0.0,
                    0.0,
                ).buffer(
                    diameter
                    / 2,
                    resolution=128,
                )

            bounds = base_mesh.bounds

            return shapely_box(
                bounds[
                    0,
                    0
                ],
                bounds[
                    0,
                    1
                ],
                bounds[
                    1,
                    0
                ],
                bounds[
                    1,
                    1
                ],
            )

        return None

    def _xy_inside(
        self,
        region,
        points,
        clearance=0.0,
    ):
        points = np.asarray(
            points,
            dtype=float,
        )

        if region is None or Point is None:
            return True

        safe_region = region

        if clearance > 0:
            buffered = region.buffer(
                -clearance
            )

            if not buffered.is_empty:
                safe_region = buffered

        return all(
            safe_region.covers(
                Point(
                    float(
                        point[0]
                    ),
                    float(
                        point[1]
                    ),
                )
            )
            for point
            in points
        )

    # ------------------------------------------------------------------
    # Pocket geometry
    # ------------------------------------------------------------------

    def _pocket_depth(
        self,
        magnet,
        base_mesh,
    ):
        magnet_thickness = float(
            np.asarray(
                magnet.dimension,
                dtype=float,
            )[2]
        )

        desired = (
            magnet_thickness
            + self.pocket_clearance_m
        )

        tray_thickness = (
            self._tray_thickness(
                base_mesh
            )
        )

        # The small extra overlap ensures a full cut if depth equals thickness.
        return min(
            desired,
            tray_thickness
            + 0.05e-3,
        )

    def _pocket_cutter(
        self,
        magnet,
        base_mesh,
        side,
    ):
        position = np.asarray(
            magnet.position,
            dtype=float,
        )

        dimension = np.asarray(
            magnet.dimension,
            dtype=float,
        ).copy()

        dimension[
            0
        ] += self.pocket_xy_clearance_m

        dimension[
            1
        ] += self.pocket_xy_clearance_m

        depth = self._pocket_depth(
            magnet,
            base_mesh,
        )

        dimension[
            2
        ] = (
            depth
            + 0.10e-3
        )

        cutter = trimesh.creation.box(
            extents=
                dimension,
        )

        if magnet.orientation is not None:
            transform = np.eye(
                4
            )

            transform[
                :3,
                :3
            ] = magnet.orientation.as_matrix()

            cutter.apply_transform(
                transform
            )

        inner_z = self._inner_face_z(
            base_mesh,
            side,
        )

        inward_sign = self._inward_sign(
            side
        )

        center_z = (
            inner_z
            + inward_sign
            * (
                depth
                / 2
                - 0.05e-3
            )
        )

        cutter.apply_translation(
            [
                position[
                    0
                ],
                position[
                    1
                ],
                center_z,
            ]
        )

        return cutter

    def _pocket_xy_bounds(
        self,
        magnet,
    ):
        # Build a local cutter at z=0 only to obtain oriented XY bounds.
        position = np.asarray(
            magnet.position,
            dtype=float,
        )

        dimension = np.asarray(
            magnet.dimension,
            dtype=float,
        ).copy()

        dimension[
            :2
        ] += self.pocket_xy_clearance_m

        dimension[
            2
        ] = 1e-3

        mesh = trimesh.creation.box(
            extents=
                dimension,
        )

        if magnet.orientation is not None:
            transform = np.eye(
                4
            )

            transform[
                :3,
                :3
            ] = magnet.orientation.as_matrix()

            mesh.apply_transform(
                transform
            )

        mesh.apply_translation(
            [
                position[
                    0
                ],
                position[
                    1
                ],
                0.0,
            ]
        )

        return (
            mesh.bounds[
                0,
                :2
            ].copy(),
            mesh.bounds[
                1,
                :2
            ].copy(),
        )

    # ------------------------------------------------------------------
    # Engraving primitives
    # ------------------------------------------------------------------

    def _engraving_center_z(
        self,
        base_mesh,
        side,
        depth,
    ):
        inner_z = self._inner_face_z(
            base_mesh,
            side,
        )

        inward_sign = self._inward_sign(
            side
        )

        # Cutter protrudes 0.05 mm beyond surface to avoid coplanar Boolean
        # ambiguity.
        return (
            inner_z
            + inward_sign
            * (
                depth
                / 2
                - 0.05e-3
            )
        )

    def _bar(
        self,
        base_mesh,
        side,
        center_xy,
        angle,
        length,
        width,
        depth,
    ):
        bar = trimesh.creation.box(
            extents=[
                float(
                    length
                ),
                float(
                    width
                ),
                float(
                    depth
                    + 0.10e-3
                ),
            ]
        )

        rotation = (
            trimesh.transformations
            .rotation_matrix(
                float(
                    angle
                ),
                [
                    0.0,
                    0.0,
                    1.0,
                ],
            )
        )

        bar.apply_transform(
            rotation
        )

        bar.apply_translation(
            [
                float(
                    center_xy[
                        0
                    ]
                ),
                float(
                    center_xy[
                        1
                    ]
                ),
                self._engraving_center_z(
                    base_mesh,
                    side,
                    depth,
                ),
            ]
        )

        return bar

    def _symbol_meshes(
        self,
        symbol,
        base_mesh,
        side,
        label_center_xy,
        char_center_xy,
        char_size,
        stroke_width,
        depth,
        label_rotation=0.0,
    ):
        meshes = []

        c = np.cos(
            label_rotation
        )

        s = np.sin(
            label_rotation
        )

        rot2 = np.array(
            [
                [
                    c,
                    -s,
                ],
                [
                    s,
                    c,
                ],
            ]
        )

        def add_bar(
            local_center,
            local_angle,
            length,
        ):
            global_center = (
                np.asarray(
                    label_center_xy,
                    dtype=float,
                )
                + rot2
                @ np.asarray(
                    local_center,
                    dtype=float,
                )
            )

            meshes.append(
                self._bar(
                    base_mesh=
                        base_mesh,
                    side=
                        side,
                    center_xy=
                        global_center,
                    angle=(
                        label_rotation
                        + local_angle
                    ),
                    length=
                        length,
                    width=
                        stroke_width,
                    depth=
                        depth,
                )
            )

        center = np.asarray(
            char_center_xy,
            dtype=float,
        )

        if symbol == "+":
            add_bar(
                center,
                0.0,
                0.85
                * char_size,
            )

            add_bar(
                center,
                np.pi
                / 2,
                0.85
                * char_size,
            )

        elif symbol == "-":
            add_bar(
                center,
                0.0,
                0.85
                * char_size,
            )

        elif symbol == "X":
            add_bar(
                center,
                np.pi
                / 4,
                char_size,
            )

            add_bar(
                center,
                -np.pi
                / 4,
                char_size,
            )

        elif symbol == "Y":
            add_bar(
                center
                + np.array(
                    [
                        -0.18
                        * char_size,
                        0.16
                        * char_size,
                    ]
                ),
                -np.pi
                / 4,
                0.55
                * char_size,
            )

            add_bar(
                center
                + np.array(
                    [
                        +0.18
                        * char_size,
                        0.16
                        * char_size,
                    ]
                ),
                +np.pi
                / 4,
                0.55
                * char_size,
            )

            add_bar(
                center
                + np.array(
                    [
                        0.0,
                        -0.16
                        * char_size,
                    ]
                ),
                np.pi
                / 2,
                0.65
                * char_size,
            )

        else:
            raise ValueError(
                f"Unsupported engraving symbol '{symbol}'."
            )

        return meshes

    def _label_meshes(
        self,
        label_text,
        base_mesh,
        side,
        center_xy,
        char_size=4.5e-3,
        stroke_width=0.8e-3,
        depth=None,
        rotation=0.0,
    ):
        if depth is None:
            depth = (
                self.orientation_engraving_depth_m
            )

        n = len(
            label_text
        )

        pitch = (
            1.20
            * char_size
        )

        x_offsets = (
            np.arange(
                n
            )
            - (
                n
                - 1
            )
            / 2
        ) * pitch

        meshes = []

        for (
            char,
            offset,
        ) in zip(
            label_text,
            x_offsets,
        ):
            meshes.extend(
                self._symbol_meshes(
                    symbol=
                        char,
                    base_mesh=
                        base_mesh,
                    side=
                        side,
                    label_center_xy=
                        np.asarray(
                            center_xy,
                            dtype=float,
                        ),
                    char_center_xy=
                        np.array(
                            [
                                offset,
                                0.0,
                            ]
                        ),
                    char_size=
                        char_size,
                    stroke_width=
                        stroke_width,
                    depth=
                        depth,
                    label_rotation=
                        rotation,
                )
            )

        return meshes

    @staticmethod
    def _label_bbox(
        center_xy,
        label_text,
        char_size,
    ):
        pitch = (
            1.20
            * char_size
        )

        width = (
            (
                len(
                    label_text
                )
                - 1
            )
            * pitch
            + char_size
        )

        height = char_size

        center_xy = np.asarray(
            center_xy,
            dtype=float,
        )

        return (
            center_xy
            - np.array(
                [
                    width
                    / 2,
                    height
                    / 2,
                ]
            ),
            center_xy
            + np.array(
                [
                    width
                    / 2,
                    height
                    / 2,
                ]
            ),
        )

    @staticmethod
    def _bbox_overlap(
        first_min,
        first_max,
        second_min,
        second_max,
        clearance=0.0,
    ):
        return not (
            first_max[
                0
            ]
            + clearance
            < second_min[
                0
            ]
            or first_min[
                0
            ]
            - clearance
            > second_max[
                0
            ]
            or first_max[
                1
            ]
            + clearance
            < second_min[
                1
            ]
            or first_min[
                1
            ]
            - clearance
            > second_max[
                1
            ]
        )

    def _label_collision(
        self,
        center_xy,
        label_text,
        char_size,
        magnets,
        clearance=1.5e-3,
    ):
        label_min, label_max = (
            self._label_bbox(
                center_xy,
                label_text,
                char_size,
            )
        )

        for magnet in magnets:
            pocket_min, pocket_max = (
                self._pocket_xy_bounds(
                    magnet
                )
            )

            if self._bbox_overlap(
                label_min,
                label_max,
                pocket_min,
                pocket_max,
                clearance=
                    clearance,
            ):
                return True

        return False

    def _label_inside_region(
        self,
        center_xy,
        label_text,
        char_size,
        region,
        boundary_clearance=1.0e-3,
    ):
        bbox_min, bbox_max = (
            self._label_bbox(
                center_xy,
                label_text,
                char_size,
            )
        )

        corners = np.array(
            [
                [
                    bbox_min[
                        0
                    ],
                    bbox_min[
                        1
                    ],
                ],
                [
                    bbox_min[
                        0
                    ],
                    bbox_max[
                        1
                    ],
                ],
                [
                    bbox_max[
                        0
                    ],
                    bbox_min[
                        1
                    ],
                ],
                [
                    bbox_max[
                        0
                    ],
                    bbox_max[
                        1
                    ],
                ],
            ]
        )

        return self._xy_inside(
            region,
            corners,
            clearance=
                boundary_clearance,
        )

    def _direction_boundary_distance(
        self,
        direction,
        region,
        base_mesh,
    ):
        direction = np.asarray(
            direction,
            dtype=float,
        )

        if region is not None:
            bounds = region.bounds

            if abs(
                direction[
                    0
                ]
            ) > 0.5:
                return (
                    bounds[
                        2
                    ]
                    if direction[
                        0
                    ] > 0
                    else abs(
                        bounds[
                            0
                        ]
                    )
                )

            return (
                bounds[
                    3
                ]
                if direction[
                    1
                ] > 0
                else abs(
                    bounds[
                        1
                    ]
                )
            )

        bounds = base_mesh.bounds

        if abs(
            direction[
                0
            ]
        ) > 0.5:
            return (
                bounds[
                    1,
                    0
                ]
                if direction[
                    0
                ] > 0
                else abs(
                    bounds[
                        0,
                        0
                    ]
                )
            )

        return (
            bounds[
                1,
                1
            ]
            if direction[
                1
            ] > 0
            else abs(
                bounds[
                    0,
                    1
                ]
            )
        )

    def _find_orientation_label_position(
        self,
        label_text,
        direction,
        magnets,
        region,
        base_mesh,
        char_size,
    ):
        direction = np.asarray(
            direction,
            dtype=float,
        )

        direction /= np.linalg.norm(
            direction
        )

        tangent = np.array(
            [
                -direction[
                    1
                ],
                direction[
                    0
                ],
            ]
        )

        boundary_distance = (
            self._direction_boundary_distance(
                direction,
                region,
                base_mesh,
            )
        )

        inward_values = np.arange(
            3.0e-3,
            50.0e-3
            + 0.5e-3,
            1.0e-3,
        )

        tangent_values = np.arange(
            0.0,
            30.0e-3
            + 1.0e-3,
            2.0e-3,
        )

        for inset in inward_values:
            axis_center = (
                direction
                * (
                    boundary_distance
                    - inset
                )
            )

            offsets = [
                0.0
            ]

            for magnitude in tangent_values[
                1:
            ]:
                offsets.extend(
                    [
                        magnitude,
                        -magnitude,
                    ]
                )

            for offset in offsets:
                candidate = (
                    axis_center
                    + tangent
                    * offset
                )

                if not self._label_inside_region(
                    candidate,
                    label_text,
                    char_size,
                    region,
                ):
                    continue

                if self._label_collision(
                    candidate,
                    label_text,
                    char_size,
                    magnets,
                ):
                    continue

                return candidate

        raise RuntimeError(
            "Could not place orientation label "
            f"{label_text} without intersecting a pocket or tray boundary."
        )

    def _orientation_mark_cutters(
        self,
        base_mesh,
        side,
        magnets,
    ):
        char_size = 4.5e-3
        stroke_width = 0.8e-3

        region = self._allowed_region(
            base_mesh
        )

        labels = {
            "X+":
                np.array(
                    [
                        +1.0,
                        0.0,
                    ]
                ),
            "X-":
                np.array(
                    [
                        -1.0,
                        0.0,
                    ]
                ),
            "Y+":
                np.array(
                    [
                        0.0,
                        +1.0,
                    ]
                ),
            "Y-":
                np.array(
                    [
                        0.0,
                        -1.0,
                    ]
                ),
        }

        cutters = []
        positions = {}

        for (
            label_text,
            direction,
        ) in labels.items():
            center = (
                self._find_orientation_label_position(
                    label_text=
                        label_text,
                    direction=
                        direction,
                    magnets=
                        magnets,
                    region=
                        region,
                    base_mesh=
                        base_mesh,
                    char_size=
                        char_size,
                )
            )

            positions[
                label_text
            ] = center

            cutters.extend(
                self._label_meshes(
                    label_text=
                        label_text,
                    base_mesh=
                        base_mesh,
                    side=
                        side,
                    center_xy=
                        center,
                    char_size=
                        char_size,
                    stroke_width=
                        stroke_width,
                    depth=
                        self.orientation_engraving_depth_m,
                )
            )

        return (
            cutters,
            positions,
        )

    # ------------------------------------------------------------------
    # Polarity marks
    # ------------------------------------------------------------------

    @staticmethod
    def _magnet_is_positive_global_z(
        magnet,
    ):
        local = _source_polarization(
            magnet
        )

        global_vector = _global_vector(
            local,
            magnet.orientation,
        )

        return bool(
            global_vector[
                2
            ] > 0
        )

    def _polarity_center_candidates(
        self,
        magnet,
    ):
        position = np.asarray(
            magnet.position,
            dtype=float,
        )

        radial = position[
            :2
        ].copy()

        norm = np.linalg.norm(
            radial
        )

        if norm > 1e-12:
            radial /= norm
        else:
            radial = np.array(
                [
                    1.0,
                    0.0,
                ]
            )

        tangent = np.array(
            [
                -radial[
                    1
                ],
                radial[
                    0
                ],
            ]
        )

        dims = np.asarray(
            magnet.dimension,
            dtype=float,
        )

        half_width = (
            max(
                dims[
                    :2
                ]
            )
            / 2
        )

        # Prefer just outside the pocket, then search tangentially/radially.
        base_distance = (
            half_width
            + 2.0e-3
        )

        candidates = []

        for radial_extra in (
            0.0,
            1.0e-3,
            -1.0e-3,
            2.0e-3,
            -2.0e-3,
        ):
            for tangent_shift in (
                0.0,
                1.5e-3,
                -1.5e-3,
                3.0e-3,
                -3.0e-3,
            ):
                candidates.append(
                    position[
                        :2
                    ]
                    + radial
                    * (
                        base_distance
                        + radial_extra
                    )
                    + tangent
                    * tangent_shift
                )

        return candidates

    def _polarity_mark_cutters(
        self,
        base_mesh,
        side,
        magnets,
        positive_polarity_mark="+",
        negative_polarity_mark=None,
    ):
        region = self._allowed_region(
            base_mesh
        )

        cutters = []
        report = []

        char_size = 3.0e-3
        stroke_width = 0.70e-3

        for magnet_index, magnet in enumerate(
            magnets
        ):
            positive = (
                self._magnet_is_positive_global_z(
                    magnet
                )
            )

            symbol = (
                positive_polarity_mark
                if positive
                else negative_polarity_mark
            )

            if symbol is None:
                continue

            if symbol not in (
                "+",
                "-",
            ):
                raise ValueError(
                    "Polarity marks currently support '+', '-', or None."
                )

            chosen = None

            for center in self._polarity_center_candidates(
                magnet
            ):
                if not self._label_inside_region(
                    center_xy=
                        center,
                    label_text=
                        symbol,
                    char_size=
                        char_size,
                    region=
                        region,
                    boundary_clearance=
                        0.5e-3,
                ):
                    continue

                # Avoid ALL magnet pockets, including the pocket being marked.
                if self._label_collision(
                    center_xy=
                        center,
                    label_text=
                        symbol,
                    char_size=
                        char_size,
                    magnets=
                        magnets,
                    clearance=
                        0.25e-3,
                ):
                    continue

                chosen = center
                break

            if chosen is None:
                # Do not silently lose polarity information. Put the marker at
                # the preferred location even if crowded and report it.
                chosen = (
                    self._polarity_center_candidates(
                        magnet
                    )[
                        0
                    ]
                )

                warnings.warn(
                    f"{side} polarity mark for magnet {magnet_index} "
                    "could not be placed collision-free; using preferred "
                    "radial location.",
                    RuntimeWarning,
                )

                collision_free = False

            else:
                collision_free = True

            cutters.extend(
                self._label_meshes(
                    label_text=
                        symbol,
                    base_mesh=
                        base_mesh,
                    side=
                        side,
                    center_xy=
                        chosen,
                    char_size=
                        char_size,
                    stroke_width=
                        stroke_width,
                    depth=
                        self.polarity_engraving_depth_m,
                )
            )

            report.append(
                {
                    "magnet_index":
                        magnet_index,
                    "symbol":
                        symbol,
                    "positive_global_z":
                        positive,
                    "center_xy_m":
                        np.asarray(
                            chosen,
                            dtype=float,
                        ),
                    "collision_free":
                        collision_free,
                }
            )

        return (
            cutters,
            report,
        )

    # ------------------------------------------------------------------
    # Alignment notch
    # ------------------------------------------------------------------

    def _notch_center_xy(
        self,
        base_mesh,
    ):
        region = self._allowed_region(
            base_mesh
        )

        if (
            region is not None
            and LineString is not None
        ):
            bounds = region.bounds

            x_span = max(
                abs(
                    bounds[
                        0
                    ]
                ),
                abs(
                    bounds[
                        2
                    ]
                ),
                1e-3,
            )

            vertical = LineString(
                [
                    (
                        0.0,
                        bounds[
                            1
                        ]
                        - 2
                        * x_span,
                    ),
                    (
                        0.0,
                        bounds[
                            3
                        ]
                        + 2
                        * x_span,
                    ),
                ]
            )

            intersection = (
                region.boundary
                .intersection(
                    vertical
                )
            )

            points = []

            if hasattr(
                intersection,
                "geoms",
            ):
                for geometry in intersection.geoms:
                    if hasattr(
                        geometry,
                        "x",
                    ):
                        points.append(
                            [
                                geometry.x,
                                geometry.y,
                            ]
                        )

            elif hasattr(
                intersection,
                "x",
            ):
                points.append(
                    [
                        intersection.x,
                        intersection.y,
                    ]
                )

            if points:
                points = np.asarray(
                    points,
                    dtype=float,
                )

                return points[
                    np.argmin(
                        points[
                            :,
                            1
                        ]
                    )
                ]

        # Robust fallback: x=0 and the lowest tray Y bound.
        return np.array(
            [
                0.0,
                float(
                    base_mesh.bounds[
                        0,
                        1
                    ]
                ),
            ]
        )

    def _alignment_notch_cutter(
        self,
        base_mesh,
    ):
        center_xy = self._notch_center_xy(
            base_mesh
        )

        height = (
            self._tray_thickness(
                base_mesh
            )
            + 4.0e-3
        )

        cutter = trimesh.creation.cylinder(
            radius=
                self.notch_diameter_m
                / 2,
            height=
                height,
            sections=64,
        )

        cutter.apply_translation(
            [
                center_xy[
                    0
                ],
                center_xy[
                    1
                ],
                float(
                    np.mean(
                        base_mesh.bounds[
                            :,
                            2
                        ]
                    )
                ),
            ]
        )

        return (
            cutter,
            center_xy,
        )

    # ------------------------------------------------------------------
    # Boolean operations
    # ------------------------------------------------------------------

    def _clean_mesh(
        self,
        mesh,
    ):
        """
        Conservative cleanup of a Trimesh after Boolean operations.
        """

        mesh = _mesh_from_any(
            mesh
        )

        # Remove obvious topology artifacts.
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

        try:
            trimesh.repair.fix_winding(
                mesh
            )
        except Exception:
            pass

        return mesh


    def _difference(
        self,
        base,
        cutter,
        description,
    ):
        """
        Robust Boolean difference with cleanup before and after.
        """

        base = self._clean_mesh(
            base
        )

        cutter = self._clean_mesh(
            cutter
        )

        try:

            result = trimesh.boolean.difference(
                [
                    base,
                    cutter,
                ],
                engine=self.boolean_engine,
            )

        except Exception as exc:

            raise RuntimeError(
                "Trimesh Boolean difference failed while cutting "
                f"{description}. Ensure a supported Boolean backend "
                "(preferably manifold3d) is installed and available."
            ) from exc

        if result is None:

            raise RuntimeError(
                f"Boolean difference returned None while cutting "
                f"{description}."
            )

        result = _mesh_from_any(
            result
        )

        result = self._clean_mesh(
            result
        )

        return result


    def _apply_cutters(
        self,
        tray,
        cutters,
        description_prefix,
    ):
        """
        Apply many independent closed-volume cutters in ONE Boolean
        difference operation.

        Do NOT pre-union or concatenate the cutters. Keeping each cutter
        as an independent volume is considerably more robust with the
        manifold backend.
        """

        if not cutters:
            return tray

        tray = self._clean_mesh(
            tray
        )

        cleaned_cutters = [
            self._clean_mesh(
                cutter
            )
            for cutter in cutters
        ]

        # ----------------------------------------------------------
        # Diagnostic before Boolean
        # ----------------------------------------------------------

        bad_cutters = [
            index
            for index, cutter in enumerate(
                cleaned_cutters
            )
            if not cutter.is_volume
        ]

        print(
            f"  Applying {len(cleaned_cutters)} "
            f"{description_prefix} cutters as one Boolean"
        )

        print(
            f"    input tray: "
            f"watertight={tray.is_watertight}, "
            f"volume={tray.is_volume}"
        )

        print(
            f"    invalid cutter volumes: "
            f"{len(bad_cutters)}"
        )

        if bad_cutters:

            print(
                f"    invalid cutter indices: "
                f"{bad_cutters[:20]}"
            )

            raise RuntimeError(
                f"{description_prefix}: "
                f"{len(bad_cutters)} cutters are not valid volumes."
            )

        if not tray.is_volume:

            raise RuntimeError(
                f"{description_prefix}: input tray is not a valid "
                "volume BEFORE this Boolean operation."
            )

        # ----------------------------------------------------------
        # Important:
        #
        # Pass each cutter as an independent manifold object.
        # No union.
        # No concatenation.
        # ----------------------------------------------------------

        try:

            result = trimesh.boolean.difference(
                [
                    tray,
                    *cleaned_cutters,
                ],
                engine=self.boolean_engine,
            )

        except Exception as exc:

            raise RuntimeError(
                f"Boolean difference failed while cutting "
                f"{description_prefix} with "
                f"{len(cleaned_cutters)} independent cutters."
            ) from exc

        if result is None:

            raise RuntimeError(
                f"Boolean difference returned None while cutting "
                f"{description_prefix}."
            )

        result = _mesh_from_any(
            result
        )

        result = self._clean_mesh(
            result
        )

        # ----------------------------------------------------------
        # Diagnose immediately after this feature class
        # ----------------------------------------------------------

        print(
            f"    result: "
            f"watertight={result.is_watertight}, "
            f"volume={result.is_volume}, "
            f"faces={len(result.faces)}"
        )

        if not result.is_watertight:

            raise RuntimeError(
                f"{description_prefix} produced a non-watertight tray."
            )

        if not result.is_volume:

            raise RuntimeError(
                f"{description_prefix} produced a non-volume tray."
            )

        return result
    # ------------------------------------------------------------------
    # One-tray fabrication
    # ------------------------------------------------------------------

    def _fabricate_side(
        self,
        side,
        notch,
        orientation_marks,
        positive_polarity_mark,
        negative_polarity_mark,
    ):
        tray = self._base_mesh(
            side
        )

        magnets = self._magnets(
            side
        )

        source_extents = tray.extents.copy()

        report = {
            "side":
                side,
            "base_source":
                (
                    "external_stl"
                    if self.mode
                    == "external_stl"
                    else "generated_circular"
                ),
            "n_magnets":
                len(
                    magnets
                ),
            "source_extents_m":
                source_extents,
            "tray_thickness_m":
                self._tray_thickness(
                    tray
                ),
            "inner_face_z_m":
                self._inner_face_z(
                    tray,
                    side,
                ),
            "notch":
                None,
            "orientation_marks":
                {},
            "polarity_marks":
                [],
        }

        # 1. Magnet pockets
        pocket_cutters = [
            self._pocket_cutter(
                magnet,
                tray,
                side,
            )
            for magnet
            in magnets
        ]

        if pocket_cutters:
            tray = self._apply_cutters(
                tray,
                pocket_cutters,
                f"{side} magnet pocket",
            )

        # 2. Alignment notch
        if notch:
            notch_cutter, notch_center = (
                self._alignment_notch_cutter(
                    tray
                )
            )

            tray = self._difference(
                tray,
                notch_cutter,
                f"{side} alignment notch",
            )

            report[
                "notch"
            ] = {
                "diameter_m":
                    self.notch_diameter_m,
                "center_xy_m":
                    notch_center,
            }

        # 3. Scanner orientation engravings
        if orientation_marks:
            (
                orientation_cutters,
                orientation_positions,
            ) = self._orientation_mark_cutters(
                base_mesh=
                    tray,
                side=
                    side,
                magnets=
                    magnets,
            )

            tray = self._apply_cutters(
                tray,
                orientation_cutters,
                f"{side} orientation engraving",
            )

            report[
                "orientation_marks"
            ] = orientation_positions

        # 4. Polarity engravings
        (
            polarity_cutters,
            polarity_report,
        ) = self._polarity_mark_cutters(
            base_mesh=
                tray,
            side=
                side,
            magnets=
                magnets,
            positive_polarity_mark=
                positive_polarity_mark,
            negative_polarity_mark=
                negative_polarity_mark,
        )

        if polarity_cutters:
            tray = self._apply_cutters(
                tray,
                polarity_cutters,
                f"{side} polarity engraving",
            )

        report[
            "polarity_marks"
        ] = polarity_report

        report[
            "final_extents_m"
        ] = tray.extents.copy()

        report[
            "watertight_before_export"
        ] = bool(
            tray.is_watertight
        )

        print(
            f"\n{side.title()} fabrication input"
        )

        print(
            f"  Base watertight: "
            f"{tray.is_watertight}"
        )

        print(
            f"  Base volume: "
            f"{tray.is_volume}"
        )

        print(
            f"  Base Euler number: "
            f"{tray.euler_number}"
        )



        return (
            tray,
            report,
        )

    # ------------------------------------------------------------------
    # Public export API
    # ------------------------------------------------------------------

    def export(
        self,
        output_directory,
        notch=True,
        orientation_marks=True,
        positive_polarity_mark="+",
        negative_polarity_mark=None,
        top_filename="shim_tray_top.stl",
        bottom_filename="shim_tray_bottom.stl",
        report_filename="shim_tray_export_report.json",
    ):
        """
        Export fabrication-ready top and bottom STL trays.

        This matches the thin top-level workflow API.
        """
        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "\nFabricating shim trays"
        )
        print(
            f"  Geometry mode: "
            f"{self.mode}"
        )

        if self.mode == "external_stl":
            print(
                "  Base geometry: standardized external STL plates"
            )

        else:
            print(
                "  Base geometry: generated circular plates"
            )

        print(
            f"  Magnet pocket XY clearance: "
            f"{self.pocket_xy_clearance_m*1e3:.3f} mm"
        )
        print(
            f"  Magnet pocket depth clearance: "
            f"{self.pocket_clearance_m*1e3:.3f} mm"
        )
        print(
            f"  Notch diameter: "
            f"{self.notch_diameter_m*1e3:.3f} mm"
        )
        print(
            f"  Polarity engraving depth: "
            f"{self.polarity_engraving_depth_m*1e3:.3f} mm"
        )

        (
            top_mesh,
            top_report,
        ) = self._fabricate_side(
            side=
                "top",
            notch=
                bool(
                    notch
                ),
            orientation_marks=
                bool(
                    orientation_marks
                ),
            positive_polarity_mark=
                positive_polarity_mark,
            negative_polarity_mark=
                negative_polarity_mark,
        )

        (
            bottom_mesh,
            bottom_report,
        ) = self._fabricate_side(
            side=
                "bottom",
            notch=
                bool(
                    notch
                ),
            orientation_marks=
                bool(
                    orientation_marks
                ),
            positive_polarity_mark=
                positive_polarity_mark,
            negative_polarity_mark=
                negative_polarity_mark,
        )

        top_path = (
            output_directory
            / top_filename
        )

        bottom_path = (
            output_directory
            / bottom_filename
        )

        report_path = (
            output_directory
            / report_filename
        )

        export_top = top_mesh.copy()
        export_bottom = bottom_mesh.copy()

        if not np.isclose(
            self.stl_scale,
            1.0,
        ):
            export_top.apply_scale(
                self.stl_scale
            )

            export_bottom.apply_scale(
                self.stl_scale
            )

        export_top.export(
            top_path
        )

        export_bottom.export(
            bottom_path
        )

        self.exported_paths = {
            "top":
                top_path,
            "bottom":
                bottom_path,
            "report":
                report_path,
        }

        self.exported_meshes = {
            "top":
                top_mesh,
            "bottom":
                bottom_mesh,
        }

        self.export_report = {
            "stl_scale":
                self.stl_scale,
            "coordinate_units_before_export":
                "m",
            "top":
                top_report,
            "bottom":
                bottom_report,
            "filenames": {
                "top":
                    top_path,
                "bottom":
                    bottom_path,
            },
        }

        with report_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                _json_ready(
                    self.export_report
                ),
                file,
                indent=2,
            )

        print(
            f"  Top STL: "
            f"{top_path}"
        )
        print(
            f"  Bottom STL: "
            f"{bottom_path}"
        )

        return self.exported_paths

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _require_exports(self):
        if not self.exported_paths:
            raise RuntimeError(
                "Call ShimTrayExporter.export() first."
            )

    @staticmethod
    def _mesh_verification(
        mesh,
        filename,
    ):
        return {
            "filename":
                filename,
            "watertight":
                bool(
                    mesh.is_watertight
                ),
            "is_volume":
                bool(
                    mesh.is_volume
                ),
            "euler_number":
                int(
                    mesh.euler_number
                ),
            "n_vertices":
                int(
                    len(
                        mesh.vertices
                    )
                ),
            "n_faces":
                int(
                    len(
                        mesh.faces
                    )
                ),
            "bounds":
                np.asarray(
                    mesh.bounds,
                    dtype=float,
                ),
            "extents":
                np.asarray(
                    mesh.extents,
                    dtype=float,
                ),
            "volume":
                float(
                    mesh.volume
                ),
        }

    def verify_exports(self):
        """
        Reload the exact requested STL paths and return a verification report.

        This intentionally avoids the historical bug where fixed working-
        directory filenames were reloaded instead of the requested outputs.
        """
        self._require_exports()

        verification = {}

        for side in (
            "top",
            "bottom",
        ):
            path = self.exported_paths[
                side
            ]

            if not Path(
                path
            ).exists():
                raise FileNotFoundError(
                    f"Exported STL does not exist: {path}"
                )

            mesh = _mesh_from_any(
                path
            )

            verification[
                side
            ] = self._mesh_verification(
                mesh,
                path,
            )

        print(
            "\nSTL verification"
        )

        for side in (
            "top",
            "bottom",
        ):
            item = verification[
                side
            ]

            print(
                f"  {side.title()}:"
            )
            print(
                f"    watertight: "
                f"{item['watertight']}"
            )
            print(
                f"    volume mesh: "
                f"{item['is_volume']}"
            )
            print(
                f"    vertices: "
                f"{item['n_vertices']}"
            )
            print(
                f"    faces: "
                f"{item['n_faces']}"
            )
            print(
                f"    extents: "
                f"{np.asarray(item['extents'])} "
                f"(export coordinates)"
            )

        return verification

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def show_exports(self):
        """
        Reload exported STLs, apply visualization-only colors, and display.
        """
        self._require_exports()

        top = _mesh_from_any(
            self.exported_paths[
                "top"
            ]
        )

        bottom = _mesh_from_any(
            self.exported_paths[
                "bottom"
            ]
        )

        top.visual.face_colors = (
            TOP_TRAY_COLOR
        )

        bottom.visual.face_colors = (
            BOTTOM_TRAY_COLOR
        )

        trimesh.Scene(
            {
                "top_tray":
                    top
            }
        ).show()

        trimesh.Scene(
            {
                "bottom_tray":
                    bottom
            }
        ).show()

    def statistics(self):
        if not self.export_report:
            return {
                "mode":
                    self.mode,
                "n_top_magnets":
                    len(
                        self._magnets(
                            "top"
                        )
                    ),
                "n_bottom_magnets":
                    len(
                        self._magnets(
                            "bottom"
                        )
                    ),
            }

        return self.export_report
