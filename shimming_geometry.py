"""
Geometry models for passive shimming.

This module contains the reusable geometry layer used by the top-level
passive-shimming workflow.

Classes
-------
TrayGeometry
    Describes either:
      - an analytic circular two-plane shim tray, or
      - a custom external STL tray handled by ``external_stl_tray``.

ShimMagnet
    Defines the physical shim magnet using Magpylib polarization only.

CandidateTray
    Uses ``shim_ring`` to generate top and bottom candidate magnets,
    maintains stable candidate IDs, and reports packing / geometry statistics.

Units
-----
All program-side geometric quantities use SI units:
    position      -> meters
    dimensions    -> meters
    area          -> square meters
    polarization  -> tesla

Only a raw STL may use another unit system, declared explicitly by
``stl_units``.

Diagnostics
-----------
Each class accepts ``debug=False`` by default.

When debug is True:
    - detailed geometry decisions are logged,
    - candidate/ring statistics are logged,
    - bounds, areas, pitches, counts, and IDs are reported.

The Python logging framework is used instead of scattering conditional print
statements throughout the implementation. Normal user-facing summaries remain
available through ``print_statistics()`` and ``validate()``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import magpylib as magpy
import numpy as np

from make_shim_rings import (
    external_stl_tray,
    shim_ring,
)


# ============================================================================
# LOGGING
# ============================================================================

LOGGER_NAME = "passive_shimming.geometry"
logger = logging.getLogger(LOGGER_NAME)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.propagate = False


def _set_debug(debug: bool) -> None:
    """Set module logging level from a class debug flag."""
    logger.setLevel(
        logging.DEBUG
        if debug
        else logging.WARNING
    )


def _as_float_array(
    value,
    shape=None,
    name="array",
) -> np.ndarray:
    """Convert to float ndarray and optionally enforce shape."""
    array = np.asarray(
        value,
        dtype=float,
    )

    if (
        shape is not None
        and array.shape != shape
    ):
        raise ValueError(
            f"{name} must have shape {shape}; "
            f"received {array.shape}."
        )

    return array


def _recursive_sources(
    collection: magpy.Collection,
) -> list:
    """
    Return all leaf Magpylib sources from a possibly nested Collection.
    """
    sources = []

    for child in collection:
        if isinstance(
            child,
            magpy.Collection,
        ):
            sources.extend(
                _recursive_sources(
                    child
                )
            )
        else:
            sources.append(
                child
            )

    return sources


def _orientation_quaternion(
    source,
):
    """Return source orientation quaternion [x,y,z,w] when available."""
    orientation = getattr(
        source,
        "orientation",
        None,
    )

    if orientation is None:
        return None

    try:
        return (
            np.asarray(
                orientation.as_quat(),
                dtype=float,
            )
            .tolist()
        )
    except Exception:
        return None


# ============================================================================
# TRAY GEOMETRY
# ============================================================================

class TrayGeometry:
    """
    Describe the physical geometry of a two-plane shim tray.

    Use one of the factory constructors:

    ``TrayGeometry.from_external_stl(...)``
        Load one physical tray STL, orient it into the XY plane, mirror it
        for the opposite pole piece, and use its footprint for candidate
        filtering.

    ``TrayGeometry.circular(...)``
        Define a conventional analytic circular tray using diameter and
        explicit bottom/top plane heights.

    Parameters
    ----------
    mode : {'external_stl', 'circular'}
        Geometry representation.

    debug : bool, optional
        Enable detailed logging. Default False.

    Notes
    -----
    For external-STL mode the actual STL footprint controls the radial
    candidate search extent. No artificial circular diameter is required.
    """

    VALID_MODES = (
        "external_stl",
        "circular",
    )

    def __init__(
        self,
        mode,
        *,
        debug=False,
    ):
        self.mode = str(
            mode
        ).lower()

        self.debug = bool(
            debug
        )

        _set_debug(
            self.debug
        )

        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {self.VALID_MODES}; "
                f"received {self.mode!r}."
            )

        self.external_tray = None

        self.stl_file = None
        self.stl_units = None
        self.margin_m = None

        self.diameter_m = None
        self.heights_m = None
        self.separation_m = None

        self.usable_area_m2 = None
        self.bounds_xy_m = None

        logger.debug(
            "TrayGeometry initialized | mode=%s",
            self.mode,
        )

    @classmethod
    def from_external_stl(
        cls,
        stl_file,
        separation_m,
        stl_units="mm",
        margin_m=0.0,
        debug=False,
    ):
        """
        Construct tray geometry from one external shim-tray STL.

        Parameters
        ----------
        stl_file : str or pathlib.Path or trimesh.Trimesh
            Physical shim-tray STL, not the pole-piece STL.

        separation_m : float
            Distance between bottom and top tray INNER surfaces in meters.

        stl_units : {'mm', 'm', 'inch'}, optional
            Coordinate units of the STL. Default 'mm'.

        margin_m : float, optional
            Inward footprint margin in meters.

        debug : bool, optional
            Enable detailed diagnostics.

        Returns
        -------
        TrayGeometry
            Prepared external tray geometry.
        """
        obj = cls(
            "external_stl",
            debug=debug,
        )

        obj.stl_file = (
            Path(stl_file)
            if isinstance(
                stl_file,
                (
                    str,
                    Path,
                ),
            )
            else stl_file
        )

        obj.stl_units = str(
            stl_units
        )

        obj.margin_m = float(
            margin_m
        )

        obj.separation_m = float(
            separation_m
        )

        logger.debug(
            "Preparing external STL tray | "
            "stl=%s | separation_m=%.9f | units=%s | margin_m=%.9f",
            obj.stl_file,
            obj.separation_m,
            obj.stl_units,
            obj.margin_m,
        )

        obj.external_tray = (
            external_stl_tray(
                stl_file=
                    obj.stl_file,
                separation=
                    obj.separation_m,
                stl_units=
                    obj.stl_units,
                margin=
                    obj.margin_m,
                show=False,
            )
        )

        obj.heights_m = np.asarray(
            obj.external_tray.heights,
            dtype=float,
        )

        obj.usable_area_m2 = float(
            obj.external_tray.allowed_xy_region.area
        )

        obj.bounds_xy_m = np.asarray(
            obj.external_tray.allowed_xy_region.bounds,
            dtype=float,
        )

        logger.debug(
            "External STL prepared | heights_m=%s | "
            "usable_area_mm2=%.3f | bounds_xy_m=%s | thickness_m=%.9f",
            obj.heights_m,
            obj.usable_area_m2 * 1e6,
            obj.bounds_xy_m,
            float(
                obj.external_tray.thickness
            ),
        )

        return obj

    @classmethod
    def circular(
        cls,
        diameter_m,
        heights_m,
        debug=False,
    ):
        """
        Construct an analytic circular two-plane tray.

        Parameters
        ----------
        diameter_m : float
            Physical tray diameter in meters.

        heights_m : array-like, shape (2,)
            Bottom and top candidate plane heights in meters.

        debug : bool, optional
            Enable detailed diagnostics.

        Returns
        -------
        TrayGeometry
            Circular tray geometry.
        """
        obj = cls(
            "circular",
            debug=debug,
        )

        obj.diameter_m = float(
            diameter_m
        )

        obj.heights_m = _as_float_array(
            heights_m,
            name="heights_m",
        ).reshape(
            -1
        )

        if len(
            obj.heights_m
        ) == 2:
            obj.separation_m = float(
                abs(
                    obj.heights_m[1]
                    - obj.heights_m[0]
                )
            )

        obj.usable_area_m2 = float(
            np.pi
            * (
                obj.diameter_m / 2.0
            ) ** 2
        )

        radius = (
            obj.diameter_m
            / 2.0
        )

        obj.bounds_xy_m = np.array(
            [
                -radius,
                -radius,
                +radius,
                +radius,
            ],
            dtype=float,
        )

        logger.debug(
            "Circular tray prepared | diameter_m=%.9f | "
            "heights_m=%s | usable_area_mm2=%.3f",
            obj.diameter_m,
            obj.heights_m,
            obj.usable_area_m2 * 1e6,
        )

        return obj

    @property
    def plane_heights_m(self) -> np.ndarray:
        """Bottom/top candidate-plane heights in meters."""
        return np.asarray(
            self.heights_m,
            dtype=float,
        )

    @property
    def bottom_height_m(self) -> float:
        """Lower candidate-plane height in meters."""
        return float(
            np.min(
                self.plane_heights_m
            )
        )

    @property
    def top_height_m(self) -> float:
        """Upper candidate-plane height in meters."""
        return float(
            np.max(
                self.plane_heights_m
            )
        )

    @property
    def candidate_diameter_m(self):
        """
        Circular diameter constraint passed to ``shim_ring``.

        Returns None for external-STL mode so the STL itself defines the
        radial candidate extent.
        """
        if self.mode == "external_stl":
            return None

        return self.diameter_m

    def validate(self):
        """
        Validate the tray geometry.

        Returns
        -------
        bool
            True if validation succeeds.
        """
        _set_debug(
            self.debug
        )

        if self.mode == "external_stl":
            if self.external_tray is None:
                raise ValueError(
                    "External STL tray has not been prepared."
                )

            if self.separation_m is None or self.separation_m <= 0:
                raise ValueError(
                    "separation_m must be positive."
                )

            if self.margin_m is None or self.margin_m < 0:
                raise ValueError(
                    "margin_m must be >= 0."
                )

            if (
                self.usable_area_m2 is None
                or self.usable_area_m2 <= 0
            ):
                raise ValueError(
                    "External tray usable area is invalid."
                )

        else:
            if self.diameter_m is None or self.diameter_m <= 0:
                raise ValueError(
                    "diameter_m must be positive."
                )

        if self.heights_m is None:
            raise ValueError(
                "Tray plane heights are undefined."
            )

        if len(
            self.heights_m
        ) != 2:
            raise ValueError(
                "Current biplanar workflow requires exactly "
                "two tray heights."
            )

        if not np.all(
            np.isfinite(
                self.heights_m
            )
        ):
            raise ValueError(
                "Tray heights contain invalid values."
            )

        if not (
            self.bottom_height_m
            < self.top_height_m
        ):
            raise ValueError(
                "Bottom tray must lie below top tray."
            )

        logger.debug(
            "TrayGeometry validation passed | %s",
            self.statistics(),
        )

        print(
            "\nTray geometry validation passed"
        )
        print(
            f"  Mode: {self.mode}"
        )
        print(
            f"  Bottom plane: "
            f"{self.bottom_height_m*1e3:.3f} mm"
        )
        print(
            f"  Top plane: "
            f"{self.top_height_m*1e3:.3f} mm"
        )
        print(
            f"  Separation: "
            f"{(self.top_height_m-self.bottom_height_m)*1e3:.3f} mm"
        )
        print(
            f"  Usable area: "
            f"{self.usable_area_m2*1e6:.1f} mm^2"
        )

        return True

    def show(self):
        """Display external STL tray geometry when available."""
        if self.mode == "external_stl":
            self.external_tray.show()

        else:
            print(
                "Circular TrayGeometry has no STL mesh to display. "
                "Candidate magnets can be displayed after CandidateTray.build()."
            )

    def statistics(self) -> dict:
        """Return tray geometry diagnostics."""
        report = {
            "mode":
                self.mode,
            "plane_heights_m":
                self.plane_heights_m,
            "bottom_height_m":
                self.bottom_height_m,
            "top_height_m":
                self.top_height_m,
            "separation_m":
                float(
                    self.top_height_m
                    - self.bottom_height_m
                ),
            "usable_area_m2":
                self.usable_area_m2,
            "usable_area_mm2":
                (
                    None
                    if self.usable_area_m2 is None
                    else self.usable_area_m2
                    * 1e6
                ),
            "bounds_xy_m":
                self.bounds_xy_m,
        }

        if self.mode == "external_stl":
            report.update(
                {
                    "stl_file":
                        self.stl_file,
                    "stl_units":
                        self.stl_units,
                    "margin_m":
                        self.margin_m,
                    "tray_thickness_m":
                        float(
                            self.external_tray.thickness
                        ),
                    "original_thickness_axis":
                        getattr(
                            self.external_tray,
                            "original_thickness_axis",
                            None,
                        ),
                }
            )

        else:
            report[
                "diameter_m"
            ] = self.diameter_m

        return report

    def to_dict(self) -> dict:
        """Alias of :meth:`statistics` for run serialization."""
        return self.statistics()


# ============================================================================
# SHIM MAGNET
# ============================================================================

class ShimMagnet:
    """
    Define one physical passive-shim magnet.

    The magnetic-strength representation is Magpylib polarization only.

    Parameters
    ----------
    material : str, optional
        Human-readable magnet material/grade label, e.g. ``'N45'``.
        This is metadata only.

    dimensions_m : array-like, shape (3,)
        Cuboid dimensions ``(dx, dy, dz)`` in meters.

    polarization_T : float, optional
        Remanent polarization magnitude in tesla. Default 1.2 T.

    debug : bool, optional
        Enable detailed diagnostics.

    Notes
    -----
    The default 1.2 T value is a nominal engineering choice and may be
    changed directly through ``polarization_T`` when better magnet-specific
    measurements are available.
    """

    DEFAULT_POLARIZATION_T = 1.2

    def __init__(
        self,
        material="N45",
        dimensions_m=None,
        polarization_T=
            DEFAULT_POLARIZATION_T,
        debug=False,
    ):
        self.debug = bool(
            debug
        )

        _set_debug(
            self.debug
        )

        self.material = str(
            material
        )

        if dimensions_m is None:
            raise ValueError(
                "dimensions_m must be supplied explicitly in SI meters."
            )

        self.dimensions_m = _as_float_array(
            dimensions_m,
            shape=(3,),
            name="dimensions_m",
        )

        self.polarization_T = float(
            polarization_T
        )

        logger.debug(
            "ShimMagnet initialized | material=%s | dimensions_m=%s | "
            "polarization_T=%.6f",
            self.material,
            self.dimensions_m,
            self.polarization_T,
        )

    @property
    def polarization_vector_T(self) -> np.ndarray:
        """Nominal +Z polarization vector in tesla."""
        return np.array(
            [
                0.0,
                0.0,
                self.polarization_T,
            ],
            dtype=float,
        )

    @property
    def footprint_area_m2(self) -> float:
        """XY footprint area of one magnet."""
        return float(
            self.dimensions_m[0]
            * self.dimensions_m[1]
        )

    @property
    def footprint_area_mm2(self) -> float:
        """XY footprint area in square millimeters."""
        return (
            self.footprint_area_m2
            * 1e6
        )

    @property
    def volume_m3(self) -> float:
        """Magnet volume in cubic meters."""
        return float(
            np.prod(
                self.dimensions_m
            )
        )

    @property
    def in_plane_diagonal_m(self) -> float:
        """XY diagonal used by ring-spacing logic."""
        return float(
            np.hypot(
                self.dimensions_m[0],
                self.dimensions_m[1],
            )
        )

    def validate(self):
        """
        Validate physical dimensions and polarization.

        Returns
        -------
        bool
            True when validation succeeds.
        """
        _set_debug(
            self.debug
        )

        if self.dimensions_m.shape != (3,):
            raise ValueError(
                "dimensions_m must have shape (3,)."
            )

        if not np.all(
            np.isfinite(
                self.dimensions_m
            )
        ):
            raise ValueError(
                "Magnet dimensions contain invalid values."
            )

        if np.any(
            self.dimensions_m <= 0
        ):
            raise ValueError(
                "All magnet dimensions must be positive."
            )

        if not np.isfinite(
            self.polarization_T
        ):
            raise ValueError(
                "polarization_T is invalid."
            )

        if self.polarization_T <= 0:
            raise ValueError(
                "polarization_T must be positive."
            )

        logger.debug(
            "ShimMagnet validation passed | %s",
            self.statistics(),
        )

        print(
            "\nShim magnet validation passed"
        )
        print(
            f"  Material: "
            f"{self.material}"
        )
        print(
            f"  Dimensions: "
            f"{self.dimensions_m*1e3} mm"
        )
        print(
            f"  Polarization: "
            f"{self.polarization_T:.3f} T"
        )
        print(
            f"  Footprint area: "
            f"{self.footprint_area_mm2:.3f} mm^2"
        )

        return True

    def statistics(self) -> dict:
        """Return magnet geometry / material diagnostics."""
        return {
            "material":
                self.material,
            "dimensions_m":
                self.dimensions_m,
            "dimensions_mm":
                self.dimensions_m
                * 1e3,
            "polarization_T":
                self.polarization_T,
            "polarization_vector_T":
                self.polarization_vector_T,
            "footprint_area_m2":
                self.footprint_area_m2,
            "footprint_area_mm2":
                self.footprint_area_mm2,
            "volume_m3":
                self.volume_m3,
            "in_plane_diagonal_m":
                self.in_plane_diagonal_m,
        }

    def to_dict(self) -> dict:
        """Alias of :meth:`statistics`."""
        return self.statistics()


# ============================================================================
# CANDIDATE TRAY
# ============================================================================

class CandidateTray:
    """
    Generate and manage top/bottom candidate shim magnets.

    ``CandidateTray`` is the bridge between physical tray geometry and the
    optimization problem. It creates two ``shim_ring`` objects, one for each
    tray plane, then flattens their leaf sources into a stable ordered list.

    Parameters
    ----------
    tray_geometry : TrayGeometry
        Validated physical tray geometry.

    shim_magnet : ShimMagnet
        Validated physical shim magnet definition.

    radial_spacing_factor : float, optional
        Ring radial pitch multiplier. Default 1.0.

    azimuthal_spacing_factor : float, optional
        Arc-pitch multiplier. Default 1.25.

    symmetry : bool, optional
        If True, retain only +X,+Y quadrant candidates.

    skip_center_magnets : bool, optional
        If True, omit center pair.

    max_candidates_per_plane : int or None, optional
        Optional cap per tray. ``None`` lets geometry determine all candidates.

    require_full_magnet_inside : bool, optional
        Require full rotated magnet footprint to remain inside the tray.

    debug : bool, optional
        Enable extensive geometry diagnostics.

    Attributes
    ----------
    template : magpylib.Collection
        Nested collection containing bottom and top candidate collections.

    candidates : list
        Flat ordered leaf-source list used by the basis matrix and optimizer.

    candidate_ids : list[str]
        Stable IDs matching ``candidates``.

    candidate_trays : list[str]
        ``'bottom'`` or ``'top'`` for each candidate.
    """

    def __init__(
        self,
        tray_geometry,
        shim_magnet,
        radial_spacing_factor=1.0,
        azimuthal_spacing_factor=1.25,
        symmetry=False,
        skip_center_magnets=False,
        max_candidates_per_plane=None,
        require_full_magnet_inside=True,
        debug=False,
    ):
        self.debug = bool(
            debug
        )

        _set_debug(
            self.debug
        )

        if not isinstance(
            tray_geometry,
            TrayGeometry,
        ):
            raise TypeError(
                "tray_geometry must be a TrayGeometry object."
            )

        if not isinstance(
            shim_magnet,
            ShimMagnet,
        ):
            raise TypeError(
                "shim_magnet must be a ShimMagnet object."
            )

        self.tray_geometry = (
            tray_geometry
        )

        self.shim_magnet = (
            shim_magnet
        )

        self.radial_spacing_factor = float(
            radial_spacing_factor
        )

        self.azimuthal_spacing_factor = float(
            azimuthal_spacing_factor
        )

        self.symmetry = bool(
            symmetry
        )

        self.skip_center_magnets = bool(
            skip_center_magnets
        )

        self.max_candidates_per_plane = (
            None
            if max_candidates_per_plane is None
            else int(
                max_candidates_per_plane
            )
        )

        self.require_full_magnet_inside = bool(
            require_full_magnet_inside
        )

        self.bottom_ring = None
        self.top_ring = None

        self.template = None

        self.bottom_candidates = []
        self.top_candidates = []
        self.candidates = []

        self.candidate_ids = []
        self.candidate_trays = []
        self.candidate_metadata = []

        self._statistics = None
        self.is_built = False

        logger.debug(
            "CandidateTray initialized | radial_spacing_factor=%.6f | "
            "azimuthal_spacing_factor=%.6f | symmetry=%s | "
            "skip_center=%s | max_candidates_per_plane=%s | "
            "require_full_inside=%s",
            self.radial_spacing_factor,
            self.azimuthal_spacing_factor,
            self.symmetry,
            self.skip_center_magnets,
            self.max_candidates_per_plane,
            self.require_full_magnet_inside,
        )

    def validate(self):
        """Validate prerequisites and candidate-generation settings."""
        self.tray_geometry.validate()
        self.shim_magnet.validate()

        if self.radial_spacing_factor <= 0:
            raise ValueError(
                "radial_spacing_factor must be > 0."
            )

        if self.azimuthal_spacing_factor <= 0:
            raise ValueError(
                "azimuthal_spacing_factor must be > 0."
            )

        if (
            self.max_candidates_per_plane is not None
            and self.max_candidates_per_plane <= 0
        ):
            raise ValueError(
                "max_candidates_per_plane must be positive or None."
            )

        logger.debug(
            "CandidateTray prerequisite validation passed."
        )

        return True

    def _ring_kwargs(
        self,
    ) -> dict:
        """Common ``shim_ring`` constructor arguments."""
        return {
            "diameter":
                self.tray_geometry.candidate_diameter_m,
            "magnet_dims":
                self.shim_magnet.dimensions_m,
            "num_magnets":
                self.max_candidates_per_plane,
            "symmetry":
                self.symmetry,
            "skip_center_magnets":
                self.skip_center_magnets,
            "polarization_T":
                self.shim_magnet.polarization_T,
            "radial_spacing_factor":
                self.radial_spacing_factor,
            "azimuthal_spacing_factor":
                self.azimuthal_spacing_factor,
            "external_tray":
                self.tray_geometry.external_tray,
            "require_full_magnet_inside":
                self.require_full_magnet_inside,
        }

    def _build_ring(
        self,
        *,
        height_m,
        style_color,
        label,
    ):
        """Construct one candidate ring plane."""
        logger.debug(
            "Building %s ring | height_m=%.9f",
            label,
            height_m,
        )

        ring = shim_ring(
            height=
                height_m,
            style_color=
                style_color,
            **self._ring_kwargs(),
        )

        geometry_report = (
            ring.get_geometry_params()
        )

        collection = (
            ring.make_magnet_collection()
        )

        leaf_sources = (
            _recursive_sources(
                collection
            )
        )

        logger.debug(
            "%s ring built | leaf_candidates=%d | geometry=%s | area=%s",
            label,
            len(
                leaf_sources
            ),
            geometry_report,
            ring.area_report,
        )

        return (
            ring,
            leaf_sources,
        )

    def build(self):
        """
        Construct bottom and top candidate trays.

        Returns
        -------
        CandidateTray
            ``self``, enabling method chaining.
        """
        _set_debug(
            self.debug
        )

        self.validate()

        logger.debug(
            "CandidateTray build started."
        )

        (
            self.bottom_ring,
            self.bottom_candidates,
        ) = self._build_ring(
            height_m=
                self.tray_geometry.bottom_height_m,
            style_color=
                "orange",
            label=
                "bottom",
        )

        (
            self.top_ring,
            self.top_candidates,
        ) = self._build_ring(
            height_m=
                self.tray_geometry.top_height_m,
            style_color=
                "blue",
            label=
                "top",
        )

        self.candidates = (
            self.bottom_candidates
            + self.top_candidates
        )

        self.candidate_ids = (
            [
                f"bottom_{index:04d}"
                for index in range(
                    len(
                        self.bottom_candidates
                    )
                )
            ]
            + [
                f"top_{index:04d}"
                for index in range(
                    len(
                        self.top_candidates
                    )
                )
            ]
        )

        self.candidate_trays = (
            [
                "bottom"
                for _ in self.bottom_candidates
            ]
            + [
                "top"
                for _ in self.top_candidates
            ]
        )

        self.template = magpy.Collection(
            style_label=
                "shim_candidates"
        )

        self.template.add(
            self.bottom_ring.collection
        )

        self.template.add(
            self.top_ring.collection
        )

        self._build_candidate_metadata()
        self._compute_statistics()

        self.is_built = True

        logger.debug(
            "CandidateTray build complete | total=%d | bottom=%d | top=%d",
            len(
                self.candidates
            ),
            len(
                self.bottom_candidates
            ),
            len(
                self.top_candidates
            ),
        )

        return self

    def _build_candidate_metadata(self):
        """Create stable per-candidate metadata for later basis/reporting use."""
        metadata = []

        for (
            candidate_id,
            tray_name,
            magnet,
        ) in zip(
            self.candidate_ids,
            self.candidate_trays,
            self.candidates,
        ):
            position = np.asarray(
                magnet.position,
                dtype=float,
            )

            dimension = np.asarray(
                magnet.dimension,
                dtype=float,
            )

            polarization = np.asarray(
                magnet.polarization,
                dtype=float,
            )

            item = {
                "candidate_id":
                    candidate_id,
                "tray":
                    tray_name,
                "position_m":
                    position,
                "position_mm":
                    position * 1e3,
                "dimension_m":
                    dimension,
                "dimension_mm":
                    dimension * 1e3,
                "polarization_T":
                    polarization,
                "orientation_quaternion_xyzw":
                    _orientation_quaternion(
                        magnet
                    ),
                "radial_distance_m":
                    float(
                        np.hypot(
                            position[0],
                            position[1],
                        )
                    ),
                "azimuth_deg":
                    float(
                        np.degrees(
                            np.arctan2(
                                position[1],
                                position[0],
                            )
                        )
                        % 360.0
                    ),
            }

            metadata.append(
                item
            )

            logger.debug(
                "Candidate %s | tray=%s | pos_mm=%s | "
                "r_mm=%.3f | theta_deg=%.3f | pol=%s",
                candidate_id,
                tray_name,
                item[
                    "position_mm"
                ],
                item[
                    "radial_distance_m"
                ] * 1e3,
                item[
                    "azimuth_deg"
                ],
                polarization,
            )

        self.candidate_metadata = (
            metadata
        )

    def _compute_statistics(self):
        """Compute candidate-density and top/bottom geometry diagnostics."""
        tray_area_m2 = float(
            self.tray_geometry.usable_area_m2
        )

        magnet_area_m2 = (
            self.shim_magnet.footprint_area_m2
        )

        n_total = len(
            self.candidates
        )

        n_bottom = len(
            self.bottom_candidates
        )

        n_top = len(
            self.top_candidates
        )

        area_limited_capacity = int(
            np.floor(
                tray_area_m2
                / magnet_area_m2
            )
        )

        # This is a diagnostic, not a rigorous packing efficiency if
        # candidate footprints overlap.
        candidate_area_fraction_per_plane = (
            (
                max(
                    n_bottom,
                    n_top,
                )
                * magnet_area_m2
                / tray_area_m2
            )
            if tray_area_m2 > 0
            else np.nan
        )

        total_candidate_footprint_area_m2 = (
            n_total
            * magnet_area_m2
        )

        positions = np.asarray(
            [
                source.position
                for source in self.candidates
            ],
            dtype=float,
        )

        radial_distances = np.hypot(
            positions[
                :,
                0,
            ],
            positions[
                :,
                1,
            ],
        )

        self._statistics = {
            "n_candidates_total":
                n_total,
            "n_candidates_bottom":
                n_bottom,
            "n_candidates_top":
                n_top,
            "candidate_balance_difference":
                abs(
                    n_bottom
                    - n_top
                ),
            "tray_area_m2":
                tray_area_m2,
            "tray_area_mm2":
                tray_area_m2
                * 1e6,
            "magnet_footprint_area_m2":
                magnet_area_m2,
            "magnet_footprint_area_mm2":
                magnet_area_m2
                * 1e6,
            "area_limited_capacity_per_plane":
                area_limited_capacity,
            "candidate_area_fraction_per_plane":
                candidate_area_fraction_per_plane,
            "candidate_area_percent_per_plane":
                candidate_area_fraction_per_plane
                * 100.0,
            "total_candidate_footprint_area_m2":
                total_candidate_footprint_area_m2,
            "radial_spacing_factor":
                self.radial_spacing_factor,
            "azimuthal_spacing_factor":
                self.azimuthal_spacing_factor,
            "radial_pitch_m":
                self.bottom_ring.radial_pitch,
            "azimuthal_pitch_m":
                self.bottom_ring.azimuthal_pitch,
            "bottom_num_rings":
                self.bottom_ring.num_rings,
            "top_num_rings":
                self.top_ring.num_rings,
            "bottom_max_center_radius_m":
                self.bottom_ring.max_center_radius,
            "top_max_center_radius_m":
                self.top_ring.max_center_radius,
            "actual_min_candidate_radius_m":
                float(
                    np.min(
                        radial_distances
                    )
                )
                if n_total > 0
                else None,
            "actual_max_candidate_radius_m":
                float(
                    np.max(
                        radial_distances
                    )
                )
                if n_total > 0
                else None,
            "bottom_area_report":
                self.bottom_ring.area_report,
            "top_area_report":
                self.top_ring.area_report,
            "symmetry":
                self.symmetry,
            "skip_center_magnets":
                self.skip_center_magnets,
            "max_candidates_per_plane":
                self.max_candidates_per_plane,
            "require_full_magnet_inside":
                self.require_full_magnet_inside,
        }

        logger.debug(
            "CandidateTray statistics | %s",
            self._statistics,
        )

    def statistics(self) -> dict:
        """
        Return candidate geometry statistics.

        The detailed per-candidate metadata is included because it is needed
        later to map field-basis columns and optimizer states back to physical
        tray locations.
        """
        if not self.is_built and self._statistics is None:
            raise RuntimeError(
                "CandidateTray.build() must be called first."
            )

        report = dict(
            self._statistics
        )

        report[
            "candidate_metadata"
        ] = self.candidate_metadata

        return report

    def print_statistics(self):
        """Print concise user-facing candidate geometry diagnostics."""
        if self._statistics is None:
            raise RuntimeError(
                "CandidateTray.build() must be called first."
            )

        stats = self._statistics

        print(
            "\nCandidate tray statistics"
        )
        print(
            f"  Total candidates: "
            f"{stats['n_candidates_total']}"
        )
        print(
            f"  Bottom candidates: "
            f"{stats['n_candidates_bottom']}"
        )
        print(
            f"  Top candidates: "
            f"{stats['n_candidates_top']}"
        )
        print(
            f"  Number of rings per plane: "
            f"bottom={stats['bottom_num_rings']}, "
            f"top={stats['top_num_rings']}"
        )
        print(
            f"  Radial pitch: "
            f"{stats['radial_pitch_m']*1e3:.3f} mm"
        )
        print(
            f"  Azimuthal target pitch: "
            f"{stats['azimuthal_pitch_m']*1e3:.3f} mm"
        )
        print(
            f"  Maximum actual candidate radius: "
            f"{stats['actual_max_candidate_radius_m']*1e3:.3f} mm"
        )
        print(
            f"  Tray area: "
            f"{stats['tray_area_mm2']:.1f} mm^2"
        )
        print(
            f"  Magnet footprint: "
            f"{stats['magnet_footprint_area_mm2']:.3f} mm^2"
        )
        print(
            f"  Area-only capacity per plane: "
            f"{stats['area_limited_capacity_per_plane']}"
        )
        print(
            f"  Candidate footprint / tray area per plane: "
            f"{stats['candidate_area_percent_per_plane']:.1f}%"
        )

        if (
            stats[
                "candidate_balance_difference"
            ]
            != 0
        ):
            print(
                f"  NOTE: top/bottom candidate-count difference = "
                f"{stats['candidate_balance_difference']}"
            )

    def show(self):
        """Display top and bottom candidate magnets."""
        if not self.is_built:
            raise RuntimeError(
                "CandidateTray.build() must be called first."
            )

        magpy.show(
            self.template
        )

    def to_dict(self) -> dict:
        """Alias of :meth:`statistics`."""
        return self.statistics()
