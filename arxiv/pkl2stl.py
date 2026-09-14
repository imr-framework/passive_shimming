import numpy as np
import magpylib as magpy
import pickle
import trimesh


class pkl2stl:
    """
    Convert an optimized Magpylib magnet-collection PKL into two shim-tray STL files.

    Geometry conventions
    --------------------
    disc:
        tray plane            = XY
        separation axis       = Z
        magnets split by      = +/- Z position
        scanner labels        = X+, X-, Y+, Y-
        alignment notch       = X=0, Y=-Ymax

    rectangle:
        tray plane            = YZ
        separation axis       = X
        magnets split by      = +/- X position
        scanner labels        = Y+, Y-, Z+, Z-
        alignment notch       = Y=0, Z=-Zmax

    Automatic tray thickness
    ------------------------
    One-shim thickness is read directly from each Cuboid dimension along the
    tray normal. Cuboids sharing one in-plane location on one tray side are
    grouped into a physical shim stack, and their thicknesses are summed.

        tray thickness = maximum shim thickness + thickness_clearance_mm

    Each magnet pocket is then cut only as deep as required for that magnet:

        pocket depth = local shim thickness + thickness_clearance_mm

    Therefore thin magnets retain material behind them while the thickest
    magnet determines the overall tray thickness.

    Surface-specific polarity marking
    ---------------------------------
    For stacked shims, polarity is evaluated separately at the two physical
    tray faces using the ORIGINAL PKL positions along the separation axis.
    Only the shim closest to a given face controls the mark on that face.

        positive surface-adjacent shim -> engrave '+'
        negative surface-adjacent shim -> leave that face unmarked

    This allows a two-shim stack with opposite polarities to be populated
    correctly from either side of the tray.

    Automatic rectangular tray size
    --------------------------------
    For rectangular YZ trays, if rectangle_y_mm and/or rectangle_z_mm are None,
    the corresponding dimension is inferred from the outermost oriented
    cuboid bounds while keeping scanner coordinate (0,0) at the tray center.

        half-size = max(abs(min bound), abs(max bound)) + rectangle_margin_mm

    rectangle_margin_mm is applied on EACH side. The default is 5 mm.

    Units
    -----
    Internal geometry is stored in meters.
    Constructor dimensions and clearances are specified in millimeters.
    """

    AXIS_INDEX = {
        "x": 0,
        "y": 1,
        "z": 2,
    }

    def __init__(
        self,
        pkl_filename,
        tray_shape="disc",
        separation_axis=None,
        disc_diameter_mm=152.4,
        rectangle_y_mm=None,
        rectangle_z_mm=None,
        rectangle_margin_mm=5.0,
        thickness_clearance_mm=0.1,
        notch_diameter_mm=1.5,
        polarity_engraving_depth_mm=0.85,
        orientation_engraving_depth_mm=0.85,
        boolean_overlap_mm=0.05,
        slot_group_tolerance_mm=0.25,
        boolean_engine=None,
    ):
        self.pkl_filename = pkl_filename
        self.tray_shape = self._normalize_shape(tray_shape)

        if separation_axis is None:
            separation_axis = (
                "z"
                if self.tray_shape == "disc"
                else "x"
            )

        self.separation_axis = separation_axis.lower()

        expected_axis = (
            "z"
            if self.tray_shape == "disc"
            else "x"
        )

        if self.separation_axis != expected_axis:
            raise ValueError(
                f"For tray_shape='{self.tray_shape}', separation_axis must be "
                f"'{expected_axis}'."
            )

        self.normal_index = self.AXIS_INDEX[
            self.separation_axis
        ]

        if self.tray_shape == "disc":
            self.plane_axes = ("x", "y")
        else:
            self.plane_axes = ("y", "z")

        self.u_axis, self.v_axis = self.plane_axes
        self.u_index = self.AXIS_INDEX[self.u_axis]
        self.v_index = self.AXIS_INDEX[self.v_axis]

        # User-specified/fixed dimensions.
        self.disc_diameter = (
            float(disc_diameter_mm) * 1e-3
        )

        self.rectangle_y_requested = (
            None
            if rectangle_y_mm is None
            else float(rectangle_y_mm) * 1e-3
        )

        self.rectangle_z_requested = (
            None
            if rectangle_z_mm is None
            else float(rectangle_z_mm) * 1e-3
        )

        self.rectangle_margin = (
            float(rectangle_margin_mm) * 1e-3
        )

        # The tray thickness is deliberately NOT supplied by the user.
        self.thickness_clearance = (
            float(thickness_clearance_mm) * 1e-3
        )

        self.notch_radius = (
            float(notch_diameter_mm)
            * 0.5e-3
        )

        self.polarity_engraving_depth = (
            float(polarity_engraving_depth_mm)
            * 1e-3
        )

        self.orientation_engraving_depth = (
            float(orientation_engraving_depth_mm)
            * 1e-3
        )

        self.boolean_overlap = (
            float(boolean_overlap_mm)
            * 1e-3
        )

        # Magnets at the same in-plane location are treated as one physical
        # shim slot/stack. This is how one-shim and two-shim locations are
        # distinguished when the PKL stores each shim as its own Cuboid.
        self.slot_group_tolerance = (
            float(slot_group_tolerance_mm)
            * 1e-3
        )

        self.boolean_engine = boolean_engine

        self.positive_color = np.array(
            [110, 180, 230, 255],
            dtype=np.uint8,
        )

        self.negative_color = np.array(
            [240, 180, 90, 255],
            dtype=np.uint8,
        )

        # Populated after load().
        self.magnets = None
        self.positive_side = None
        self.negative_side = None

        self.tray_thickness = None
        self.max_shim_thickness = None

        self.rectangle_y_auto = None
        self.rectangle_z_auto = None
        self.rectangle_y = None
        self.rectangle_z = None

        self.magnet_normal_thicknesses = []
        self.slot_groups_all = []

    # ==================================================================
    # Basic helpers
    # ==================================================================

    @staticmethod
    def _normalize_shape(shape):
        shape = shape.strip().lower()

        if shape in {
            "disk",
            "circle",
            "circular",
        }:
            shape = "disc"

        if shape in {
            "rect",
            "rectangular",
            "slab",
        }:
            shape = "rectangle"

        if shape not in {
            "disc",
            "rectangle",
        }:
            raise ValueError(
                "tray_shape must be 'disc' or 'rectangle'."
            )

        return shape

    def _world_to_plane(self, xyz):
        """
        Convert scanner/world XYZ coordinates to local tray-plane [U,V].

        disc      : U=X, V=Y
        rectangle : U=Y, V=Z
        """
        xyz = np.asarray(
            xyz,
            dtype=float,
        )

        return np.array([
            xyz[self.u_index],
            xyz[self.v_index],
        ])

    def _plane_to_world(
        self,
        uv,
        normal=0.0,
    ):
        """
        Convert local tray coordinates [U,V,N] back to scanner/world XYZ.
        """
        uv = np.asarray(
            uv,
            dtype=float,
        )

        xyz = np.zeros(
            3,
            dtype=float,
        )

        xyz[self.u_index] = uv[0]
        xyz[self.v_index] = uv[1]
        xyz[self.normal_index] = normal

        return xyz

    def _tray_half_sizes(self):
        if self.tray_shape == "disc":
            radius = (
                self.disc_diameter / 2
            )

            return radius, radius

        self._require_inferred_geometry()

        return (
            self.rectangle_y / 2,
            self.rectangle_z / 2,
        )

    def _require_inferred_geometry(self):
        if self.tray_thickness is None:
            raise RuntimeError(
                "Tray geometry has not been inferred yet. "
                "Call load() first."
            )

    # ==================================================================
    # Magnet geometry inference
    # ==================================================================

    @staticmethod
    def _oriented_cuboid_at_origin(magnet):
        """
        Create the physical Cuboid mesh at the origin and apply the
        Magpylib orientation.
        """
        mesh = trimesh.creation.box(
            extents=np.asarray(
                magnet.dimension,
                dtype=float,
            )
        )

        if magnet.orientation is not None:
            transform = np.eye(4)
            transform[:3, :3] = (
                magnet.orientation.as_matrix()
            )
            mesh.apply_transform(
                transform
            )

        return mesh

    def _magnet_normal_thickness(
        self,
        magnet,
    ):
        """
        Physical shim-stack thickness along the tray-normal axis.

        IMPORTANT
        ---------
        The shim-stack thickness is read DIRECTLY from magnet.dimension
        along the tray-normal/separation axis:

            disc      -> dimension[2] (Z thickness)
            rectangle -> dimension[0] (X thickness)

        We deliberately do NOT infer thickness from the orientation-
        transformed bounding box. Magpylib orientation is needed to
        represent the magnetization/polarity direction, but rotating the
        bounding box can make an in-plane cuboid dimension appear to be
        the tray thickness and therefore grossly overestimate both tray
        thickness and pocket depth.

        This direct dimension therefore preserves the PKL meaning:
            one shim  -> one-shim thickness
            two shims -> doubled thickness
            no shim   -> no cuboid / no pocket
        """
        dimension = np.asarray(
            magnet.dimension,
            dtype=float,
        )

        return float(
            dimension[self.normal_index]
        )

    def _magnet_world_mesh(
        self,
        magnet,
    ):
        """
        Oriented magnet mesh at its original scanner/world position.
        Used for automatic tray-size inference.
        """
        mesh = self._oriented_cuboid_at_origin(
            magnet
        )

        mesh.apply_translation(
            np.asarray(
                magnet.position,
                dtype=float,
            )
        )

        return mesh

    def _group_magnets_into_slots(
        self,
        magnets,
        already_split=False,
    ):
        """
        Group Cuboids that represent the same physical tray slot.

        A two-shim location is commonly stored in the PKL as two Cuboids
        with the same in-plane center and the same tray side, but different
        positions along the separation axis. The total required pocket depth
        is therefore the SUM of the individual Cuboid shim thicknesses.
        """
        groups = []

        for magnet in magnets:
            position = np.asarray(magnet.position, dtype=float)
            uv = self._world_to_plane(position)
            side = np.sign(position[self.normal_index])
            if side == 0:
                side = -1

            matched = None
            for group in groups:
                if not already_split and group["side"] != side:
                    continue

                if np.linalg.norm(uv - group["center_uv"]) <= self.slot_group_tolerance:
                    matched = group
                    break

            if matched is None:
                groups.append({
                    "magnets": [magnet],
                    "representative": magnet,
                    "center_uv": uv.copy(),
                    "side": side,
                })
            else:
                matched["magnets"].append(magnet)
                centers = [self._world_to_plane(m.position) for m in matched["magnets"]]
                matched["center_uv"] = np.mean(centers, axis=0)

        for group in groups:
            thicknesses = [
                self._magnet_normal_thickness(magnet)
                for magnet in group["magnets"]
            ]
            group["shim_thicknesses"] = thicknesses
            group["stack_count"] = len(group["magnets"])
            group["stack_thickness"] = float(np.sum(thicknesses))

        groups.sort(key=lambda g: (g["side"], g["center_uv"][0], g["center_uv"][1]))
        return groups

    def _print_slot_summary(self, groups, title):
        """Print single/double/multi-shim slot counts and stack thicknesses."""
        print(f"\n{title}:")
        if not groups:
            print("  no occupied slots")
            return

        histogram = {}
        for group in groups:
            n = group["stack_count"]
            histogram[n] = histogram.get(n, 0) + 1

        print(f"  Occupied physical slots: {len(groups)}")
        print("  Stack-count histogram: " + ", ".join(
            f"{n} shim(s): {count} slot(s)"
            for n, count in sorted(histogram.items())
        ))

        unique_stack_mm = sorted({
            round(group["stack_thickness"] * 1000.0, 4)
            for group in groups
        })
        print("  Stack thicknesses present: " + ", ".join(
            f"{value:.4f} mm" for value in unique_stack_mm
        ))


    def _print_original_pkl_debug(self):
        """
        Print a source-by-source / physical-slot summary directly from
        the ORIGINAL loaded PKL before projection into separate trays.

        This is intended to answer three questions explicitly:

        1. Does the PKL really contain 262 individual Cuboid sources?
        2. Do +normal and -normal source counts sum back to the PKL total?
        3. At repeated in-plane locations, do two 3.18-mm Cuboids become
           one physical 6.36-mm stack?

        For a rectangular YZ tray the relevant physical thickness is
        magnet.dimension[0] because X is the tray normal.
        """
        if self.magnets is None:
            return

        axis_name = self.separation_axis.upper()

        total_sources = len(self.magnets)

        positive_sources = [
            m for m in self.magnets
            if np.asarray(m.position, dtype=float)[self.normal_index] > 0
        ]

        negative_sources = [
            m for m in self.magnets
            if np.asarray(m.position, dtype=float)[self.normal_index] <= 0
        ]

        groups = self._group_magnets_into_slots(
            self.magnets,
            already_split=False,
        )

        histogram = {}
        for group in groups:
            n = group["stack_count"]
            histogram[n] = histogram.get(n, 0) + 1

        sources_reconstructed = sum(
            stack_count * slot_count
            for stack_count, slot_count in histogram.items()
        )

        print(
            "\n============================================================"
        )
        print(
            "ORIGINAL PKL SOURCE / STACK VERIFICATION"
        )
        print(
            "============================================================"
        )

        print(
            f"Original PKL Cuboid sources: {total_sources}"
        )

        print(
            f"+{axis_name} original sources: {len(positive_sources)}"
        )

        print(
            f"-{axis_name} original sources: {len(negative_sources)}"
        )

        split_sum = (
            len(positive_sources)
            + len(negative_sources)
        )

        print(
            f"Source-count check: "
            f"{len(positive_sources)} + {len(negative_sources)} "
            f"= {split_sum} "
            f"{'PASS' if split_sum == total_sources else 'FAIL'}"
        )

        print(
            f"Occupied physical slots inferred directly from original PKL: "
            f"{len(groups)}"
        )

        print(
            "Original PKL stack-count histogram: "
            + ", ".join(
                f"{n}-shim={count}"
                for n, count in sorted(histogram.items())
            )
        )

        print(
            f"Reconstructed source count from stack histogram: "
            f"{sources_reconstructed} "
            f"{'PASS' if sources_reconstructed == total_sources else 'FAIL'}"
        )

        if self.tray_shape == "rectangle":
            print(
                "Thickness convention for this rectangular tray: "
                "dimension[0] = X thickness."
            )
        else:
            print(
                "Thickness convention for this disc tray: "
                "dimension[2] = Z thickness."
            )

        print(
            "\nDetailed ORIGINAL PKL slot comparison:"
        )

        for index, group in enumerate(groups):
            source_positions_normal_mm = [
                float(
                    np.asarray(
                        magnet.position,
                        dtype=float,
                    )[self.normal_index]
                    * 1000.0
                )
                for magnet in group["magnets"]
            ]

            source_dimensions_mm = [
                [
                    float(value * 1000.0)
                    for value in np.asarray(
                        magnet.dimension,
                        dtype=float,
                    )
                ]
                for magnet in group["magnets"]
            ]

            individual_thicknesses_mm = [
                float(
                    self._magnet_normal_thickness(
                        magnet
                    )
                    * 1000.0
                )
                for magnet in group["magnets"]
            ]

            stack_thickness_mm = (
                group["stack_thickness"]
                * 1000.0
            )

            side_text = (
                f"+{axis_name}"
                if group["side"] > 0
                else f"-{axis_name}"
            )

            print(
                f"  ORIGINAL SLOT {index:03d}: "
                f"{side_text} | "
                f"{self.u_axis.upper()}="
                f"{group['center_uv'][0] * 1000:.2f} mm, "
                f"{self.v_axis.upper()}="
                f"{group['center_uv'][1] * 1000:.2f} mm | "
                f"{group['stack_count']} source(s) | "
                f"{axis_name} source positions="
                f"{[round(v, 3) for v in source_positions_normal_mm]} mm | "
                f"raw dimensions XYZ="
                f"{[[round(v, 3) for v in dims] for dims in source_dimensions_mm]} mm | "
                f"thickness components="
                f"{[round(v, 3) for v in individual_thicknesses_mm]} mm | "
                f"PHYSICAL STACK={stack_thickness_mm:.3f} mm"
            )

        print(
            "\nExpected rectangular interpretation:"
            if self.tray_shape == "rectangle"
            else "\nExpected disc interpretation:"
        )

        if self.tray_shape == "rectangle":
            print(
                "  one shim  : 3.18 x 6.35 x 6.35 mm approximately"
            )
            print(
                "  two shims : 6.36 x 6.35 x 6.35 mm approximately"
            )
            print(
                "  (X thickness doubles; Y and Z footprint remain unchanged)"
            )
        else:
            print(
                "  stack thickness changes along Z; XY footprint remains unchanged"
            )

        print(
            "============================================================\n"
        )

    def _infer_geometry(self):
        """
        Infer tray geometry from PHYSICAL SLOT STACKS, not single Cuboids.

        If two 3.18-mm Cuboids occupy the same in-plane slot on one tray
        side, that slot is a 6.36-mm shim stack. The tray thickness is
        therefore max(stack thickness) + clearance.
        """
        if self.magnets is None:
            raise RuntimeError("Magnets must be loaded before geometry can be inferred.")
        if len(self.magnets) == 0:
            raise ValueError("The loaded magnet collection is empty.")

        self.magnet_normal_thicknesses = [
            self._magnet_normal_thickness(magnet)
            for magnet in self.magnets
        ]

        self.slot_groups_all = self._group_magnets_into_slots(
            self.magnets,
            already_split=False,
        )

        if not self.slot_groups_all:
            raise ValueError("No occupied shim slots could be inferred.")

        self.max_shim_thickness = max(
            group["stack_thickness"]
            for group in self.slot_groups_all
        )
        self.tray_thickness = self.max_shim_thickness + self.thickness_clearance

        u_min = np.inf
        u_max = -np.inf
        v_min = np.inf
        v_max = -np.inf

        for magnet in self.magnets:
            world_mesh = self._magnet_world_mesh(magnet)
            bounds = world_mesh.bounds
            u_min = min(u_min, bounds[0, self.u_index])
            u_max = max(u_max, bounds[1, self.u_index])
            v_min = min(v_min, bounds[0, self.v_index])
            v_max = max(v_max, bounds[1, self.v_index])

        if self.tray_shape == "rectangle":
            half_u_required = max(abs(u_min), abs(u_max)) + self.rectangle_margin
            half_v_required = max(abs(v_min), abs(v_max)) + self.rectangle_margin
            self.rectangle_y_auto = 2 * half_u_required
            self.rectangle_z_auto = 2 * half_v_required
            self.rectangle_y = self.rectangle_y_auto if self.rectangle_y_requested is None else self.rectangle_y_requested
            self.rectangle_z = self.rectangle_z_auto if self.rectangle_z_requested is None else self.rectangle_z_requested
            self._validate_requested_rectangle_size(u_min, u_max, v_min, v_max)

    def _validate_requested_rectangle_size(
        self,
        u_min,
        u_max,
        v_min,
        v_max,
    ):
        """
        Ensure any explicitly requested rectangle dimensions contain
        all magnets. A warning is emitted if they contain the magnets
        but provide less than the requested automatic marking margin.
        """
        if self.tray_shape != "rectangle":
            return

        magnet_half_y = max(
            abs(u_min),
            abs(u_max),
        )

        magnet_half_z = max(
            abs(v_min),
            abs(v_max),
        )

        if self.rectangle_y / 2 < magnet_half_y:
            raise ValueError(
                "Requested rectangle Y dimension is too small to contain "
                "the magnet collection."
            )

        if self.rectangle_z / 2 < magnet_half_z:
            raise ValueError(
                "Requested rectangle Z dimension is too small to contain "
                "the magnet collection."
            )

        if (
            self.rectangle_y_requested is not None
            and self.rectangle_y
            < self.rectangle_y_auto
        ):
            print(
                "WARNING: requested rectangle Y dimension is smaller than "
                "the automatically recommended size with the full marking margin."
            )

        if (
            self.rectangle_z_requested is not None
            and self.rectangle_z
            < self.rectangle_z_auto
        ):
            print(
                "WARNING: requested rectangle Z dimension is smaller than "
                "the automatically recommended size with the full marking margin."
            )

    # ==================================================================
    # Loading and splitting
    # ==================================================================

    def load(
        self,
        show_original=True,
    ):
        """
        Load PKL collection and immediately infer tray geometry.
        """
        with open(
            self.pkl_filename,
            "rb",
        ) as file:
            self.magnets = pickle.load(
                file
            )

        print(
            f"Number of magnets in optimized shim collection: "
            f"{len(self.magnets)}"
        )

        self._infer_geometry()
        self.report_geometry()

        # Print the same stack interpretation directly from the ORIGINAL
        # PKL before any projection into separate tray collections.
        self._print_original_pkl_debug()

        if show_original:
            print(
                "\nOpening ORIGINAL PKL visualization. "
                "This window contains every individual Cuboid source "
                "at its original scanner position."
            )
            self.magnets.show()

        return self.magnets

    def report_geometry(self):
        """
        Print automatically inferred tray dimensions.
        """
        self._require_inferred_geometry()

        print(
            "\nInferred shim/tray geometry:"
        )

        print(
            f"  Tray shape: "
            f"{self.tray_shape}"
        )

        print(
            f"  Separation axis: "
            f"{self.separation_axis.upper()}"
        )

        print(
            f"  Tray plane: "
            f"{self.u_axis.upper()}"
            f"{self.v_axis.upper()}"
        )

        print(
            f"  Maximum PHYSICAL SHIM-STACK thickness: "
            f"{self.max_shim_thickness * 1000:.3f} mm"
        )

        if self.tray_shape == "rectangle":
            print(
                "  Rectangle thickness source: "
                "magnet.dimension[0] (X), grouped/summed at identical YZ slots"
            )
        else:
            print(
                "  Disc thickness source: "
                "magnet.dimension[2] (Z), grouped/summed at identical XY slots"
            )

        print(
            f"  Thickness clearance: "
            f"{self.thickness_clearance * 1000:.3f} mm"
        )

        print(
            f"  Inferred tray thickness: "
            f"{self.tray_thickness * 1000:.3f} mm"
        )

        individual_mm = np.asarray(self.magnet_normal_thicknesses, dtype=float) * 1000.0
        unique_individual, individual_counts = np.unique(
            np.round(individual_mm, 4),
            return_counts=True,
        )
        print("  Individual Cuboid thicknesses in PKL: " + ", ".join(
            f"{value:.4f} mm ({count} cuboids)"
            for value, count in zip(unique_individual, individual_counts)
        ))
        self._print_slot_summary(
            self.slot_groups_all,
            "Physical shim stacks inferred from PKL",
        )
        print("  Tray thickness above is based on the THICKEST SLOT STACK.")

        if self.tray_shape == "disc":
            print(
                f"  Disc diameter: "
                f"{self.disc_diameter * 1000:.3f} mm"
            )

        else:
            print(
                f"  Auto rectangle Y size "
                f"(includes {self.rectangle_margin * 1000:.1f} mm "
                f"margin per side): "
                f"{self.rectangle_y_auto * 1000:.3f} mm"
            )

            print(
                f"  Auto rectangle Z size "
                f"(includes {self.rectangle_margin * 1000:.1f} mm "
                f"margin per side): "
                f"{self.rectangle_z_auto * 1000:.3f} mm"
            )

            print(
                f"  Rectangle Y size used: "
                f"{self.rectangle_y * 1000:.3f} mm"
            )

            print(
                f"  Rectangle Z size used: "
                f"{self.rectangle_z * 1000:.3f} mm"
            )

    def split_magnets(self):
        """
        Split magnets by sign of the separation-axis coordinate.

        disc      : +Z / -Z
        rectangle : +X / -X

        For STL generation the separation-axis coordinate is projected
        to zero while all in-plane scanner coordinates are retained.
        """
        if self.magnets is None:
            self.load(
                show_original=False
            )

        self.positive_side = magpy.Collection(
            style_label=(
                f"positive_"
                f"{self.separation_axis}_shims"
            )
        )

        self.negative_side = magpy.Collection(
            style_label=(
                f"negative_"
                f"{self.separation_axis}_shims"
            )
        )

        for magnet in self.magnets:
            original_position = np.asarray(
                magnet.position,
                dtype=float,
            ).copy()

            stored_position = (
                original_position.copy()
            )

            stored_position[
                self.normal_index
            ] = 0.0

            cuboid = magpy.magnet.Cuboid(
                magnetization=np.asarray(
                    magnet.magnetization,
                    dtype=float,
                ).copy(),
                dimension=np.asarray(
                    magnet.dimension,
                    dtype=float,
                ).copy(),
                position=stored_position,
                orientation=magnet.orientation,
            )

            if (
                original_position[
                    self.normal_index
                ]
                > 0
            ):
                self.positive_side.add(
                    cuboid
                )

            else:
                self.negative_side.add(
                    cuboid
                )

        print(
            f"\nTotal magnets: "
            f"{len(self.positive_side) + len(self.negative_side)}"
        )

        print(
            f"+{self.separation_axis.upper()} tray magnets: "
            f"{len(self.positive_side)}"
        )

        print(
            f"-{self.separation_axis.upper()} tray magnets: "
            f"{len(self.negative_side)}"
        )

        split_source_count = (
            len(self.positive_side)
            + len(self.negative_side)
        )

        original_source_count = len(
            self.magnets
        )

        print(
            f"Split source-count check: "
            f"{len(self.positive_side)} + {len(self.negative_side)} "
            f"= {split_source_count}; original PKL = "
            f"{original_source_count} -> "
            f"{'PASS' if split_source_count == original_source_count else 'FAIL'}"
        )

        if split_source_count != original_source_count:
            raise RuntimeError(
                "Source-count mismatch after splitting magnets."
            )

        # --------------------------------------------------------------
        # DEBUG / VERIFICATION:
        # After creating the two tray-side collections, inspect the
        # physical shim stacks before any STL Boolean operations.
        #
        # IMPORTANT for rectangle:
        #   tray normal = X
        #   shim thickness = magnet.dimension[0]
        #
        # Two Cuboids at the same YZ location on the same X side are
        # treated as a two-shim stack, so their X thicknesses are SUMMED.
        # --------------------------------------------------------------
        positive_groups = self._group_magnets_into_slots(
            self.positive_side,
            already_split=True,
        )

        negative_groups = self._group_magnets_into_slots(
            self.negative_side,
            already_split=True,
        )

        normal_axis_name = self.separation_axis.upper()

        print(
            f"\n=== {normal_axis_name}-THICKNESS CHECK AFTER TRAY SPLIT ==="
        )

        for side_name, groups in (
            (f"+{normal_axis_name}", positive_groups),
            (f"-{normal_axis_name}", negative_groups),
        ):
            print(
                f"\n{side_name} tray: "
                f"{len(groups)} occupied physical slot(s)"
            )

            count_histogram = {}
            for group in groups:
                n = group["stack_count"]
                count_histogram[n] = count_histogram.get(n, 0) + 1

            if count_histogram:
                print(
                    "  Stack-count summary: "
                    + ", ".join(
                        f"{n}-shim={count}"
                        for n, count in sorted(count_histogram.items())
                    )
                )

            for index, group in enumerate(groups):
                dims_mm = [
                    np.asarray(m.dimension, dtype=float)[self.normal_index]
                    * 1000.0
                    for m in group["magnets"]
                ]

                total_mm = sum(dims_mm)

                print(
                    f"  SLOT {index:03d}: "
                    f"{self.u_axis.upper()}={group['center_uv'][0]*1000:.2f} mm, "
                    f"{self.v_axis.upper()}={group['center_uv'][1]*1000:.2f} mm | "
                    f"{group['stack_count']} shim(s) | "
                    f"raw dimension[{self.normal_index}] = "
                    f"{[round(v, 4) for v in dims_mm]} mm | "
                    f"TOTAL {normal_axis_name} THICKNESS = "
                    f"{total_mm:.4f} mm"
                )

        print(
            "\nNOTE: Empty physical slots do not appear as Cuboids in the "
            "optimized PKL, so only occupied one-/two-/multi-shim slots can "
            "be printed from this file alone."
        )

        return (
            self.positive_side,
            self.negative_side,
        )

    def _make_physical_stack_collection(
        self,
        source_collection,
        label,
        color,
    ):
        """
        Build a DISPLAY-ONLY Magpylib collection containing ONE Cuboid
        per physical slot.

        This fixes an important visualization ambiguity:
        after split_magnets(), all source Cuboids are projected onto the
        local tray plane. Two source Cuboids at a double-shim location
        therefore occupy exactly the same coordinates and visually hide
        one another.

        Here they are merged for display:
            one 3.18-mm shim  -> 3.18-mm normal dimension
            two 3.18-mm shims -> 6.36-mm normal dimension

        The STL generation is unchanged; this collection is only for
        visual verification.
        """
        groups = self._group_magnets_into_slots(
            source_collection,
            already_split=True,
        )

        display_collection = magpy.Collection(
            style_label=label
        )

        for group in groups:
            representative = group[
                "representative"
            ]

            dimension = np.asarray(
                representative.dimension,
                dtype=float,
            ).copy()

            # Force the visible physical dimension along the tray normal
            # to equal the SUMMED stack thickness.
            dimension[
                self.normal_index
            ] = group[
                "stack_thickness"
            ]

            position = np.asarray(
                representative.position,
                dtype=float,
            ).copy()

            position[
                self.normal_index
            ] = 0.0

            cuboid = magpy.magnet.Cuboid(
                magnetization=np.asarray(
                    representative.magnetization,
                    dtype=float,
                ).copy(),
                dimension=dimension,
                position=position,
                orientation=representative.orientation,
            )

            cuboid.style.color = color

            display_collection.add(
                cuboid
            )

        return display_collection, groups

    def show_split(self):
        """
        Show the separated trays as PHYSICAL STACKS rather than overlapping
        projected source Cuboids.

        Source counts are still preserved in self.positive_side and
        self.negative_side and are checked separately.

        The display collections contain one object per occupied slot, so
        a double stack is visibly twice as thick.
        """
        if (
            self.positive_side is None
            or self.negative_side is None
        ):
            self.split_magnets()

        positive_display, positive_groups = (
            self._make_physical_stack_collection(
                self.positive_side,
                label=(
                    f"+{self.separation_axis.upper()} "
                    f"physical shim stacks"
                ),
                color="#6EB4E6",
            )
        )

        negative_display, negative_groups = (
            self._make_physical_stack_collection(
                self.negative_side,
                label=(
                    f"-{self.separation_axis.upper()} "
                    f"physical shim stacks"
                ),
                color="#F0B45A",
            )
        )

        print(
            "\n============================================================"
        )
        print(
            "SEPARATED-TRAY VISUALIZATION CHECK"
        )
        print(
            "============================================================"
        )

        print(
            f"Raw +{self.separation_axis.upper()} sources: "
            f"{len(self.positive_side)}"
        )

        print(
            f"Displayed +{self.separation_axis.upper()} physical slots: "
            f"{len(positive_groups)}"
        )

        print(
            f"Raw -{self.separation_axis.upper()} sources: "
            f"{len(self.negative_side)}"
        )

        print(
            f"Displayed -{self.separation_axis.upper()} physical slots: "
            f"{len(negative_groups)}"
        )

        if self.tray_shape == "rectangle":
            print(
                "Visualization thickness axis = X = dimension[0]."
            )

            print(
                "A double stack should now appear as approximately "
                "6.36 x 6.35 x 6.35 mm (a cube)."
            )

            print(
                "A single stack should appear as approximately "
                "3.18 x 6.35 x 6.35 mm."
            )

        else:
            print(
                "Visualization thickness axis = Z = dimension[2]."
            )

        print(
            "NOTE: displayed object count is PHYSICAL SLOT count, "
            "not original source count."
        )

        print(
            "============================================================\n"
        )

        positive_display.show()
        negative_display.show()

    # ==================================================================
    # Base tray and variable-depth magnet pockets
    # ==================================================================

    def _make_base_tray(self):
        self._require_inferred_geometry()

        if self.tray_shape == "disc":
            # Disc in XY plane; thickness is along Z.
            return trimesh.creation.cylinder(
                radius=(
                    self.disc_diameter / 2
                ),
                height=self.tray_thickness,
                sections=128,
            )

        # Rectangle in YZ plane; thickness is along X.
        return trimesh.creation.box(
            extents=[
                self.tray_thickness,
                self.rectangle_y,
                self.rectangle_z,
            ]
        )

    def _slot_depth(self, slot_group):
        """Depth of one physical slot = summed shim stack + clearance."""
        return min(
            slot_group["stack_thickness"] + self.thickness_clearance,
            self.tray_thickness,
        )

    def _cuboid_slot_mesh(self, slot_group):
        """
        Build ONE pocket cutter for one grouped physical slot.
        The footprint comes from a representative Cuboid; the normal depth
        comes from the summed stack thickness of every Cuboid in that slot.
        """
        representative = slot_group["representative"]
        mesh = self._oriented_cuboid_at_origin(representative)
        slot_depth = self._slot_depth(slot_group)

        bounds = mesh.bounds
        current_extent = bounds[1, self.normal_index] - bounds[0, self.normal_index]
        cutter_extent = slot_depth + 2 * self.boolean_overlap
        scale = cutter_extent / max(current_extent, 1e-12)

        vertices = mesh.vertices.copy()
        center_normal = 0.5 * (bounds[0, self.normal_index] + bounds[1, self.normal_index])
        vertices[:, self.normal_index] = center_normal + (
            vertices[:, self.normal_index] - center_normal
        ) * scale
        mesh.vertices = vertices

        position = self._plane_to_world(slot_group["center_uv"], normal=0.0)
        slot_center_normal = self.tray_thickness / 2 - slot_depth / 2
        position[self.normal_index] = slot_center_normal
        mesh.apply_translation(position)
        return mesh

    # ==================================================================
    # Engraving primitives in tray-plane coordinates
    # ==================================================================

    def _engraving_normal_position(
        self,
        depth,
        surface_sign=+1,
    ):
        """
        Center position of an engraving cutter on either tray face.

        surface_sign = +1 -> +normal face
        surface_sign = -1 -> -normal face
        """
        self._require_inferred_geometry()

        surface_sign = 1 if surface_sign >= 0 else -1

        return surface_sign * (
            self.tray_thickness / 2
            - depth / 2
        )

    def _make_plane_bar(
        self,
        center_uv,
        angle,
        length,
        width,
        depth,
        surface_sign=+1,
    ):
        """
        Rectangular engraving bar lying in the local tray plane.
        """
        bar = trimesh.creation.box(
            extents=[
                length,
                width,
                depth * 2,
            ]
        )

        c = np.cos(angle)
        s = np.sin(angle)

        rot = np.array(
            [
                [c, -s, 0.0],
                [s,  c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        local = (
            bar.vertices
            @ rot.T
        )

        world = np.zeros_like(
            local
        )

        world[
            :,
            self.u_index
        ] = local[:, 0]

        world[
            :,
            self.v_index
        ] = local[:, 1]

        world[
            :,
            self.normal_index
        ] = local[:, 2]

        world += self._plane_to_world(
            center_uv,
            self._engraving_normal_position(
                depth,
                surface_sign=surface_sign,
            ),
        )

        bar.vertices = world

        return bar

    def _make_bar_local(
        self,
        label_center_uv,
        local_center_uv,
        angle_local,
        length,
        width,
        depth,
        label_rotation=0.0,
        surface_sign=+1,
    ):
        c = np.cos(
            label_rotation
        )

        s = np.sin(
            label_rotation
        )

        rot2 = np.array([
            [c, -s],
            [s,  c],
        ])

        center_uv = (
            np.asarray(
                label_center_uv,
                dtype=float,
            )
            + rot2
            @ np.asarray(
                local_center_uv,
                dtype=float,
            )
        )

        return self._make_plane_bar(
            center_uv,
            (
                label_rotation
                + angle_local
            ),
            length,
            width,
            depth,
            surface_sign=surface_sign,
        )

    def _symbol_meshes(
        self,
        symbol,
        label_center_uv,
        char_center_uv,
        char_size,
        stroke_width,
        depth,
        label_rotation=0.0,
    ):
        meshes = []

        def add(
            angle,
            length,
            offset=(0.0, 0.0),
        ):
            meshes.append(
                self._make_bar_local(
                    label_center_uv,
                    (
                        np.asarray(
                            char_center_uv,
                            dtype=float,
                        )
                        + np.asarray(
                            offset,
                            dtype=float,
                        )
                    ),
                    angle,
                    length,
                    stroke_width,
                    depth,
                    label_rotation,
                )
            )

        if symbol == "+":
            add(
                0.0,
                0.85 * char_size,
            )

            add(
                np.pi / 2,
                0.85 * char_size,
            )

        elif symbol == "-":
            add(
                0.0,
                0.85 * char_size,
            )

        elif symbol == "X":
            add(
                np.pi / 4,
                char_size,
            )

            add(
                -np.pi / 4,
                char_size,
            )

        elif symbol == "Y":
            add(
                -np.pi / 4,
                0.55 * char_size,
                (
                    -0.18 * char_size,
                    0.16 * char_size,
                ),
            )

            add(
                np.pi / 4,
                0.55 * char_size,
                (
                    0.18 * char_size,
                    0.16 * char_size,
                ),
            )

            add(
                np.pi / 2,
                0.65 * char_size,
                (
                    0.0,
                    -0.16 * char_size,
                ),
            )

        elif symbol == "Z":
            add(
                0.0,
                0.85 * char_size,
                (
                    0.0,
                    0.36 * char_size,
                ),
            )

            add(
                0.0,
                0.85 * char_size,
                (
                    0.0,
                    -0.36 * char_size,
                ),
            )

            add(
                -np.pi / 4,
                1.15 * char_size,
            )

        else:
            raise ValueError(
                f"Unsupported engraving symbol "
                f"'{symbol}'."
            )

        return meshes

    def _label_meshes(
        self,
        text,
        center_uv,
        char_size,
        stroke_width,
        depth,
    ):
        meshes = []

        pitch = (
            1.20 * char_size
        )

        offsets = (
            np.arange(
                len(text)
            )
            - (
                len(text) - 1
            ) / 2
        ) * pitch

        for char, offset in zip(
            text,
            offsets,
        ):
            meshes.extend(
                self._symbol_meshes(
                    char,
                    center_uv,
                    np.array([
                        offset,
                        0.0,
                    ]),
                    char_size,
                    stroke_width,
                    depth,
                )
            )

        return meshes

    # ==================================================================
    # Orientation-label placement and collision detection
    # ==================================================================

    @staticmethod
    def _bbox_overlap(
        bbox1_min,
        bbox1_max,
        bbox2_min,
        bbox2_max,
        clearance=0.0,
    ):
        return not (
            bbox1_max[0]
            + clearance
            < bbox2_min[0]
            or bbox1_min[0]
            - clearance
            > bbox2_max[0]
            or bbox1_max[1]
            + clearance
            < bbox2_min[1]
            or bbox1_min[1]
            - clearance
            > bbox2_max[1]
        )

    @staticmethod
    def _label_bbox(
        center_uv,
        text,
        char_size,
    ):
        pitch = (
            1.20 * char_size
        )

        width = (
            (
                len(text) - 1
            )
            * pitch
            + char_size
        )

        height = char_size

        center_uv = np.asarray(
            center_uv,
            dtype=float,
        )

        bbox_min = (
            center_uv
            - np.array([
                width / 2,
                height / 2,
            ])
        )

        bbox_max = (
            center_uv
            + np.array([
                width / 2,
                height / 2,
            ])
        )

        return (
            bbox_min,
            bbox_max,
        )

    def _label_inside_tray(
        self,
        center_uv,
        text,
        char_size,
        boundary_clearance=0.001,
    ):
        bbox_min, bbox_max = (
            self._label_bbox(
                center_uv,
                text,
                char_size,
            )
        )

        corners = np.array([
            [
                bbox_min[0],
                bbox_min[1],
            ],
            [
                bbox_min[0],
                bbox_max[1],
            ],
            [
                bbox_max[0],
                bbox_min[1],
            ],
            [
                bbox_max[0],
                bbox_max[1],
            ],
        ])

        if self.tray_shape == "disc":
            safe_radius = (
                self.disc_diameter / 2
                - boundary_clearance
            )

            return np.all(
                np.linalg.norm(
                    corners,
                    axis=1,
                )
                <= safe_radius
            )

        half_u, half_v = (
            self._tray_half_sizes()
        )

        return (
            np.all(
                np.abs(
                    corners[:, 0]
                )
                <= (
                    half_u
                    - boundary_clearance
                )
            )
            and
            np.all(
                np.abs(
                    corners[:, 1]
                )
                <= (
                    half_v
                    - boundary_clearance
                )
            )
        )

    def _label_hits_magnet(
        self,
        center_uv,
        text,
        char_size,
        collection,
        clearance=0.0015,
    ):
        label_min, label_max = self._label_bbox(center_uv, text, char_size)
        slot_groups = self._group_magnets_into_slots(collection, already_split=True)

        for slot_group in slot_groups:
            slot = self._cuboid_slot_mesh(slot_group)
            bounds = slot.bounds
            magnet_min = np.array([bounds[0, self.u_index], bounds[0, self.v_index]])
            magnet_max = np.array([bounds[1, self.u_index], bounds[1, self.v_index]])
            if self._bbox_overlap(
                label_min, label_max, magnet_min, magnet_max, clearance=clearance
            ):
                return True
        return False

    def _find_orientation_label_position(
        self,
        text,
        direction_uv,
        collection,
        char_size,
    ):
        """Search essentially the full corresponding tray edge for a clear label location."""
        direction_uv = np.asarray(direction_uv, dtype=float)
        direction_uv /= np.linalg.norm(direction_uv)
        tangent = np.array([-direction_uv[1], direction_uv[0]])
        half_u, half_v = self._tray_half_sizes()
        along_u = abs(direction_uv[0]) > 0.5
        boundary_distance = half_u if along_u else half_v
        tangential_half_extent = half_v if along_u else half_u

        initial_inset = 0.003
        inward_step = 0.001
        max_inset = max(initial_inset, boundary_distance - 0.002)
        tangent_step = 0.001
        max_tangent = max(0.0, tangential_half_extent - char_size)

        inward_values = np.arange(initial_inset, max_inset + inward_step/2, inward_step)
        tangent_magnitudes = np.arange(0.0, max_tangent + tangent_step/2, tangent_step)

        for inset in inward_values:
            axis_center = direction_uv * (boundary_distance - inset)
            offsets = [0.0]
            for magnitude in tangent_magnitudes[1:]:
                offsets.extend([magnitude, -magnitude])

            for tangent_offset in offsets:
                candidate = axis_center + tangent * tangent_offset
                if not self._label_inside_tray(candidate, text, char_size):
                    continue
                if not self._label_hits_magnet(candidate, text, char_size, collection):
                    return candidate
        return None

    def _orientation_marks(self, collection):
        depth = self.orientation_engraving_depth

        if self.tray_shape == "disc":
            labels = {
                "X+": np.array([+1.0, 0.0]),
                "X-": np.array([-1.0, 0.0]),
                "Y+": np.array([0.0, +1.0]),
                "Y-": np.array([0.0, -1.0]),
            }
        else:
            labels = {
                "Y+": np.array([+1.0, 0.0]),
                "Y-": np.array([-1.0, 0.0]),
                "Z+": np.array([0.0, +1.0]),
                "Z-": np.array([0.0, -1.0]),
            }

        marks = []
        print("\nScanner orientation marks:")
        candidate_sizes = [0.0045, 0.0040, 0.0035, 0.0030, 0.0025]

        for label_text, direction in labels.items():
            center_uv = None
            chosen_size = None
            for char_size in candidate_sizes:
                center_uv = self._find_orientation_label_position(
                    label_text, direction, collection, char_size
                )
                if center_uv is not None:
                    chosen_size = char_size
                    break

            if center_uv is None:
                # Last-resort nonfatal fallback so STL generation continues.
                chosen_size = 0.0020
                half_u, half_v = self._tray_half_sizes()
                boundary_distance = half_u if abs(direction[0]) > 0.5 else half_v
                center_uv = direction * max(0.0, boundary_distance - 0.004)
                print(f"  WARNING: using fallback position for {label_text}")

            stroke_width = max(0.00045, chosen_size * (0.0008 / 0.0045))
            center_world = self._plane_to_world(center_uv, normal=0.0)
            print(
                f"  {label_text}: world XYZ = ("
                f"{center_world[0]*1000:.1f}, {center_world[1]*1000:.1f}, "
                f"{center_world[2]*1000:.1f}) mm; char={chosen_size*1000:.1f} mm"
            )
            marks.extend(self._label_meshes(
                label_text, center_uv, chosen_size, stroke_width, depth
            ))
        return marks

    # ==================================================================
    # Polarity marks
    # ==================================================================

    def _magnet_global_polarity(self, magnet):
        """
        Return +1 when the magnetization points along +tray-normal,
        otherwise -1.

        disc      : tray normal = Z
        rectangle : tray normal = X
        """
        magnetization_local = np.asarray(
            magnet.magnetization,
            dtype=float,
        )

        if magnet.orientation is not None:
            magnetization_global = magnet.orientation.apply(
                magnetization_local
            )
        else:
            magnetization_global = magnetization_local.copy()

        component = float(
            magnetization_global[self.normal_index]
        )

        return +1 if component > 0 else -1

    def _original_group_for_slot(
        self,
        slot_group,
        side_sign,
    ):
        """
        Match a projected tray slot back to the corresponding ORIGINAL
        PKL slot group so the original normal-axis source positions are
        available.

        Those original positions tell us which source magnet is closest
        to the +normal and -normal faces of the physical tray.
        """
        side_sign = +1 if side_sign > 0 else -1
        center_uv = np.asarray(
            slot_group["center_uv"],
            dtype=float,
        )

        matches = []

        for original_group in self.slot_groups_all:
            original_side = (
                +1
                if original_group["side"] > 0
                else -1
            )

            if original_side != side_sign:
                continue

            distance = np.linalg.norm(
                center_uv
                - np.asarray(
                    original_group["center_uv"],
                    dtype=float,
                )
            )

            if distance <= self.slot_group_tolerance:
                matches.append(
                    (
                        distance,
                        original_group,
                    )
                )

        if not matches:
            raise RuntimeError(
                "Could not match projected tray slot back to "
                "its original PKL slot."
            )

        matches.sort(
            key=lambda item: item[0]
        )

        return matches[0][1]

    def _normal_layer_positions_for_side(
        self,
        side_sign,
    ):
        """
        Return the distinct ORIGINAL source positions along the tray normal
        for one side of the scanner.

        These are used only to decide which physical tray face a single
        shim is closest to.
        """
        side_sign = +1 if side_sign > 0 else -1

        values = []

        for magnet in self.magnets:
            normal_position = float(
                np.asarray(
                    magnet.position,
                    dtype=float,
                )[self.normal_index]
            )

            magnet_side = (
                +1
                if normal_position > 0
                else -1
            )

            if magnet_side == side_sign:
                values.append(
                    normal_position
                )

        if not values:
            return []

        values = np.asarray(
            values,
            dtype=float,
        )

        # Cluster near-identical layer positions using the same tolerance
        # scale already used for slot grouping.
        values = np.sort(values)
        clusters = []

        for value in values:
            if (
                not clusters
                or abs(
                    value
                    - np.mean(
                        clusters[-1]
                    )
                )
                > self.slot_group_tolerance
            ):
                clusters.append(
                    [value]
                )
            else:
                clusters[-1].append(
                    value
                )

        return [
            float(
                np.mean(cluster)
            )
            for cluster in clusters
        ]

    def _surface_adjacent_magnets(
        self,
        original_group,
        side_sign,
    ):
        """
        Determine which source magnet is closest to each physical tray face.

        Returns
        -------
        dict
            {
                +1: magnet nearest +normal face or None,
                -1: magnet nearest -normal face or None,
            }

        For two shims this is unambiguous:
            largest original normal coordinate  -> +normal face
            smallest original normal coordinate -> -normal face

        For one shim, its original normal-layer position is compared with
        the available source layers on that scanner side to choose the
        nearest physical tray face.
        """
        magnets = list(
            original_group["magnets"]
        )

        result = {
            +1: None,
            -1: None,
        }

        if not magnets:
            return result

        positions = np.asarray(
            [
                float(
                    np.asarray(
                        magnet.position,
                        dtype=float,
                    )[self.normal_index]
                )
                for magnet in magnets
            ],
            dtype=float,
        )

        order = np.argsort(
            positions
        )

        if len(magnets) >= 2:
            # Only the two OUTERMOST members of a multi-shim stack can
            # physically touch/face the two tray surfaces.
            result[-1] = magnets[
                int(order[0])
            ]

            result[+1] = magnets[
                int(order[-1])
            ]

            return result

        # Single occupied shim: infer whether it belongs to the inner or
        # outer normal layer using the layer positions present in the
        # ORIGINAL PKL for this tray side.
        magnet = magnets[0]
        p = positions[0]

        layers = (
            self._normal_layer_positions_for_side(
                side_sign
            )
        )

        if len(layers) >= 2:
            nearest_min = abs(
                p - min(layers)
            )

            nearest_max = abs(
                p - max(layers)
            )

            surface_sign = (
                -1
                if nearest_min < nearest_max
                else +1
            )
        else:
            # If only one layer exists on this scanner side, the best
            # physical convention is the outward/global side.
            surface_sign = (
                +1
                if side_sign > 0
                else -1
            )

        result[
            surface_sign
        ] = magnet

        return result

    def _plus_mark_for_surface(
        self,
        representative,
        center_uv,
        surface_sign,
    ):
        """
        Create the engraved '+' next to one slot on a specified tray face.
        """
        center_uv = np.asarray(
            center_uv,
            dtype=float,
        )

        radial = center_uv.copy()
        radial_norm = np.linalg.norm(
            radial
        )

        if radial_norm > 0:
            radial /= radial_norm
        else:
            radial = np.array([
                1.0,
                0.0,
            ])

        tangent = np.array([
            -radial[1],
            radial[0],
        ])

        # Use the oriented representative cuboid to estimate in-plane
        # footprint for marker spacing.
        slot = self._oriented_cuboid_at_origin(
            representative
        )

        bounds = slot.bounds

        half_u = (
            bounds[1, self.u_index]
            - bounds[0, self.u_index]
        ) / 2

        half_v = (
            bounds[1, self.v_index]
            - bounds[0, self.v_index]
        ) / 2

        conservative_half_width = max(
            half_u,
            half_v,
        )

        symbol_gap = 0.0020
        symbol_length = 0.0030
        symbol_width = 0.0007

        symbol_center_uv = (
            center_uv
            + radial
            * (
                conservative_half_width
                + symbol_gap
            )
        )

        marks = []

        marks.append(
            self._make_plane_bar(
                symbol_center_uv,
                np.arctan2(
                    tangent[1],
                    tangent[0],
                ),
                symbol_length,
                symbol_width,
                self.polarity_engraving_depth,
                surface_sign=surface_sign,
            )
        )

        marks.append(
            self._make_plane_bar(
                symbol_center_uv,
                np.arctan2(
                    radial[1],
                    radial[0],
                ),
                symbol_length,
                symbol_width,
                self.polarity_engraving_depth,
                surface_sign=surface_sign,
            )
        )

        return marks

    def _slot_surface_polarity_marks(
        self,
        slot_group,
        side_sign,
    ):
        """
        Engrave polarity independently on the TWO tray surfaces.

        Convention remains:
            '+' = positive magnetization along +tray-normal
            blank = negative magnetization

        For a double stack, the source nearest each tray surface controls
        that surface's mark. This correctly handles mixed-polarity pairs:

            + / +  -> '+' on both tray faces
            + / -  -> '+' only on the face nearest the positive shim
            - / +  -> '+' only on the opposite face
            - / -  -> no '+' on either face

        The source-to-surface assignment comes from the ORIGINAL PKL
        normal-axis positions, before the sources are projected onto the
        local STL tray plane.
        """
        original_group = (
            self._original_group_for_slot(
                slot_group,
                side_sign,
            )
        )

        adjacent = (
            self._surface_adjacent_magnets(
                original_group,
                side_sign,
            )
        )

        marks = []

        print(
            f"    surface polarity: "
            f"-{self.separation_axis.upper()} face="
            f"{'+' if adjacent[-1] is not None and self._magnet_global_polarity(adjacent[-1]) > 0 else '-/blank'}, "
            f"+{self.separation_axis.upper()} face="
            f"{'+' if adjacent[+1] is not None and self._magnet_global_polarity(adjacent[+1]) > 0 else '-/blank'}"
        )

        for surface_sign in (-1, +1):
            magnet = adjacent[
                surface_sign
            ]

            if magnet is None:
                continue

            if (
                self._magnet_global_polarity(
                    magnet
                )
                <= 0
            ):
                continue

            marks.extend(
                self._plus_mark_for_surface(
                    original_group[
                        "representative"
                    ],
                    slot_group[
                        "center_uv"
                    ],
                    surface_sign,
                )
            )

        return marks

    # ==================================================================
    # Alignment notch
    # ==================================================================

    def _alignment_notch(self):
        """
        Small semicircular through-notch at the centered negative end
        of the second in-plane axis.

        disc:
            X=0, Y=-Ymax

        rectangle:
            Y=0, Z=-Zmax
        """
        half_u, half_v = (
            self._tray_half_sizes()
        )

        center_uv = np.array([
            0.0,
            -half_v,
        ])

        notch_height = (
            self.tray_thickness
            + 0.004
        )

        notch = (
            trimesh.creation.cylinder(
                radius=self.notch_radius,
                height=notch_height,
                sections=64,
            )
        )

        if self.normal_index == 0:
            # Initial cylinder axis is Z.
            # Rotate Z -> X for a YZ tray.
            rotation = (
                trimesh.transformations.rotation_matrix(
                    np.pi / 2,
                    [
                        0.0,
                        1.0,
                        0.0,
                    ],
                )
            )

            notch.apply_transform(
                rotation
            )

        elif self.normal_index == 1:
            # Included for completeness.
            rotation = (
                trimesh.transformations.rotation_matrix(
                    -np.pi / 2,
                    [
                        1.0,
                        0.0,
                        0.0,
                    ],
                )
            )

            notch.apply_transform(
                rotation
            )

        center_world = (
            self._plane_to_world(
                center_uv,
                normal=0.0,
            )
        )

        notch.apply_translation(
            center_world
        )

        return notch

    # ==================================================================
    # Boolean operations and STL writing
    # ==================================================================

    def _boolean_difference(
        self,
        tray,
        cutter,
    ):
        result = (
            trimesh.boolean.difference(
                [
                    tray,
                    cutter,
                ],
                engine=self.boolean_engine,
            )
        )

        if result is None:
            raise RuntimeError(
                "Boolean subtraction returned None."
            )

        return result

    def _write_tray(
        self,
        collection,
        filename,
        side_sign,
    ):
        tray = self._make_base_tray()
        cutters = [self._alignment_notch()]
        cutters.extend(self._orientation_marks(collection))

        slot_groups = self._group_magnets_into_slots(
            collection,
            already_split=True,
        )
        self._print_slot_summary(
            slot_groups,
            f"Physical slot stacks for {filename}",
        )

        print(f"\nDetailed pocket-depth verification for {filename}:")
        print(f"  Tray thickness = {self.tray_thickness*1000:.3f} mm")
        print(f"  Pocket clearance = {self.thickness_clearance*1000:.3f} mm")

        for index, slot_group in enumerate(slot_groups):
            stack_thickness = slot_group["stack_thickness"]
            slot_depth = self._slot_depth(slot_group)
            floor = max(0.0, self.tray_thickness - slot_depth)
            center_uv = slot_group["center_uv"]
            individual = ", ".join(
                f"{value*1000:.3f}" for value in slot_group["shim_thicknesses"]
            )
            print(
                f"  SLOT {index:03d}: {self.u_axis.upper()}={center_uv[0]*1000:.2f} mm, "
                f"{self.v_axis.upper()}={center_uv[1]*1000:.2f} mm | "
                f"stack={slot_group['stack_count']} shim(s) [{individual}] mm | "
                f"SUMMED {self.separation_axis.upper()} shim thickness="
                f"{stack_thickness*1000:.3f} mm | "
                f"ACTUAL POCKET DEPTH={slot_depth*1000:.3f} mm | "
                f"remaining floor={floor*1000:.3f} mm"
            )

            # Hard numerical verification that the Boolean cutter depth
            # matches the grouped shim stack depth.
            expected_depth = min(
                stack_thickness + self.thickness_clearance,
                self.tray_thickness,
            )

            if not np.isclose(
                slot_depth,
                expected_depth,
                rtol=0.0,
                atol=1e-9,
            ):
                raise RuntimeError(
                    f"Pocket-depth verification failed at slot {index}: "
                    f"computed={slot_depth*1000:.6f} mm, "
                    f"expected={expected_depth*1000:.6f} mm"
                )
            cutters.append(
                self._cuboid_slot_mesh(
                    slot_group
                )
            )

            cutters.extend(
                self._slot_surface_polarity_marks(
                    slot_group,
                    side_sign,
                )
            )

        print(f"\nWriting {filename}")
        for cutter in cutters:
            tray = self._boolean_difference(tray, cutter)

        tray.remove_unreferenced_vertices()
        tray.export(filename)
        print(f"  Wrote {filename}")
        print(f"  Watertight: {tray.is_watertight}")
        print(f"  Final tray thickness: {self.tray_thickness*1000:.3f} mm")

        # Final concise verification of slot depths in this specific tray.
        depth_histogram = {}
        for slot_group in slot_groups:
            depth_mm = round(
                self._slot_depth(slot_group) * 1000.0,
                4,
            )
            depth_histogram[depth_mm] = (
                depth_histogram.get(depth_mm, 0) + 1
            )

        print("  Pocket-depth histogram:")
        for depth_mm, count in sorted(depth_histogram.items()):
            print(
                f"    {depth_mm:.4f} mm deep : {count} slot(s)"
            )

        return tray

    def write(
        self,
        positive_filename=None,
        negative_filename=None,
        show=True,
    ):
        """
        Generate both tray STL files.
        """
        if self.magnets is None:
            self.load(
                show_original=False
            )

        if (
            self.positive_side is None
            or self.negative_side is None
        ):
            self.split_magnets()

        axis = (
            self.separation_axis
        )

        if positive_filename is None:
            positive_filename = (
                f"shim_tray_"
                f"{axis}_positive.stl"
            )

        if negative_filename is None:
            negative_filename = (
                f"shim_tray_"
                f"{axis}_negative.stl"
            )

        positive_mesh = (
            self._write_tray(
                self.positive_side,
                positive_filename,
                side_sign=+1,
            )
        )

        negative_mesh = (
            self._write_tray(
                self.negative_side,
                negative_filename,
                side_sign=-1,
            )
        )

        positive_mesh.visual.face_colors = (
            self.positive_color
        )

        negative_mesh.visual.face_colors = (
            self.negative_color
        )

        if show:
            trimesh.Scene(
                {
                    "positive_tray":
                        positive_mesh
                }
            ).show()

            trimesh.Scene(
                {
                    "negative_tray":
                        negative_mesh
                }
            ).show()

        return (
            positive_mesh,
            negative_mesh,
        )


# ======================================================================
# Interactive command-line entry point
# ======================================================================

if __name__ == "__main__":

    tray_shape = input(
        "Enter tray shape ('disc' or 'rectangle') [rectangle]: "
    ).strip().lower()

    if tray_shape == "":
        tray_shape = "rectangle"

    if tray_shape in {
        "rect",
        "rectangular",
        "slab",
    }:
        tray_shape = "rectangle"

    if tray_shape in {
        "disk",
        "circle",
        "circular",
    }:
        tray_shape = "disc"

    if tray_shape == "disc":

        pkl_filename = input(
            "PKL filename "
            "[./data/magnet_collection_shims_20260825.pkl]: "
        ).strip()

        if pkl_filename == "":
            pkl_filename = (
                "./data/"
                "magnet_collection_shims_20260825.pkl"
            )

        diameter_str = input(
            "Disc diameter in mm [152.4]: "
        ).strip()

        diameter_mm = (
            float(diameter_str)
            if diameter_str
            else 152.4
        )

        converter = pkl2stl(
            pkl_filename=pkl_filename,
            tray_shape="disc",
            separation_axis="z",
            disc_diameter_mm=diameter_mm,
        )

    else:

        pkl_filename = input(
            "PKL filename "
            "[./data/rectangle_magnet_collection_shims.pkl]: "
        ).strip()

        if pkl_filename == "":
            pkl_filename = (
                "./data/"
                "rectangle_magnet_collection_shims.pkl"
            )

        print(
            "\nRectangle dimensions can be inferred automatically "
            "from the outermost magnet bounds."
        )

        print(
            "Press Enter for automatic sizing. "
            "Enter a value only if you want to force a specific size."
        )

        y_str = input(
            "Rectangle Y size in mm [auto]: "
        ).strip()

        z_str = input(
            "Rectangle Z size in mm [auto]: "
        ).strip()

        y_mm = (
            float(y_str)
            if y_str
            else None
        )

        z_mm = (
            float(z_str)
            if z_str
            else None
        )

        converter = pkl2stl(
            pkl_filename=pkl_filename,
            tray_shape="rectangle",
            separation_axis="x",
            rectangle_y_mm=y_mm,
            rectangle_z_mm=z_mm,
            rectangle_margin_mm=5.0,
        )

    converter.load(
        show_original=True
    )

    converter.split_magnets()

    converter.show_split()

    converter.write(
        show=True
    )
