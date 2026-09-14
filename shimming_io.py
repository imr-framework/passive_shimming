"""
Input/output and experiment bookkeeping for passive shimming.

Classes
-------
ShimmingRun
    Creates and manages one timestamped run directory, reproducibility
    settings, incremental workflow stopping, and JSON output.

FieldMap
    Loads a measured field map, converts it to SI units, validates it,
    builds Magpylib sensors, and provides basic visualization/statistics.

All internal geometry uses SI units:
    position -> meters
    field    -> tesla
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import magpylib as magpy
import numpy as np

from utils import get_field_pos, display_scatter_3D


def _json_ready(value: Any) -> Any:
    """Convert lightweight project objects into JSON-serializable values."""
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
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "to_dict"):
        return _json_ready(value.to_dict())
    if hasattr(value, "statistics") and callable(value.statistics):
        return _json_ready(value.statistics())
    if hasattr(value, "__dict__"):
        data = {}
        for key, item in vars(value).items():
            if key.startswith("_"):
                continue
            if isinstance(item, (magpy.Collection, magpy.Sensor)):
                continue
            data[key] = _json_ready(item)
        return data
    return value


class ShimmingRun:
    """
    Manage one passive-shimming experiment run.

    Parameters
    ----------
    output_root : str or pathlib.Path
        Parent directory for all passive-shimming runs.
    random_seed : int, optional
        Reproducibility seed. Default 42.
    stop_after_step : int or None, optional
        Workflow checkpoint. None means run all steps.
    run_id : str or None, optional
        Explicit run identifier. If omitted, a timestamp is generated.

    Attributes
    ----------
    run_id : str
        Unique run identifier.
    run_dir : pathlib.Path
        Directory containing this run's outputs.
    random_seed : int
        Reproducibility seed.
    stop_after_step : int or None
        Requested workflow stopping point.
    """

    def __init__(
        self,
        output_root,
        random_seed=42,
        stop_after_step=None,
        run_id=None,
    ):
        self.output_root = Path(output_root)
        self.random_seed = int(random_seed)
        self.stop_after_step = (
            None if stop_after_step is None else int(stop_after_step)
        )
        self.run_id = (
            str(run_id)
            if run_id is not None
            else datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.run_dir = self.output_root / self.run_id

        self.run_dir.mkdir(parents=True, exist_ok=False)
        np.random.seed(self.random_seed)

        print("\nPassive shimming run initialized")
        print(f"  Run ID: {self.run_id}")
        print(f"  Output directory: {self.run_dir}")
        print(f"  Random seed: {self.random_seed}")

        if self.stop_after_step is not None:
            print(f"  Stop after step: {self.stop_after_step}")

        self.save_report("run", self.to_dict())

    def path(self, filename) -> Path:
        """Return a path inside this run directory."""
        output_path = self.run_dir / Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def stop_after(self, step) -> bool:
        """Return True when execution should stop after ``step``."""
        if self.stop_after_step is None:
            return False
        return int(step) >= self.stop_after_step

    def save_json(self, data, filename) -> Path:
        """Save JSON data inside the run directory."""
        output_path = self.path(filename)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                _json_ready(data),
                f,
                indent=2,
                sort_keys=False,
            )
        return output_path

    def save_config(self, name, config) -> Path:
        """Save configuration as ``<name>_config.json``."""
        return self.save_json(config, f"{name}_config.json")

    def save_report(self, name, report) -> Path:
        """Save report as ``<name>_report.json``."""
        return self.save_json(report, f"{name}_report.json")

    def to_dict(self) -> dict:
        """Return lightweight run metadata."""
        return {
            "run_id": self.run_id,
            "output_root": self.output_root,
            "run_dir": self.run_dir,
            "random_seed": self.random_seed,
            "stop_after_step": self.stop_after_step,
        }


class FieldMap:
    """
    Measured static magnetic-field map in SI units.

    Parameters
    ----------
    positions_m : array-like, shape (N, 3)
        Measurement coordinates in meters.
    B_T : array-like, shape (N,)
        Measured static field values in tesla.
    gammabar_Hz_per_T : float
        Proton gamma-bar in Hz/T.
    V : array-like or None, optional
        Optional sensor-voltage channel.
    source_filename : str or pathlib.Path or None, optional
        Original field-map filename.
    grid_spacing_input : dict or None, optional
        Original spacing information returned by ``get_field_pos``.
    position_scale_to_m : float, optional
        Conversion used when loading positions.
    field_scale_to_T : float, optional
        Conversion used when loading field values.
    """

    def __init__(
        self,
        positions_m,
        B_T,
        gammabar_Hz_per_T,
        V=None,
        source_filename=None,
        grid_spacing_input=None,
        position_scale_to_m=1.0,
        field_scale_to_T=1.0,
    ):
        self.positions_m = np.asarray(positions_m, dtype=float)
        self.B_T = np.asarray(B_T, dtype=float).reshape(-1)
        self.gammabar_Hz_per_T = float(gammabar_Hz_per_T)
        self.V = None if V is None else np.asarray(V)
        self.source_filename = (
            None if source_filename is None else Path(source_filename)
        )
        self.grid_spacing_input = (
            {} if grid_spacing_input is None else grid_spacing_input
        )
        self.position_scale_to_m = float(position_scale_to_m)
        self.field_scale_to_T = float(field_scale_to_T)

        self.sensors = None
        self._build_sensors()

    @classmethod
    def from_npy(
        cls,
        filename,
        position_scale_to_m,
        field_scale_to_T,
        gammabar_Hz_per_T,
    ):
        """
        Load the project's field-map ``.npy`` format using ``get_field_pos``.

        Returns
        -------
        FieldMap
            Field map converted to SI units.
        """
        filename = Path(filename)

        if not filename.exists():
            raise FileNotFoundError(
                f"Field-map file not found: {filename}"
            )

        data = np.load(filename)

        x, y, z, B, V, dx, dy, dz = get_field_pos(data)

        x_m = np.asarray(x, dtype=float) * position_scale_to_m
        y_m = np.asarray(y, dtype=float) * position_scale_to_m
        z_m = np.asarray(z, dtype=float) * position_scale_to_m
        B_T = np.asarray(B, dtype=float) * field_scale_to_T

        positions_m = np.column_stack((x_m, y_m, z_m))

        finite_mask = (
            np.all(np.isfinite(positions_m), axis=1)
            & np.isfinite(B_T)
        )

        n_invalid = int(np.sum(~finite_mask))

        if n_invalid > 0:
            print(
                f"Removing {n_invalid} invalid field-map samples."
            )
            positions_m = positions_m[finite_mask]
            B_T = B_T[finite_mask]

            if V is not None:
                V = np.asarray(V)[finite_mask]

        return cls(
            positions_m=positions_m,
            B_T=B_T,
            V=V,
            gammabar_Hz_per_T=gammabar_Hz_per_T,
            source_filename=filename,
            grid_spacing_input={
                "dx": dx,
                "dy": dy,
                "dz": dz,
            },
            position_scale_to_m=position_scale_to_m,
            field_scale_to_T=field_scale_to_T,
        )

    def _build_sensors(self):
        """Build one Magpylib Sensor containing all measurement positions."""
        self.sensors = magpy.Collection(style_label="sensors")
        self.sensors.add(
            magpy.Sensor(
                position=self.positions_m,
                style_size=2,
            )
        )

    @property
    def x_m(self):
        """Measurement X coordinates in meters."""
        return self.positions_m[:, 0]

    @property
    def y_m(self):
        """Measurement Y coordinates in meters."""
        return self.positions_m[:, 1]

    @property
    def z_m(self):
        """Measurement Z coordinates in meters."""
        return self.positions_m[:, 2]

    @property
    def n_points(self):
        """Number of measured field samples."""
        return int(len(self.B_T))

    def validate(self):
        """
        Validate field-map dimensions, finite values, and sensor setup.

        Returns
        -------
        bool
            True when validation succeeds.
        """
        if (
            self.positions_m.ndim != 2
            or self.positions_m.shape[1] != 3
        ):
            raise ValueError(
                "positions_m must have shape (N, 3)."
            )

        if len(self.positions_m) != len(self.B_T):
            raise ValueError(
                "positions_m and B_T must contain "
                "the same number of samples."
            )

        if self.n_points == 0:
            raise ValueError(
                "Field map contains no samples."
            )

        if not np.all(np.isfinite(self.positions_m)):
            raise ValueError(
                "Field-map positions contain invalid values."
            )

        if not np.all(np.isfinite(self.B_T)):
            raise ValueError(
                "Field values contain invalid values."
            )

        if self.gammabar_Hz_per_T <= 0:
            raise ValueError(
                "gammabar_Hz_per_T must be positive."
            )

        if abs(np.mean(self.B_T)) <= np.finfo(float).eps:
            raise ValueError(
                "Mean measured field is effectively zero."
            )

        if self.sensors is None:
            raise ValueError(
                "Magpylib sensors were not created."
            )

        print("\nField-map validation passed")
        print(f"  Samples: {self.n_points}")
        print(
            f"  Mean field: "
            f"{np.mean(self.B_T)*1e3:.6f} mT"
        )
        print(
            f"  Peak-to-peak: "
            f"{np.ptp(self.B_T)*1e3:.6f} mT"
        )
        print(
            f"  Peak-to-peak off-resonance: "
            f"{self.peak_to_peak_kHz():.3f} kHz"
        )

        return True

    def peak_to_peak_kHz(self) -> float:
        """Return peak-to-peak proton off-resonance in kHz."""
        return float(
            np.ptp(self.B_T)
            * self.gammabar_Hz_per_T
            * 1e-3
        )

    def show(
        self,
        title="Measured B field",
        center=False,
        half_range_T=0.5e-3,
    ):
        """
        Display the measured field with the existing scatter utility.
        """
        kwargs = {
            "center": center,
            "title": title,
        }

        if half_range_T is not None:
            mean_B = float(np.mean(self.B_T))
            kwargs["vmin"] = mean_B - half_range_T
            kwargs["vmax"] = mean_B + half_range_T

        display_scatter_3D(
            self.x_m,
            self.y_m,
            self.z_m,
            self.B_T,
            **kwargs,
        )

    def statistics(self) -> dict:
        """Return lightweight measured-field statistics."""
        mean_T = float(np.mean(self.B_T))
        std_T = float(np.std(self.B_T))
        peak_to_peak_T = float(np.ptp(self.B_T))

        return {
            "n_points": self.n_points,
            "mean_field_T": mean_T,
            "mean_field_mT": mean_T * 1e3,
            "std_field_T": std_T,
            "std_field_uT": std_T * 1e6,
            "peak_to_peak_T": peak_to_peak_T,
            "peak_to_peak_mT": peak_to_peak_T * 1e3,
            "peak_to_peak_kHz": self.peak_to_peak_kHz(),
            "position_bounds_m": np.array(
                [
                    np.min(self.positions_m, axis=0),
                    np.max(self.positions_m, axis=0),
                ]
            ),
        }

    def to_dict(self) -> dict:
        """
        Return metadata/configuration without embedding the full field arrays.
        """
        report = {
            "source_filename": self.source_filename,
            "position_scale_to_m": self.position_scale_to_m,
            "field_scale_to_T": self.field_scale_to_T,
            "gammabar_Hz_per_T": self.gammabar_Hz_per_T,
            "grid_spacing_input": self.grid_spacing_input,
        }
        report.update(self.statistics())
        return report
