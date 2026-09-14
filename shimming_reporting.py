"""
Reporting utilities for passive shimming.

This module provides a single reusable class:

``ShimmingReporter``

It evaluates magnetic-field homogeneity before and after shimming using the
same objective components as ``ShimOptimizationProblem`` and adds additional
diagnostic quantities useful for MRI:

    - mean field
    - standard deviation
    - standard deviation in ppm
    - robust percentile range
    - robust percentile range in ppm
    - minimum / maximum field
    - peak-to-peak field
    - peak-to-peak ppm
    - peak-to-peak off-resonance in Hz and kHz
    - objective cost terms
    - optional shim-state counts
    - pre/post improvement percentages

All magnetic-field values are assumed to be in tesla.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _json_ready(value):
    """Convert NumPy values and arrays to JSON-safe Python objects."""
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


class ShimmingReporter:
    """
    Evaluate and report passive-shimming field metrics.

    Parameters
    ----------
    field_map : FieldMap-like object
        Must provide:
            ``B_T``
            ``gammabar_Hz_per_T``

    optimization_problem : ShimOptimizationProblem-like object
        Must provide:
            ``cost_components(B_total_T)``
            ``candidate_ids``

    debug : bool, optional
        Enable additional diagnostic output. Default False.

    Notes
    -----
    The optimization cost is not reimplemented independently here. The
    reporter calls ``optimization_problem.cost_components(...)`` directly so
    reported objective terms are guaranteed to match those used by Pymoo.
    """

    def __init__(
        self,
        field_map,
        optimization_problem,
        debug=False,
    ):
        self.field_map = (
            field_map
        )

        self.optimization_problem = (
            optimization_problem
        )

        self.debug = bool(
            debug
        )

        self._validate_inputs()

    def _validate_inputs(self):
        """Validate required interfaces."""
        if not hasattr(
            self.field_map,
            "B_T",
        ):
            raise TypeError(
                "field_map must provide B_T."
            )

        if not hasattr(
            self.field_map,
            "gammabar_Hz_per_T",
        ):
            raise TypeError(
                "field_map must provide gammabar_Hz_per_T."
            )

        if not hasattr(
            self.optimization_problem,
            "cost_components",
        ):
            raise TypeError(
                "optimization_problem must provide cost_components()."
            )

    @property
    def gammabar_Hz_per_T(self) -> float:
        """Proton gamma-bar in Hz/T."""
        return float(
            self.field_map.gammabar_Hz_per_T
        )

    @staticmethod
    def _safe_improvement_percent(
        before,
        after,
    ):
        """
        Return percentage reduction from before to after.

        Positive values indicate improvement.
        Negative values indicate worsening.
        """
        before = float(
            before
        )

        after = float(
            after
        )

        if not (
            np.isfinite(
                before
            )
            and np.isfinite(
                after
            )
        ):
            return None

        if abs(
            before
        ) <= np.finfo(
            float
        ).eps:
            return None

        return float(
            100.0
            * (
                before
                - after
            )
            / before
        )

    def evaluate(
        self,
        B_total_T,
        label,
        states=None,
    ):
        """
        Evaluate one field distribution.

        Parameters
        ----------
        B_total_T : array-like, shape (N,)
            Total magnetic field in tesla.

        label : str
            Report label, e.g. ``'pre_shim'`` or ``'post_shim'``.

        states : array-like or None, optional
            Optional candidate state vector. If supplied, counts of positive,
            negative, absent, and active shim magnets are added.

        Returns
        -------
        dict
            Complete field-homogeneity report.
        """
        B_total_T = np.asarray(
            B_total_T,
            dtype=float,
        ).reshape(
            -1
        )

        if len(
            B_total_T
        ) == 0:
            raise ValueError(
                "B_total_T contains no field samples."
            )

        if not np.all(
            np.isfinite(
                B_total_T
            )
        ):
            raise ValueError(
                "B_total_T contains non-finite values."
            )

        if len(
            B_total_T
        ) != len(
            self.field_map.B_T
        ):
            raise ValueError(
                "B_total_T length does not match the field-map sample count."
            )

        mean_T = float(
            np.mean(
                B_total_T
            )
        )

        abs_mean_T = abs(
            mean_T
        )

        std_T = float(
            np.std(
                B_total_T
            )
        )

        min_T = float(
            np.min(
                B_total_T
            )
        )

        max_T = float(
            np.max(
                B_total_T
            )
        )

        peak_to_peak_T = (
            max_T
            - min_T
        )

        if abs_mean_T > np.finfo(
            float
        ).eps:
            std_ppm = (
                1e6
                * std_T
                / abs_mean_T
            )

            peak_to_peak_ppm = (
                1e6
                * peak_to_peak_T
                / abs_mean_T
            )

        else:
            std_ppm = np.inf
            peak_to_peak_ppm = np.inf

        peak_to_peak_Hz = (
            peak_to_peak_T
            * self.gammabar_Hz_per_T
        )

        objective = (
            self.optimization_problem
            .cost_components(
                B_total_T
            )
        )

        report = {
            "label":
                str(
                    label
                ),
            "n_points":
                int(
                    len(
                        B_total_T
                    )
                ),
            "field": {
                "mean_T":
                    mean_T,
                "mean_mT":
                    mean_T
                    * 1e3,
                "std_T":
                    std_T,
                "std_uT":
                    std_T
                    * 1e6,
                "std_ppm":
                    float(
                        std_ppm
                    ),
                "minimum_T":
                    min_T,
                "minimum_mT":
                    min_T
                    * 1e3,
                "maximum_T":
                    max_T,
                "maximum_mT":
                    max_T
                    * 1e3,
                "peak_to_peak_T":
                    peak_to_peak_T,
                "peak_to_peak_mT":
                    peak_to_peak_T
                    * 1e3,
                "peak_to_peak_ppm":
                    float(
                        peak_to_peak_ppm
                    ),
                "peak_to_peak_Hz":
                    float(
                        peak_to_peak_Hz
                    ),
                "peak_to_peak_kHz":
                    float(
                        peak_to_peak_Hz
                        * 1e-3
                    ),
            },
            "objective":
                objective,
        }

        if states is not None:
            states = np.asarray(
                states,
                dtype=int,
            ).reshape(
                -1
            )

            expected_n = len(
                self.optimization_problem
                .candidate_ids
            )

            if len(
                states
            ) != expected_n:
                raise ValueError(
                    f"states has length {len(states)}; expected "
                    f"{expected_n}."
                )

            report[
                "shim_state"
            ] = {
                "n_candidates":
                    int(
                        len(
                            states
                        )
                    ),
                "n_active":
                    int(
                        np.count_nonzero(
                            states
                        )
                    ),
                "n_positive":
                    int(
                        np.sum(
                            states == 1
                        )
                    ),
                "n_negative":
                    int(
                        np.sum(
                            states == -1
                        )
                    ),
                "n_absent":
                    int(
                        np.sum(
                            states == 0
                        )
                    ),
            }

        if self.debug:
            print(
                f"\nShimmingReporter evaluated: {label}"
            )
            print(
                f"  Mean field: {mean_T*1e3:.6f} mT"
            )
            print(
                f"  Std: {std_ppm:.3f} ppm"
            )
            print(
                f"  Peak-to-peak: {peak_to_peak_T*1e3:.6f} mT"
            )
            print(
                f"  Peak-to-peak off-resonance: "
                f"{peak_to_peak_Hz*1e-3:.3f} kHz"
            )
            print(
                f"  Cost: {objective['total']:.3f}"
            )

        return report

    def add_improvement(
        self,
        pre_report,
        post_report,
    ):
        """
        Add pre-to-post improvement metrics to a post-shimming report.

        Parameters
        ----------
        pre_report : dict
            Report returned by :meth:`evaluate` before shimming.

        post_report : dict
            Report returned by :meth:`evaluate` after shimming.

        Returns
        -------
        dict
            Copy of ``post_report`` with an ``improvement`` block.
        """
        pre_field = (
            pre_report[
                "field"
            ]
        )

        post_field = (
            post_report[
                "field"
            ]
        )

        pre_objective = (
            pre_report[
                "objective"
            ]
        )

        post_objective = (
            post_report[
                "objective"
            ]
        )

        improvement = {
            "std_ppm_percent":
                self._safe_improvement_percent(
                    pre_field[
                        "std_ppm"
                    ],
                    post_field[
                        "std_ppm"
                    ],
                ),
            "percentile_range_ppm_percent":
                self._safe_improvement_percent(
                    pre_objective[
                        "percentile_range_ppm"
                    ],
                    post_objective[
                        "percentile_range_ppm"
                    ],
                ),
            "peak_to_peak_ppm_percent":
                self._safe_improvement_percent(
                    pre_field[
                        "peak_to_peak_ppm"
                    ],
                    post_field[
                        "peak_to_peak_ppm"
                    ],
                ),
            "peak_to_peak_kHz_percent":
                self._safe_improvement_percent(
                    pre_field[
                        "peak_to_peak_kHz"
                    ],
                    post_field[
                        "peak_to_peak_kHz"
                    ],
                ),
            "objective_cost_percent":
                self._safe_improvement_percent(
                    pre_objective[
                        "total"
                    ],
                    post_objective[
                        "total"
                    ],
                ),
        }

        output = dict(
            post_report
        )

        output[
            "improvement"
        ] = improvement

        return output

    @staticmethod
    def _format_percent(
        value,
    ):
        """Format optional percentage for user-facing output."""
        if value is None:
            return "n/a"

        return f"{value:.2f}%"

    def print_report(
        self,
        report,
    ):
        """
        Print a concise human-readable field-homogeneity report.
        """
        label = report[
            "label"
        ]

        field = report[
            "field"
        ]

        objective = report[
            "objective"
        ]

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"Shimming report — {label}"
        )

        print(
            "=" * 60
        )

        print(
            f"Field points: "
            f"{report['n_points']}"
        )

        print(
            f"Mean field: "
            f"{field['mean_mT']:.6f} mT"
        )

        print(
            f"Standard deviation: "
            f"{field['std_uT']:.3f} uT "
            f"({field['std_ppm']:.2f} ppm)"
        )

        print(
            f"P{objective['low_percentile']:g}-"
            f"P{objective['high_percentile']:g} range: "
            f"{objective['percentile_range_T']*1e3:.6f} mT "
            f"({objective['percentile_range_ppm']:.2f} ppm)"
        )

        print(
            f"Peak-to-peak field: "
            f"{field['peak_to_peak_mT']:.6f} mT "
            f"({field['peak_to_peak_ppm']:.2f} ppm)"
        )

        print(
            f"Peak-to-peak off-resonance: "
            f"{field['peak_to_peak_kHz']:.3f} kHz"
        )

        print(
            "\nObjective"
        )

        print(
            f"  std term: "
            f"{objective['std_term']:.3f}"
        )

        print(
            f"  percentile-range term: "
            f"{objective['percentile_range_term']:.3f}"
        )

        print(
            f"  total cost: "
            f"{objective['total']:.3f}"
        )

        if "shim_state" in report:
            state = report[
                "shim_state"
            ]

            print(
                "\nShim population"
            )

            print(
                f"  active: "
                f"{state['n_active']} / "
                f"{state['n_candidates']}"
            )

            print(
                f"  positive: "
                f"{state['n_positive']}"
            )

            print(
                f"  negative: "
                f"{state['n_negative']}"
            )

            print(
                f"  absent: "
                f"{state['n_absent']}"
            )

        if "improvement" in report:
            improvement = report[
                "improvement"
            ]

            print(
                "\nImprovement relative to pre-shim"
            )

            print(
                f"  standard deviation: "
                f"{self._format_percent(improvement['std_ppm_percent'])}"
            )

            print(
                f"  robust percentile range: "
                f"{self._format_percent(improvement['percentile_range_ppm_percent'])}"
            )

            print(
                f"  peak-to-peak ppm: "
                f"{self._format_percent(improvement['peak_to_peak_ppm_percent'])}"
            )

            print(
                f"  peak-to-peak off-resonance: "
                f"{self._format_percent(improvement['peak_to_peak_kHz_percent'])}"
            )

            print(
                f"  objective cost: "
                f"{self._format_percent(improvement['objective_cost_percent'])}"
            )

        print(
            "=" * 60
        )

    def save_report(
        self,
        report,
        filename,
    ):
        """
        Save one report as JSON.

        Parameters
        ----------
        report : dict
            Report returned by :meth:`evaluate` or :meth:`add_improvement`.

        filename : str or pathlib.Path
            Destination JSON path.

        Returns
        -------
        pathlib.Path
            Saved path.
        """
        filename = Path(
            filename
        )

        filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with filename.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                _json_ready(
                    report
                ),
                f,
                indent=2,
                sort_keys=False,
            )

        return filename
