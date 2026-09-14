"""
Discrete passive-shimming optimization using a precomputed ShimBasis.

Each shim candidate has one discrete state:

    -1  present, negative polarity
     0  absent
    +1  present, positive polarity

The Pymoo problem is element-wise and contains no Magpylib field computation.
Every objective evaluation is therefore only:

    B_total = B_measured + basis @ states

Default objective
-----------------
The scalar objective is a weighted combination of:

1. Standard-deviation homogeneity in ppm.
2. Robust percentile range (default P01-P99) in ppm.

    cost = std_weight * std_ppm
         + percentile_range_weight * percentile_range_ppm

The denominator is abs(mean(B_total)); no fixed target field is imposed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pymoo.core.callback import Callback
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Choice
from pymoo.optimize import minimize


def _json_ready(value):
    """Convert NumPy values/arrays into JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {
            str(k): _json_ready(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_ready(v)
            for v in value
        ]
    return value


class ShimOptimizationProblem(
    ElementwiseProblem
):
    """
    Element-wise discrete passive-shimming problem for Pymoo.

    Parameters
    ----------
    field_map : FieldMap-like object
        Must provide ``B_T``.

    candidate_tray : CandidateTray-like object
        Must provide ``candidate_ids`` and ``candidates``.

    allowed_states : iterable of int, optional
        Discrete candidate states. Default ``(-1, 0, +1)``.

    std_weight : float, optional
        Weight of standard-deviation ppm term. Default 0.75.

    percentile_range_weight : float, optional
        Weight of robust percentile-range ppm term. Default 0.25.

    low_percentile : float, optional
        Lower robust-range percentile. Default 1.

    high_percentile : float, optional
        Upper robust-range percentile. Default 99.

    debug : bool, optional
        Enable detailed diagnostics.

    Notes
    -----
    The basis may be attached after construction using :meth:`attach_basis`.
    This matches the top-level workflow where optimization variables are
    defined before the expensive basis calculation.
    """

    def __init__(
        self,
        field_map,
        candidate_tray,
        allowed_states=(-1, 0, 1),
        std_weight=0.75,
        percentile_range_weight=0.25,
        low_percentile=1.0,
        high_percentile=99.0,
        debug=False,
    ):
        self.field_map = field_map
        self.candidate_tray = candidate_tray
        self.candidate_ids = list(
            candidate_tray.candidate_ids
        )

        self.allowed_states = tuple(
            int(
                state
            )
            for state in allowed_states
        )

        self.std_weight = float(
            std_weight
        )

        self.percentile_range_weight = float(
            percentile_range_weight
        )

        self.low_percentile = float(
            low_percentile
        )

        self.high_percentile = float(
            high_percentile
        )

        self.debug = bool(
            debug
        )

        self.shim_basis = None

        variables = {
            candidate_id:
                Choice(
                    options=list(
                        self.allowed_states
                    )
                )
            for candidate_id
            in self.candidate_ids
        }

        super().__init__(
            vars=variables,
            n_obj=1,
        )

    @property
    def n_candidates(self):
        """Number of discrete shim variables."""
        return int(
            len(
                self.candidate_ids
            )
        )

    def validate(self):
        """
        Validate problem definition before optimization.

        This validation does not require a ShimBasis to be attached yet.
        """
        if self.n_candidates == 0:
            raise ValueError(
                "candidate_tray contains no candidates."
            )

        if len(
            set(
                self.candidate_ids
            )
        ) != self.n_candidates:
            raise ValueError(
                "candidate_ids must be unique."
            )

        if len(
            self.allowed_states
        ) < 2:
            raise ValueError(
                "At least two allowed states are required."
            )

        if 0 not in self.allowed_states:
            raise ValueError(
                "allowed_states must contain 0 for the absent state."
            )

        if self.std_weight < 0:
            raise ValueError(
                "std_weight must be >= 0."
            )

        if self.percentile_range_weight < 0:
            raise ValueError(
                "percentile_range_weight must be >= 0."
            )

        if (
            self.std_weight
            + self.percentile_range_weight
            <= 0
        ):
            raise ValueError(
                "At least one objective weight must be positive."
            )

        if not (
            0 <= self.low_percentile
            < self.high_percentile
            <= 100
        ):
            raise ValueError(
                "Percentiles must satisfy "
                "0 <= low < high <= 100."
            )

        B = np.asarray(
            self.field_map.B_T,
            dtype=float,
        ).reshape(
            -1
        )

        if len(
            B
        ) == 0:
            raise ValueError(
                "field_map contains no field samples."
            )

        if not np.all(
            np.isfinite(
                B
            )
        ):
            raise ValueError(
                "field_map contains non-finite field values."
            )

        if self.debug:
            print(
                "\nShimOptimizationProblem validation passed"
            )
            print(
                f"  Variables: {self.n_candidates}"
            )
            print(
                f"  States: {self.allowed_states}"
            )
            print(
                f"  Cost weights: std={self.std_weight}, "
                f"range={self.percentile_range_weight}"
            )
            print(
                f"  Robust range: "
                f"P{self.low_percentile:g}-P{self.high_percentile:g}"
            )

        return True

    def attach_basis(
        self,
        shim_basis,
    ):
        """
        Attach a precomputed :class:`ShimBasis`.

        Candidate IDs and matrix dimensions are checked before use.
        """
        if not getattr(
            shim_basis,
            "is_computed",
            False,
        ):
            raise ValueError(
                "ShimBasis must be computed before attachment."
            )

        if list(
            shim_basis.candidate_ids
        ) != self.candidate_ids:
            raise ValueError(
                "ShimBasis candidate ordering does not match "
                "ShimOptimizationProblem."
            )

        if (
            shim_basis.basis_T.shape[
                1
            ]
            != self.n_candidates
        ):
            raise ValueError(
                "ShimBasis column count does not match "
                "number of optimization variables."
            )

        if (
            shim_basis.basis_T.shape[
                0
            ]
            != len(
                self.field_map.B_T
            )
        ):
            raise ValueError(
                "ShimBasis row count does not match field-map samples."
            )

        self.shim_basis = (
            shim_basis
        )

        return self

    def states_from_mapping(
        self,
        mapping,
    ):
        """
        Convert Pymoo's variable dictionary to an ordered state vector.
        """
        return np.asarray(
            [
                int(
                    mapping[
                        candidate_id
                    ]
                )
                for candidate_id
                in self.candidate_ids
            ],
            dtype=int,
        )

    def mapping_from_states(
        self,
        states,
    ):
        """Convert an ordered state vector to candidate-ID dictionary."""
        states = np.asarray(
            states,
            dtype=int,
        ).reshape(
            -1
        )

        if len(
            states
        ) != self.n_candidates:
            raise ValueError(
                "Invalid state-vector length."
            )

        return {
            candidate_id:
                int(
                    state
                )
            for candidate_id, state in zip(
                self.candidate_ids,
                states,
            )
        }

    def field_from_states(
        self,
        states,
    ):
        """Return total measured+shim field for a discrete state vector."""
        if self.shim_basis is None:
            raise RuntimeError(
                "Attach ShimBasis before evaluating the problem."
            )

        return (
            np.asarray(
                self.field_map.B_T,
                dtype=float,
            )
            + self.shim_basis.field_from_states(
                states
            )
        )

    def cost_components(
        self,
        B_total_T,
    ):
        """
        Calculate all scalar objective terms.

        Returns
        -------
        dict
            Mean field, standard-deviation ppm, percentile-range ppm,
            weighted terms, and total cost.
        """
        B_total_T = np.asarray(
            B_total_T,
            dtype=float,
        ).reshape(
            -1
        )

        mean_T = float(
            np.mean(
                B_total_T
            )
        )

        denominator = abs(
            mean_T
        )

        if denominator <= np.finfo(
            float
        ).eps:
            # A collapsed mean field is physically unacceptable and should
            # never become an attractive solution.
            return {
                "mean_field_T":
                    mean_T,
                "std_T":
                    np.inf,
                "std_ppm":
                    np.inf,
                "low_percentile_T":
                    np.nan,
                "high_percentile_T":
                    np.nan,
                "percentile_range_T":
                    np.inf,
                "percentile_range_ppm":
                    np.inf,
                "std_term":
                    np.inf,
                "percentile_range_term":
                    np.inf,
                "total":
                    np.inf,
            }

        std_T = float(
            np.std(
                B_total_T
            )
        )

        low_T = float(
            np.percentile(
                B_total_T,
                self.low_percentile,
            )
        )

        high_T = float(
            np.percentile(
                B_total_T,
                self.high_percentile,
            )
        )

        range_T = (
            high_T
            - low_T
        )

        std_ppm = (
            1e6
            * std_T
            / denominator
        )

        range_ppm = (
            1e6
            * range_T
            / denominator
        )

        std_term = (
            self.std_weight
            * std_ppm
        )

        range_term = (
            self.percentile_range_weight
            * range_ppm
        )

        total = (
            std_term
            + range_term
        )

        return {
            "mean_field_T":
                mean_T,
            "std_T":
                std_T,
            "std_ppm":
                std_ppm,
            "low_percentile":
                self.low_percentile,
            "high_percentile":
                self.high_percentile,
            "low_percentile_T":
                low_T,
            "high_percentile_T":
                high_T,
            "percentile_range_T":
                range_T,
            "percentile_range_ppm":
                range_ppm,
            "std_weight":
                self.std_weight,
            "percentile_range_weight":
                self.percentile_range_weight,
            "std_term":
                std_term,
            "percentile_range_term":
                range_term,
            "total":
                total,
        }

    def evaluate_states(
        self,
        states,
    ):
        """
        Evaluate one explicit state vector outside Pymoo.
        """
        B_total = (
            self.field_from_states(
                states
            )
        )

        return self.cost_components(
            B_total
        )

    def _evaluate(
        self,
        x,
        out,
        *args,
        **kwargs,
    ):
        states = (
            self.states_from_mapping(
                x
            )
        )

        B_total_T = (
            self.field_from_states(
                states
            )
        )

        components = (
            self.cost_components(
                B_total_T
            )
        )

        out[
            "F"
        ] = components[
            "total"
        ]


class _OptimizerHistory(
    Callback
):
    """Internal callback recording compact GA convergence statistics."""

    def __init__(
        self,
        problem,
    ):
        super().__init__()

        self.problem = (
            problem
        )

        self.records = []

    def notify(
        self,
        algorithm,
    ):
        F = np.asarray(
            algorithm.pop.get(
                "F"
            ),
            dtype=float,
        ).reshape(
            -1
        )

        best_index = int(
            np.argmin(
                F
            )
        )

        best_X = (
            algorithm.pop[
                best_index
            ].X
        )

        best_states = (
            self.problem.states_from_mapping(
                best_X
            )
        )

        self.records.append(
            {
                "generation":
                    int(
                        algorithm.n_gen
                    ),
                "best_cost":
                    float(
                        np.min(
                            F
                        )
                    ),
                "mean_cost":
                    float(
                        np.mean(
                            F
                        )
                    ),
                "std_cost":
                    float(
                        np.std(
                            F
                        )
                    ),
                "n_active_best":
                    int(
                        np.count_nonzero(
                            best_states
                        )
                    ),
                "n_positive_best":
                    int(
                        np.sum(
                            best_states == 1
                        )
                    ),
                "n_negative_best":
                    int(
                        np.sum(
                            best_states == -1
                        )
                    ),
            }
        )


class ShimOptimizer:
    """
    Run the Pymoo mixed-variable genetic algorithm.

    Parameters
    ----------
    problem : ShimOptimizationProblem
        Problem with an attached ShimBasis.

    population_size : int, optional
        GA population size.

    n_generations : int, optional
        Number of generations.

    random_seed : int, optional
        Reproducibility seed.

    eliminate_duplicates : bool, optional
        Passed to ``MixedVariableGA``.

    verbose : bool, optional
        Print Pymoo convergence table.

    debug : bool, optional
        Print additional optimizer diagnostics.
    """

    def __init__(
        self,
        problem,
        population_size=500,
        n_generations=200,
        random_seed=42,
        eliminate_duplicates=True,
        verbose=True,
        debug=False,
    ):
        if not isinstance(
            problem,
            ShimOptimizationProblem,
        ):
            raise TypeError(
                "problem must be a ShimOptimizationProblem."
            )

        self.problem = problem
        self.population_size = int(
            population_size
        )
        self.n_generations = int(
            n_generations
        )
        self.random_seed = int(
            random_seed
        )
        self.eliminate_duplicates = bool(
            eliminate_duplicates
        )
        self.verbose = bool(
            verbose
        )
        self.debug = bool(
            debug
        )

        self.result = None
        self.history = []
        self.best_states = None
        self.best_cost = None
        self.best_components = None

    def validate(self):
        """Validate optimizer and attached problem."""
        self.problem.validate()

        if self.problem.shim_basis is None:
            raise RuntimeError(
                "ShimOptimizationProblem must have a ShimBasis attached "
                "before optimization."
            )

        if self.population_size <= 0:
            raise ValueError(
                "population_size must be positive."
            )

        if self.n_generations <= 0:
            raise ValueError(
                "n_generations must be positive."
            )

        return True

    def run(self):
        """
        Run the mixed-variable GA.

        Returns
        -------
        pymoo.core.result.Result
            Pymoo optimization result.
        """
        self.validate()

        algorithm = MixedVariableGA(
            pop_size=
                self.population_size,
            eliminate_duplicates=
                self.eliminate_duplicates,
        )

        callback = _OptimizerHistory(
            self.problem
        )

        self.result = minimize(
            self.problem,
            algorithm,
            (
                "n_gen",
                self.n_generations,
            ),
            seed=
                self.random_seed,
            callback=
                callback,
            verbose=
                self.verbose,
        )

        if self.result.X is None:
            raise RuntimeError(
                "Optimizer returned no solution."
            )

        self.history = (
            callback.records
        )

        self.best_states = (
            self.problem.states_from_mapping(
                self.result.X
            )
        )

        self.best_components = (
            self.problem.evaluate_states(
                self.best_states
            )
        )

        self.best_cost = float(
            self.best_components[
                "total"
            ]
        )

        if self.debug:
            print(
                "\nOptimization complete"
            )
            print(
                f"  Best cost: "
                f"{self.best_cost:.3f}"
            )
            print(
                f"  Active magnets: "
                f"{np.count_nonzero(self.best_states)}"
            )
            print(
                f"  Positive: "
                f"{np.sum(self.best_states == 1)}"
            )
            print(
                f"  Negative: "
                f"{np.sum(self.best_states == -1)}"
            )

        return self.result

    def _require_result(self):
        if self.result is None:
            raise RuntimeError(
                "ShimOptimizer.run() must be called first."
            )

    def save_history(
        self,
        filename,
    ):
        """Save generation-by-generation convergence history to JSON."""
        self._require_result()

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
                    {
                        "population_size":
                            self.population_size,
                        "n_generations":
                            self.n_generations,
                        "random_seed":
                            self.random_seed,
                        "history":
                            self.history,
                    }
                ),
                f,
                indent=2,
            )

        return filename

    def save_best_solution(
        self,
        filename,
    ):
        """Save ordered state vector and final objective components to JSON."""
        self._require_result()

        filename = Path(
            filename
        )

        filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        solution = {
            "candidate_ids":
                self.problem.candidate_ids,
            "states":
                self.best_states,
            "n_candidates":
                self.problem.n_candidates,
            "n_active":
                int(
                    np.count_nonzero(
                        self.best_states
                    )
                ),
            "n_positive":
                int(
                    np.sum(
                        self.best_states == 1
                    )
                ),
            "n_negative":
                int(
                    np.sum(
                        self.best_states == -1
                    )
                ),
            "n_absent":
                int(
                    np.sum(
                        self.best_states == 0
                    )
                ),
            "cost_components":
                self.best_components,
        }

        with filename.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                _json_ready(
                    solution
                ),
                f,
                indent=2,
            )

        return filename

    def statistics(self):
        """Return compact final optimizer statistics."""
        self._require_result()

        return {
            "population_size":
                self.population_size,
            "n_generations":
                self.n_generations,
            "random_seed":
                self.random_seed,
            "best_cost":
                self.best_cost,
            "n_active":
                int(
                    np.count_nonzero(
                        self.best_states
                    )
                ),
            "n_positive":
                int(
                    np.sum(
                        self.best_states == 1
                    )
                ),
            "n_negative":
                int(
                    np.sum(
                        self.best_states == -1
                    )
                ),
            "cost_components":
                self.best_components,
        }
