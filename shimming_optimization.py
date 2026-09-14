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
from types import SimpleNamespace

import numpy as np

from pymoo.core.callback import Callback

from pymoo.core.mixed import (
    MixedVariableGA,
    MixedVariableMating,
    MixedVariableDuplicateElimination,
)
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling

from pymoo.core.variable import Choice
from pymoo.optimize import minimize
from pymoo.core.population import Population

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

class SparseMixedVariableSampling(Sampling):

    def __init__(
        self,
        state_probabilities=None,
        include_zero_baseline=True,
    ):
        super().__init__()

        if state_probabilities is None:
            state_probabilities = {
                -1: 0.05,
                 0: 0.90,
                +1: 0.05,
            }

        self.state_probabilities = {
            int(k): float(v)
            for k, v in state_probabilities.items()
        }

        self.include_zero_baseline = bool(
            include_zero_baseline
        )

    def _do(
        self,
        problem,
        n_samples,
        random_state=None,
        **kwargs,
    ):
        if random_state is None:
            random_state = np.random.default_rng()

        variable_names = list(
            problem.vars.keys()
        )

        allowed_states = np.asarray(
            problem.allowed_states,
            dtype=int,
        )

        probabilities = np.asarray(
            [
                self.state_probabilities[
                    int(state)
                ]
                for state in allowed_states
            ],
            dtype=float,
        )

        probabilities /= np.sum(
            probabilities
        )

        X = []

        # Exact no-shim solution
        if self.include_zero_baseline:
            X.append(
                {
                    name: 0
                    for name in variable_names
                }
            )

        # Sparse random remainder
        while len(X) < n_samples:

            states = random_state.choice(
                allowed_states,
                size=len(variable_names),
                p=probabilities,
            )

            X.append(
                {
                    name: int(state)
                    for name, state
                    in zip(
                        variable_names,
                        states,
                    )
                }
            )

        return X



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
    Discrete passive-shimming optimizer.

    Two optimization backends are retained:

    1. ``iterated_greedy`` (default)
       Greedy coordinate descent -> random perturbation -> greedy descent,
       repeated over many restarts. This is the preferred search for the
       present sparse ternary shim problem.

    2. ``ga``
       Pymoo MixedVariableGA. The GA implementation is retained intact as an
       optional backend for later comparison or hybridization.

    Parameters
    ----------
    problem : ShimOptimizationProblem
        Problem with an attached ShimBasis.

    population_size : int, optional
        GA population size. Used only by the ``ga`` backend.

    n_generations : int, optional
        GA generation count. Used only by the ``ga`` backend.

    random_seed : int, optional
        Reproducibility seed for all stochastic operations.

    eliminate_duplicates : bool, optional
        Retained for GA compatibility.

    verbose : bool, optional
        Print optimization progress.

    debug : bool, optional
        Print additional optimizer diagnostics.

    initial_state_probabilities : dict, optional
        Sparse random state probabilities for GA initialization.

    initial_states : array-like, optional
        Explicit starting state vector. For iterated greedy this is refined
        to a one-coordinate local optimum before perturbation restarts.

    search_method : {"iterated_greedy", "ga"}, optional
        Optimization backend. Default ``"iterated_greedy"``.

    n_restarts : int, optional
        Number of perturb-and-greedy restarts for iterated greedy.

    perturbation_sizes : iterable of int, optional
        Number of coordinates changed during random basin jumps.

    max_greedy_steps : int, optional
        Maximum accepted coordinate-descent moves per local search.

    greedy_tolerance : float, optional
        Minimum absolute cost improvement required to accept a greedy move.
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
        initial_state_probabilities=None,
        initial_states=None,
        search_method="iterated_greedy",
        n_restarts=50,
        perturbation_sizes=(2, 3, 5, 8, 10),
        max_greedy_steps=100,
        greedy_tolerance=0.0,
    ):
        if not isinstance(problem, ShimOptimizationProblem):
            raise TypeError(
                "problem must be a ShimOptimizationProblem."
            )

        self.problem = problem
        self.population_size = int(population_size)
        self.n_generations = int(n_generations)
        self.random_seed = int(random_seed)
        self.eliminate_duplicates = bool(eliminate_duplicates)
        self.verbose = bool(verbose)
        self.debug = bool(debug)

        if initial_state_probabilities is None:
            initial_state_probabilities = {
                -1: 0.05,
                 0: 0.90,
                +1: 0.05,
            }

        self.initial_state_probabilities = dict(
            initial_state_probabilities
        )

        self.initial_states = (
            None
            if initial_states is None
            else np.asarray(
                initial_states,
                dtype=int,
            ).reshape(-1)
        )

        self.search_method = str(search_method).strip().lower()
        self.n_restarts = int(n_restarts)
        self.perturbation_sizes = tuple(
            int(value)
            for value in perturbation_sizes
        )
        self.max_greedy_steps = int(max_greedy_steps)
        self.greedy_tolerance = float(greedy_tolerance)

        self.baseline_states = None
        self.baseline_components = None
        self.baseline_cost = None

        # GA-specific outputs are intentionally retained.
        self.ga_best_states = None
        self.ga_best_components = None
        self.ga_best_cost = None

        # Iterated-greedy-specific outputs.
        self.initial_local_states = None
        self.initial_local_components = None
        self.initial_local_cost = None
        self.iterated_best_states = None
        self.iterated_best_components = None
        self.iterated_best_cost = None

        self.best_source = None
        self.result = None
        self.history = []
        self.best_states = None
        self.best_cost = None
        self.best_components = None

    # ------------------------------------------------------------------
    # Validation and common helpers
    # ------------------------------------------------------------------

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

        if self.initial_states is not None:
            if len(self.initial_states) != self.problem.n_candidates:
                raise ValueError(
                    "initial_states length does not match "
                    "the number of shim candidates."
                )

            if not np.all(
                np.isin(
                    self.initial_states,
                    self.problem.allowed_states,
                )
            ):
                raise ValueError(
                    "initial_states contains values outside "
                    f"{self.problem.allowed_states}."
                )

        if self.search_method not in {
            "iterated_greedy",
            "ils",
            "greedy",
            "ga",
            "genetic_algorithm",
        }:
            raise ValueError(
                "search_method must be 'iterated_greedy' or 'ga'."
            )

        if self.n_restarts < 0:
            raise ValueError(
                "n_restarts must be >= 0."
            )

        if self.max_greedy_steps <= 0:
            raise ValueError(
                "max_greedy_steps must be positive."
            )

        if len(self.perturbation_sizes) == 0:
            raise ValueError(
                "perturbation_sizes must contain at least one value."
            )

        if any(value <= 0 for value in self.perturbation_sizes):
            raise ValueError(
                "perturbation_sizes must contain positive integers."
            )

        return True

    def _set_baseline(self):
        """Evaluate and store the exact zero-shim baseline."""
        self.baseline_states = np.zeros(
            self.problem.n_candidates,
            dtype=int,
        )

        self.baseline_components = (
            self.problem.evaluate_states(
                self.baseline_states
            )
        )

        self.baseline_cost = float(
            self.baseline_components["total"]
        )

    def _perturb_states(
        self,
        states,
        n_changes,
        rng,
    ):
        """
        Randomly perturb exactly ``n_changes`` coordinates.

        Every selected coordinate is forced to one of its other allowed
        states, so the requested Hamming distance is exact.
        """
        states = np.asarray(
            states,
            dtype=int,
        ).copy()

        n_changes = min(
            int(n_changes),
            len(states),
        )

        indices = rng.choice(
            len(states),
            size=n_changes,
            replace=False,
        )

        allowed_states = np.asarray(
            self.problem.allowed_states,
            dtype=int,
        )

        for index in indices:
            alternatives = allowed_states[
                allowed_states != states[index]
            ]
            states[index] = int(
                rng.choice(alternatives)
            )

        return states

    def _greedy_local_search(
        self,
        start_states,
        max_steps=None,
        verbose=False,
        label=None,
    ):
        """
        Greedy one-coordinate descent from an arbitrary starting state.

        This implementation uses incremental field updates. For a change
        ``s_j -> s'_j`` it evaluates

            B_trial = B_current + (s'_j-s_j) * basis[:, j]

        rather than recomputing ``basis @ states`` for every trial. This keeps
        repeated local searches fast while preserving the exact objective.
        """
        states = np.asarray(
            start_states,
            dtype=int,
        ).copy()

        if len(states) != self.problem.n_candidates:
            raise ValueError(
                "Invalid start-state vector length."
            )

        if not np.all(
            np.isin(
                states,
                self.problem.allowed_states,
            )
        ):
            raise ValueError(
                "start_states contains invalid state values."
            )

        if max_steps is None:
            max_steps = self.max_greedy_steps
        max_steps = int(max_steps)

        basis_T = np.asarray(
            self.problem.shim_basis.basis_T,
            dtype=float,
        )

        current_B = self.problem.field_from_states(
            states
        )
        current_components = self.problem.cost_components(
            current_B
        )
        current_cost = float(
            current_components["total"]
        )

        allowed_states = tuple(
            int(value)
            for value in self.problem.allowed_states
        )

        history = []
        termination_reason = "max_steps"

        for step in range(max_steps):
            best_cost = current_cost
            best_index = None
            best_state = None
            best_B = None
            best_components = None

            for candidate_index in range(
                self.problem.n_candidates
            ):
                current_state = int(
                    states[candidate_index]
                )

                basis_column = basis_T[
                    :,
                    candidate_index,
                ]

                for new_state in allowed_states:
                    if new_state == current_state:
                        continue

                    delta_state = (
                        new_state
                        - current_state
                    )

                    trial_B = (
                        current_B
                        + delta_state
                        * basis_column
                    )

                    components = (
                        self.problem.cost_components(
                            trial_B
                        )
                    )
                    cost = float(
                        components["total"]
                    )

                    if (
                        cost
                        < best_cost
                        - self.greedy_tolerance
                    ):
                        best_cost = cost
                        best_index = candidate_index
                        best_state = int(new_state)
                        best_B = trial_B
                        best_components = components

            # One-coordinate local optimum reached.
            if best_index is None:
                termination_reason = "local_optimum"
                break

            previous_cost = current_cost
            previous_state = int(
                states[best_index]
            )

            states[best_index] = best_state
            current_B = best_B
            current_components = best_components
            current_cost = float(best_cost)

            record = {
                "step": int(step + 1),
                "candidate_index": int(best_index),
                "candidate_id": self.problem.candidate_ids[
                    best_index
                ],
                "old_state": int(previous_state),
                "new_state": int(best_state),
                "previous_cost": float(previous_cost),
                "new_cost": float(current_cost),
                "improvement": float(
                    previous_cost
                    - current_cost
                ),
                "n_active": int(
                    np.count_nonzero(states)
                ),
            }
            history.append(record)

            if verbose:
                prefix = ""
                if label:
                    prefix = f"[{label}] "
                print(
                    f"  {prefix}greedy step {step + 1:3d}: "
                    f"{record['candidate_id']} "
                    f"{previous_state:+d}->{best_state:+d} | "
                    f"cost {previous_cost:.3f} "
                    f"-> {current_cost:.3f} | "
                    f"active={record['n_active']}"
                )

        return {
            "states": states,
            "components": current_components,
            "cost": float(current_cost),
            "B_total_T": current_B,
            "history": history,
            "n_steps": int(len(history)),
            "termination_reason": termination_reason,
        }

    # ------------------------------------------------------------------
    # Public dispatcher
    # ------------------------------------------------------------------

    def run(self):
        """
        Run the selected optimization backend.

        The default backend is iterated greedy. Set

            search_method="ga"

        when constructing ``ShimOptimizer`` to use the retained Pymoo GA.
        """
        self.validate()

        if self.search_method in {
            "ga",
            "genetic_algorithm",
        }:
            return self.run_ga()

        return self.run_iterated_greedy()

    # ------------------------------------------------------------------
    # Iterated greedy / discrete basin hopping
    # ------------------------------------------------------------------

    def run_iterated_greedy(self):
        """
        Run greedy -> perturb -> greedy iterated local search.

        Workflow
        --------
        1. Evaluate zero-shim baseline.
        2. Greedily refine ``initial_states`` when supplied; otherwise greedily
           descend from zero.
        3. Randomly perturb the current global-best local optimum by 2/3/5/8/10
           coordinates (configurable).
        4. Greedily descend from the perturbed state to a new local optimum.
        5. Retain it only when it improves the global best.
        6. Repeat for ``n_restarts`` basin jumps.
        """
        self.validate()
        self._set_baseline()

        rng = np.random.default_rng(
            self.random_seed
        )

        if self.initial_states is None:
            start_states = self.baseline_states.copy()
            start_label = "zero-shim"
        else:
            start_states = np.asarray(
                self.initial_states,
                dtype=int,
            ).copy()
            start_label = "supplied initial_states"

        start_components = self.problem.evaluate_states(
            start_states
        )
        start_cost = float(
            start_components["total"]
        )

        if self.verbose:
            print(
                "\nIterated-greedy initialization"
            )
            print(
                f"  Starting state: {start_label}"
            )
            print(
                f"  Starting cost: {start_cost:.3f}"
            )
            print(
                f"  Starting active magnets: "
                f"{np.count_nonzero(start_states)}"
            )
            print(
                f"  Zero-shim baseline: "
                f"{self.baseline_cost:.3f}"
            )

        initial_local = self._greedy_local_search(
            start_states=start_states,
            max_steps=self.max_greedy_steps,
            verbose=False,
            label="initial",
        )

        self.initial_local_states = (
            initial_local["states"].copy()
        )
        self.initial_local_components = (
            initial_local["components"]
        )
        self.initial_local_cost = float(
            initial_local["cost"]
        )

        best_states = self.initial_local_states.copy()
        best_components = self.initial_local_components
        best_cost = self.initial_local_cost

        # Guard against a pathological supplied seed.
        if self.baseline_cost < best_cost:
            best_states = self.baseline_states.copy()
            best_components = self.baseline_components
            best_cost = self.baseline_cost

        self.history = [
            {
                "restart": 0,
                "source": "initial_local_search",
                "perturbation_size": 0,
                "perturbed_cost": float(start_cost),
                "local_cost": float(
                    self.initial_local_cost
                ),
                "best_cost": float(best_cost),
                "greedy_steps": int(
                    initial_local["n_steps"]
                ),
                "accepted": bool(
                    self.initial_local_cost
                    <= best_cost
                ),
                "n_active": int(
                    np.count_nonzero(best_states)
                ),
                "termination_reason": initial_local[
                    "termination_reason"
                ],
            }
        ]

        if self.verbose:
            print(
                "\nInitial greedy local optimum"
            )
            print(
                f"  Cost: {best_cost:.3f}"
            )
            print(
                f"  Active magnets: "
                f"{np.count_nonzero(best_states)}"
            )
            print(
                f"  Greedy accepted moves: "
                f"{initial_local['n_steps']}"
            )
            print(
                f"  Termination: "
                f"{initial_local['termination_reason']}"
            )
            print(
                "\nIterated greedy / random perturbation"
            )
            print(
                f"  Restarts: {self.n_restarts}"
            )
            print(
                f"  Perturbation sizes: "
                f"{self.perturbation_sizes}"
            )
            print(
                f"  Max greedy steps/restart: "
                f"{self.max_greedy_steps}"
            )
            print(
                "\nrestart | jump | perturbed | local min | "
                "global best | steps | active | accepted"
            )
            print(
                "-" * 83
            )

        for restart in range(
            1,
            self.n_restarts + 1,
        ):
            perturbation_size = int(
                rng.choice(
                    self.perturbation_sizes
                )
            )

            # Always jump from the best known basin. This makes each restart
            # an independent attempt to leave the current best local optimum.
            perturbed_states = self._perturb_states(
                states=best_states,
                n_changes=perturbation_size,
                rng=rng,
            )

            perturbed_components = (
                self.problem.evaluate_states(
                    perturbed_states
                )
            )
            perturbed_cost = float(
                perturbed_components["total"]
            )

            local = self._greedy_local_search(
                start_states=perturbed_states,
                max_steps=self.max_greedy_steps,
                verbose=False,
                label=f"restart {restart}",
            )

            local_cost = float(
                local["cost"]
            )

            accepted = bool(
                local_cost
                < best_cost
                - self.greedy_tolerance
            )

            if accepted:
                best_states = local["states"].copy()
                best_components = local["components"]
                best_cost = local_cost

            record = {
                "restart": int(restart),
                "source": "perturb_then_greedy",
                "perturbation_size": int(
                    perturbation_size
                ),
                "perturbed_cost": float(
                    perturbed_cost
                ),
                "local_cost": float(local_cost),
                "best_cost": float(best_cost),
                "greedy_steps": int(
                    local["n_steps"]
                ),
                "accepted": accepted,
                "n_active_local": int(
                    np.count_nonzero(
                        local["states"]
                    )
                ),
                "n_active_best": int(
                    np.count_nonzero(
                        best_states
                    )
                ),
                "termination_reason": local[
                    "termination_reason"
                ],
            }
            self.history.append(record)

            if self.verbose:
                print(
                    f"{restart:7d} | "
                    f"{perturbation_size:4d} | "
                    f"{perturbed_cost:9.3f} | "
                    f"{local_cost:9.3f} | "
                    f"{best_cost:11.3f} | "
                    f"{local['n_steps']:5d} | "
                    f"{np.count_nonzero(local['states']):6d} | "
                    f"{'YES' if accepted else 'no'}"
                )

        self.iterated_best_states = best_states.copy()
        self.iterated_best_components = best_components
        self.iterated_best_cost = float(best_cost)

        self.best_states = best_states.copy()
        self.best_components = best_components
        self.best_cost = float(best_cost)
        self.best_source = "iterated_greedy"

        # Provide a result object with the fields most downstream code expects.
        self.result = SimpleNamespace(
            X=self.problem.mapping_from_states(
                self.best_states
            ),
            F=np.asarray(
                [self.best_cost],
                dtype=float,
            ),
            search_method="iterated_greedy",
        )

        if self.verbose:
            n_accepted = int(
                sum(
                    bool(record.get("accepted", False))
                    for record in self.history[1:]
                )
            )

            print(
                "\nOptimization selection"
            )
            print(
                f"  Method: iterated_greedy"
            )
            print(
                f"  Zero-shim cost: "
                f"{self.baseline_cost:.3f}"
            )
            print(
                f"  Initial local cost: "
                f"{self.initial_local_cost:.3f}"
            )
            print(
                f"  Final best cost: "
                f"{self.best_cost:.3f}"
            )
            print(
                f"  Accepted better basins: "
                f"{n_accepted} / {self.n_restarts}"
            )
            print(
                f"  Active magnets: "
                f"{np.count_nonzero(self.best_states)}"
            )
            print(
                f"  Positive magnets: "
                f"{np.sum(self.best_states == +1)}"
            )
            print(
                f"  Negative magnets: "
                f"{np.sum(self.best_states == -1)}"
            )

        return self.result

    # ------------------------------------------------------------------
    # Retained GA backend
    # ------------------------------------------------------------------

    def run_ga(self):
        """
        Run the retained Pymoo MixedVariableGA backend.

        The GA keeps the locally seeded initialization developed previously:
        when ``initial_states`` is supplied, 80% of generation 1 is composed
        of controlled perturbations around that state and 20% remains sparse
        random for diversity.
        """
        self.validate()
        self._set_baseline()

        sampling_operator = SparseMixedVariableSampling(
            state_probabilities=
                self.initial_state_probabilities,
            include_zero_baseline=False,
        )

        initial_population = sampling_operator.do(
            self.problem,
            self.population_size,
        )

        rng = np.random.default_rng(
            self.random_seed
        )

        seed_states = None
        seed_components = None
        seed_cost = None

        if self.initial_states is not None:
            seed_states = np.asarray(
                self.initial_states,
                dtype=int,
            ).copy()
            seed_components = self.problem.evaluate_states(
                seed_states
            )
            seed_cost = float(
                seed_components["total"]
            )

            initial_population[0].X = (
                self.problem.mapping_from_states(
                    seed_states
                )
            )

            population_index = 1
            n_local_target = int(
                0.80
                * self.population_size
            )
            n_local_perturbed = max(
                0,
                n_local_target - 1,
            )

            counts = []
            remaining = n_local_perturbed
            for fraction in (0.20, 0.20, 0.20):
                count = min(
                    int(
                        fraction
                        * self.population_size
                    ),
                    remaining,
                )
                counts.append(count)
                remaining -= count
            counts.append(remaining)

            perturbation_plan = list(
                zip(
                    (1, 2, 5, 10),
                    counts,
                )
            )

            for n_changes, n_copies in perturbation_plan:
                for _ in range(n_copies):
                    if population_index >= self.population_size:
                        break

                    perturbed_states = self._perturb_states(
                        states=seed_states,
                        n_changes=n_changes,
                        rng=rng,
                    )

                    initial_population[
                        population_index
                    ].X = (
                        self.problem.mapping_from_states(
                            perturbed_states
                        )
                    )
                    population_index += 1

            n_sparse_random = (
                self.population_size
                - population_index
            )

            if self.verbose:
                print(
                    "\nGA initialization"
                )
                print(
                    f"  Seed cost: {seed_cost:.3f}"
                )
                print(
                    f"  Seed active magnets: "
                    f"{np.count_nonzero(seed_states)}"
                )
                print(
                    f"  Local seeded population: "
                    f"{population_index}"
                )
                print(
                    f"  Sparse random population: "
                    f"{n_sparse_random}"
                )
        else:
            initial_population[0].X = (
                self.problem.mapping_from_states(
                    self.baseline_states
                )
            )

        initialization_duplicate_elimination = (
            MixedVariableDuplicateElimination()
        )
        mating_duplicate_elimination = (
            MixedVariableDuplicateElimination()
        )
        mating = MixedVariableMating(
            eliminate_duplicates=
                mating_duplicate_elimination,
        )

        algorithm = MixedVariableGA(
            pop_size=self.population_size,
            sampling=initial_population,
            mating=mating,
            eliminate_duplicates=
                initialization_duplicate_elimination,
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
            seed=self.random_seed,
            callback=callback,
            verbose=self.verbose,
        )

        if self.result.X is None:
            raise RuntimeError(
                "Optimizer returned no solution."
            )

        self.history = callback.records

        self.ga_best_states = (
            self.problem.states_from_mapping(
                self.result.X
            )
        )
        self.ga_best_components = (
            self.problem.evaluate_states(
                self.ga_best_states
            )
        )
        self.ga_best_cost = float(
            self.ga_best_components["total"]
        )

        # Best of zero baseline, supplied seed, and GA result.
        self.best_states = self.baseline_states.copy()
        self.best_components = self.baseline_components
        self.best_cost = self.baseline_cost
        self.best_source = "zero_shim_baseline"

        if (
            seed_states is not None
            and seed_cost < self.best_cost
        ):
            self.best_states = seed_states.copy()
            self.best_components = seed_components
            self.best_cost = seed_cost
            self.best_source = "initial_greedy_seed"

        if self.ga_best_cost < self.best_cost:
            self.best_states = self.ga_best_states.copy()
            self.best_components = self.ga_best_components
            self.best_cost = self.ga_best_cost
            self.best_source = "genetic_algorithm"

        if self.verbose:
            print(
                "\nOptimization selection"
            )
            print(
                "  Method: genetic_algorithm"
            )
            print(
                f"  Zero-shim cost: "
                f"{self.baseline_cost:.3f}"
            )
            if seed_cost is not None:
                print(
                    f"  Greedy-seed cost: "
                    f"{seed_cost:.3f}"
                )
            print(
                f"  GA best cost: "
                f"{self.ga_best_cost:.3f}"
            )
            print(
                f"  Returned solution: "
                f"{self.best_source}"
            )
            print(
                f"  Returned cost: "
                f"{self.best_cost:.3f}"
            )

        return self.result

    # ------------------------------------------------------------------
    # Saving/reporting
    # ------------------------------------------------------------------

    def _require_result(self):
        if self.result is None:
            raise RuntimeError(
                "ShimOptimizer.run() must be called first."
            )

    def save_history(
        self,
        filename,
    ):
        """Save optimization history to JSON."""
        self._require_result()

        filename = Path(filename)
        filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "search_method": self.search_method,
            "population_size": self.population_size,
            "n_generations": self.n_generations,
            "n_restarts": self.n_restarts,
            "perturbation_sizes": self.perturbation_sizes,
            "max_greedy_steps": self.max_greedy_steps,
            "random_seed": self.random_seed,
            "history": self.history,
        }

        with filename.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                _json_ready(payload),
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

        filename = Path(filename)
        filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        solution = {
            "search_method": self.search_method,
            "best_source": self.best_source,
            "candidate_ids": self.problem.candidate_ids,
            "states": self.best_states,
            "n_candidates": self.problem.n_candidates,
            "n_active": int(
                np.count_nonzero(
                    self.best_states
                )
            ),
            "n_positive": int(
                np.sum(
                    self.best_states == 1
                )
            ),
            "n_negative": int(
                np.sum(
                    self.best_states == -1
                )
            ),
            "n_absent": int(
                np.sum(
                    self.best_states == 0
                )
            ),
            "cost_components": self.best_components,
        }

        with filename.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                _json_ready(solution),
                f,
                indent=2,
            )

        return filename

    def statistics(self):
        """Return compact final optimizer statistics."""
        self._require_result()

        return {
            "search_method": self.search_method,
            "best_source": self.best_source,
            "population_size": self.population_size,
            "n_generations": self.n_generations,
            "n_restarts": self.n_restarts,
            "perturbation_sizes": self.perturbation_sizes,
            "max_greedy_steps": self.max_greedy_steps,
            "random_seed": self.random_seed,
            "best_cost": self.best_cost,
            "n_active": int(
                np.count_nonzero(
                    self.best_states
                )
            ),
            "n_positive": int(
                np.sum(
                    self.best_states == 1
                )
            ),
            "n_negative": int(
                np.sum(
                    self.best_states == -1
                )
            ),
            "cost_components": self.best_components,
        }

    # ------------------------------------------------------------------
    # Landscape diagnostic retained for compatibility
    # ------------------------------------------------------------------

    def diagnose_search_landscape(
        self,
        max_greedy_steps=50,
        verbose=True,
    ):
        """
        Diagnose zero-shim, best first move, and greedy local optimum.

        The first accepted greedy move from zero is necessarily the best
        single-magnet move because the local search exhaustively evaluates all
        one-coordinate alternatives before accepting a move.
        """
        self.validate()

        baseline_states = np.zeros(
            self.problem.n_candidates,
            dtype=int,
        )
        baseline_components = (
            self.problem.evaluate_states(
                baseline_states
            )
        )
        baseline_cost = float(
            baseline_components["total"]
        )

        greedy = self._greedy_local_search(
            start_states=baseline_states,
            max_steps=max_greedy_steps,
            verbose=verbose,
            label=None,
        )

        if greedy["history"]:
            first = greedy["history"][0]
            best_single_cost = float(
                first["new_cost"]
            )
            best_single_candidate = int(
                first["candidate_index"]
            )
            best_single_state = int(
                first["new_state"]
            )
        else:
            best_single_cost = baseline_cost
            best_single_candidate = None
            best_single_state = 0

        report = {
            "baseline_cost": baseline_cost,
            "best_single_cost": best_single_cost,
            "best_single_candidate_index": best_single_candidate,
            "best_single_candidate_id": (
                None
                if best_single_candidate is None
                else self.problem.candidate_ids[
                    best_single_candidate
                ]
            ),
            "best_single_state": best_single_state,
            "greedy_states": greedy["states"],
            "greedy_cost": greedy["cost"],
            "greedy_components": greedy["components"],
            "greedy_history": greedy["history"],
            "greedy_termination_reason": greedy[
                "termination_reason"
            ],
        }

        if verbose:
            print(
                "\n"
                + "=" * 60
            )
            print(
                "SHIM SEARCH-LANDSCAPE DIAGNOSTIC"
            )
            print(
                "=" * 60
            )
            print(
                f"Zero-shim cost: "
                f"{baseline_cost:.3f}"
            )
            print(
                f"Best single-magnet cost: "
                f"{best_single_cost:.3f}"
            )

            if best_single_candidate is not None:
                print(
                    f"Best single candidate: "
                    f"{self.problem.candidate_ids[best_single_candidate]}"
                )
                print(
                    f"Best single polarity: "
                    f"{best_single_state:+d}"
                )

            print(
                f"Greedy final cost: "
                f"{greedy['cost']:.3f}"
            )
            print(
                f"Greedy active magnets: "
                f"{np.count_nonzero(greedy['states'])}"
            )
            print(
                f"Greedy accepted steps: "
                f"{greedy['n_steps']}"
            )
            print(
                f"Greedy termination: "
                f"{greedy['termination_reason']}"
            )
            print(
                f"Greedy improvement: "
                f"{baseline_cost - greedy['cost']:.3f} "
                f"("
                f"{100.0 * (baseline_cost - greedy['cost']) / baseline_cost:.2f}%"
                f")"
            )
            print(
                "=" * 60
            )

        return report
