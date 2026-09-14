"""
Precomputed magnetic-field basis for passive shimming.

The expensive Magpylib field calculation is performed exactly once for each
candidate shim magnet. Subsequent shim configurations are evaluated by linear
superposition:

    B_shim = basis_T @ states

where each state is typically one of:

    -1  present, negative polarity
     0  absent
    +1  present, positive polarity

The basis matrix shape is:

    (n_field_points, n_candidates)

Each column contains the Bz field, in tesla, produced by one candidate magnet
with its nominal positive polarization.

Magpylib is used only during basis construction and optional validation.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import magpylib as magpy
import numpy as np

from utils import get_magnetic_field


class ShimBasis:
    """
    Precompute and manage the field contribution of every shim candidate.

    Parameters
    ----------
    field_map : FieldMap-like object
        Must provide:
            ``B_T``
            ``sensors``
            ``positions_m``

    candidate_tray : CandidateTray-like object
        Must provide:
            ``candidates``
            ``candidate_ids``

    axis : int, optional
        Magnetic-field component returned by ``get_magnetic_field``.
        ``axis=2`` corresponds to Bz and is the passive-shimming default.

    dtype : numpy dtype, optional
        Storage dtype for the basis matrix. Default ``np.float64``.

    debug : bool, optional
        Print detailed diagnostics when True.

    Attributes
    ----------
    basis_T : numpy.ndarray
        Shape ``(n_field_points, n_candidates)``.

    candidate_ids : list[str]
        Stable column identifiers matching the candidate-tray ordering.

    is_computed : bool
        True after :meth:`compute`.
    """

    def __init__(
        self,
        field_map,
        candidate_tray,
        axis=2,
        dtype=np.float64,
        debug=False,
    ):
        self.field_map = field_map
        self.candidate_tray = candidate_tray
        self.axis = int(axis)
        self.dtype = np.dtype(dtype)
        self.debug = bool(debug)

        self.candidates = list(
            candidate_tray.candidates
        )

        self.candidate_ids = list(
            candidate_tray.candidate_ids
        )

        self.basis_T = None
        self.is_computed = False
        self.validation_report = None

        self._validate_inputs()

    def _validate_inputs(self):
        if self.axis not in (0, 1, 2):
            raise ValueError(
                "axis must be 0, 1, or 2."
            )

        if not hasattr(
            self.field_map,
            "sensors",
        ):
            raise TypeError(
                "field_map must provide a Magpylib sensors attribute."
            )

        if not hasattr(
            self.field_map,
            "B_T",
        ):
            raise TypeError(
                "field_map must provide B_T."
            )

        if len(
            self.candidates
        ) == 0:
            raise ValueError(
                "candidate_tray contains no candidates."
            )

        if len(
            self.candidate_ids
        ) != len(
            self.candidates
        ):
            raise ValueError(
                "candidate_ids and candidates must have equal length."
            )

        if len(
            set(
                self.candidate_ids
            )
        ) != len(
            self.candidate_ids
        ):
            raise ValueError(
                "candidate_ids must be unique."
            )

    @property
    def n_points(self):
        """Number of measured field samples."""
        return int(
            len(
                self.field_map.B_T
            )
        )

    @property
    def n_candidates(self):
        """Number of shim candidates."""
        return int(
            len(
                self.candidates
            )
        )

    def compute(self):
        """
        Compute the positive-polarity Bz field for each candidate magnet.

        Returns
        -------
        ShimBasis
            ``self`` for method chaining.
        """
        basis = np.empty(
            (
                self.n_points,
                self.n_candidates,
            ),
            dtype=self.dtype,
        )

        if self.debug:
            print(
                "\nComputing shim basis"
            )
            print(
                f"  Field points: {self.n_points}"
            )
            print(
                f"  Candidates: {self.n_candidates}"
            )

        for column, magnet in enumerate(
            self.candidates
        ):
            field = get_magnetic_field(
                magnets=magnet,
                sensors=self.field_map.sensors,
                axis=self.axis,
            )

            field = np.asarray(
                field,
                dtype=self.dtype,
            ).reshape(
                -1
            )

            if len(
                field
            ) != self.n_points:
                raise RuntimeError(
                    f"Candidate {column} ({self.candidate_ids[column]}) "
                    f"returned {len(field)} samples; expected "
                    f"{self.n_points}."
                )

            if not np.all(
                np.isfinite(
                    field
                )
            ):
                raise ValueError(
                    f"Candidate {column} ({self.candidate_ids[column]}) "
                    "produced non-finite field values."
                )

            basis[
                :,
                column,
            ] = field

            if (
                self.debug
                and (
                    column < 5
                    or (column + 1) % 25 == 0
                    or column == self.n_candidates - 1
                )
            ):
                print(
                    f"  Column {column+1}/{self.n_candidates}: "
                    f"{self.candidate_ids[column]}"
                )

        self.basis_T = basis
        self.is_computed = True

        if self.debug:
            stats = self.statistics()
            print(
                f"  Matrix shape: {stats['matrix_shape']}"
            )
            print(
                f"  Memory: {stats['memory_MB']:.3f} MB"
            )

        return self

    def _require_computed(self):
        if (
            not self.is_computed
            or self.basis_T is None
        ):
            raise RuntimeError(
                "ShimBasis.compute() must be called first."
            )

    def field_from_states(
        self,
        states,
    ):
        """
        Compute shim Bz by matrix superposition.

        Parameters
        ----------
        states : array-like, shape (n_candidates,)
            Shim states. Values need not be restricted here, although the
            optimizer normally uses {-1, 0, +1}.

        Returns
        -------
        numpy.ndarray
            Shim field Bz in tesla at all field-map positions.
        """
        self._require_computed()

        states = np.asarray(
            states,
            dtype=float,
        ).reshape(
            -1
        )

        if len(
            states
        ) != self.n_candidates:
            raise ValueError(
                f"states has length {len(states)}; expected "
                f"{self.n_candidates}."
            )

        return (
            self.basis_T
            @ states
        )

    def total_field_from_states(
        self,
        states,
    ):
        """
        Return measured field plus shim field for one state vector.
        """
        return (
            np.asarray(
                self.field_map.B_T,
                dtype=float,
            )
            + self.field_from_states(
                states
            )
        )

    def _direct_field_from_states(
        self,
        states,
    ):
        """
        Compute one shim configuration directly through Magpylib.

        Used only for validation; never used by the optimizer.

        Fresh Cuboid objects are constructed from candidate geometry so that
        they do not retain parent Collection relationships from CandidateTray.
        """
        states = np.asarray(
            states,
            dtype=int,
        ).reshape(-1)

        if len(states) != self.n_candidates:
            raise ValueError(
                "Invalid state-vector length."
            )

        selected = magpy.Collection()

        for state, candidate in zip(
            states,
            self.candidates,
        ):
            state = int(state)

            if state == 0:
                continue

            magnet = magpy.magnet.Cuboid(
                dimension=np.asarray(
                    candidate.dimension,
                    dtype=float,
                ).copy(),
                position=np.asarray(
                    candidate.position,
                    dtype=float,
                ).copy(),
                polarization=(
                    np.asarray(
                        candidate.polarization,
                        dtype=float,
                    )
                    * state
                ),
                orientation=copy.deepcopy(
                    candidate.orientation
                ),
            )

            selected.add(
                magnet
            )

        if np.count_nonzero(states) == 0:
            return np.zeros(
                self.n_points,
                dtype=float,
            )

        direct = get_magnetic_field(
            magnets=selected,
            sensors=self.field_map.sensors,
            axis=self.axis,
        )

        return np.asarray(
            direct,
            dtype=float,
        ).reshape(-1)

    def validate(
        self,
        n_tests=5,
        atol_T=1e-10,
        rtol=1e-8,
        random_seed=42,
        allowed_states=(-1, 0, 1),
        raise_on_failure=True,
    ):
        """
        Validate matrix superposition against direct Magpylib evaluation.

        Random state vectors are generated and evaluated by both methods.

        Returns
        -------
        dict
            Validation statistics.
        """
        self._require_computed()

        if n_tests <= 0:
            raise ValueError(
                "n_tests must be positive."
            )

        allowed_states = np.asarray(
            allowed_states,
            dtype=int,
        )

        rng = np.random.default_rng(
            random_seed
        )

        tests = []
        all_passed = True

        for index in range(
            int(
                n_tests
            )
        ):
            states = rng.choice(
                allowed_states,
                size=self.n_candidates,
                replace=True,
            )

            basis_field = (
                self.field_from_states(
                    states
                )
            )

            direct_field = (
                self._direct_field_from_states(
                    states
                )
            )

            error = (
                basis_field
                - direct_field
            )

            max_abs_error_T = float(
                np.max(
                    np.abs(
                        error
                    )
                )
            )

            rms_error_T = float(
                np.sqrt(
                    np.mean(
                        error ** 2
                    )
                )
            )

            direct_scale = max(
                float(
                    np.max(
                        np.abs(
                            direct_field
                        )
                    )
                ),
                np.finfo(
                    float
                ).eps,
            )

            max_relative_error = (
                max_abs_error_T
                / direct_scale
            )

            passed = bool(
                np.allclose(
                    basis_field,
                    direct_field,
                    atol=float(
                        atol_T
                    ),
                    rtol=float(
                        rtol
                    ),
                )
            )

            all_passed = (
                all_passed
                and passed
            )

            tests.append(
                {
                    "test":
                        index + 1,
                    "n_active":
                        int(
                            np.count_nonzero(
                                states
                            )
                        ),
                    "max_abs_error_T":
                        max_abs_error_T,
                    "rms_error_T":
                        rms_error_T,
                    "max_relative_error":
                        max_relative_error,
                    "passed":
                        passed,
                }
            )

            if self.debug:
                print(
                    f"  Validation {index+1}/{n_tests}: "
                    f"active={np.count_nonzero(states)}, "
                    f"max_abs={max_abs_error_T:.3e} T, "
                    f"passed={passed}"
                )

        self.validation_report = {
            "all_passed":
                bool(
                    all_passed
                ),
            "n_tests":
                int(
                    n_tests
                ),
            "atol_T":
                float(
                    atol_T
                ),
            "rtol":
                float(
                    rtol
                ),
            "random_seed":
                int(
                    random_seed
                ),
            "tests":
                tests,
        }

        if (
            not all_passed
            and raise_on_failure
        ):
            raise RuntimeError(
                "ShimBasis validation failed. "
                "Do not continue to optimization."
            )

        return self.validation_report

    def statistics(self):
        """Return matrix dimensions and numerical diagnostics."""
        self._require_computed()

        return {
            "matrix_shape":
                list(
                    self.basis_T.shape
                ),
            "dtype":
                str(
                    self.basis_T.dtype
                ),
            "memory_bytes":
                int(
                    self.basis_T.nbytes
                ),
            "memory_MB":
                float(
                    self.basis_T.nbytes
                    / (
                        1024 ** 2
                    )
                ),
            "n_field_points":
                self.n_points,
            "n_candidates":
                self.n_candidates,
            "axis":
                self.axis,
            "basis_min_T":
                float(
                    np.min(
                        self.basis_T
                    )
                ),
            "basis_max_T":
                float(
                    np.max(
                        self.basis_T
                    )
                ),
            "basis_rms_T":
                float(
                    np.sqrt(
                        np.mean(
                            self.basis_T ** 2
                        )
                    )
                ),
            "validation":
                self.validation_report,
        }

    def save(
        self,
        filename,
    ):
        """
        Save basis matrix, candidate IDs, and field positions to compressed NPZ.
        """
        self._require_computed()

        filename = Path(
            filename
        )

        filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        positions_m = getattr(
            self.field_map,
            "positions_m",
            None,
        )

        np.savez_compressed(
            filename,
            basis_T=
                self.basis_T,
            candidate_ids=
                np.asarray(
                    self.candidate_ids,
                    dtype=str,
                ),
            field_positions_m=
                positions_m,
            axis=
                np.asarray(
                    self.axis,
                    dtype=int,
                ),
        )

        return filename

    @classmethod
    def load(
        cls,
        filename,
        field_map,
        candidate_tray,
        debug=False,
    ):
        """
        Load a previously saved basis and verify candidate-column ordering.
        """
        filename = Path(
            filename
        )

        data = np.load(
            filename,
            allow_pickle=False,
        )

        obj = cls(
            field_map=
                field_map,
            candidate_tray=
                candidate_tray,
            axis=int(
                np.asarray(
                    data[
                        "axis"
                    ]
                )
            ),
            debug=
                debug,
        )

        stored_ids = [
            str(
                item
            )
            for item in data[
                "candidate_ids"
            ]
        ]

        if stored_ids != obj.candidate_ids:
            raise ValueError(
                "Saved basis candidate IDs do not match current "
                "CandidateTray ordering."
            )

        obj.basis_T = np.asarray(
            data[
                "basis_T"
            ],
            dtype=float,
        )

        expected_shape = (
            obj.n_points,
            obj.n_candidates,
        )

        if (
            obj.basis_T.shape
            != expected_shape
        ):
            raise ValueError(
                f"Saved basis shape {obj.basis_T.shape} does not match "
                f"expected {expected_shape}."
            )

        obj.is_computed = True

        return obj
