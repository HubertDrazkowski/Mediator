"""Known column-stochastic mechanism; no losses or oracle information."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Mechanism:
    """Known column-stochastic matrix M[z,a].

    baseline_groups encodes labels that share a reduced-variance threshold.
    It is available to a learner, unlike the loss schedule.  It does not reveal
    losses, gaps, the optimum, or equality of unknown losses to other baselines.
    """

    M: np.ndarray
    baseline_groups: np.ndarray | None = None

    def __post_init__(self) -> None:
        matrix = np.array(self.M, dtype=float, copy=True)
        if matrix.ndim != 2 or min(matrix.shape) < 1:
            raise ValueError("M must be a nonempty Z-by-A matrix")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
            raise ValueError("M must contain finite nonnegative probabilities")
        if not np.allclose(matrix.sum(axis=0), 1, atol=1e-12, rtol=0):
            raise ValueError("Every mechanism column must sum to one")
        matrix.setflags(write=False)
        object.__setattr__(self, "M", matrix)
        if self.baseline_groups is None:
            groups = np.arange(matrix.shape[0], dtype=int)
        else:
            supplied = np.asarray(self.baseline_groups)
            if supplied.shape != (matrix.shape[0],) or not np.issubdtype(supplied.dtype, np.integer):
                raise ValueError("baseline_groups must contain one integer per context")
            if np.any(supplied < 0):
                raise ValueError("baseline group identifiers must be nonnegative")
            _, groups = np.unique(supplied, return_inverse=True)
        groups.setflags(write=False)
        object.__setattr__(self, "baseline_groups", groups)

    @property
    def n_actions(self) -> int:
        return int(self.M.shape[1])

    @property
    def n_contexts(self) -> int:
        return int(self.M.shape[0])

    def q_from_p(self, p: np.ndarray) -> np.ndarray:
        return self.M @ p

    def lift(self, g: np.ndarray) -> np.ndarray:
        return self.M.T @ g

