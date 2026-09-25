"""Public, observation-only learner registry."""
from __future__ import annotations

from .tsallis import (
    CTsallisAction, CTsallisContext, CTsallisJensen, CTsallisJensenNoAlpha,
    CTsallisJensenOnly, NumericalSolverError, TsallisINF,
)
from .baselines import CUCB1, CTS, CUCB2, PECUCB, IncompatibleMechanismError
from .exp4mf import EXP4MF

ALGORITHM_CLASSES = {
    cls.name: cls for cls in (
        TsallisINF, CTsallisAction, CTsallisContext, CTsallisJensen,
        CTsallisJensenNoAlpha, EXP4MF, PECUCB, CUCB1, CTS, CUCB2,
        CTsallisJensenOnly,
    )
}
ALIASES = {
    "CTsallis-Jense": "CTsallis-Jensen-NoAlpha",
    "CTsallis-Jensen-Only": "CTsallis-JensenOnly",
    "PhaseElimination CUCB": "PE-CUCB",
    "PhaseElimination-CUCB": "PE-CUCB",
}


def factory(name, mechanism, horizon, rng, options=None):
    """Construct one learner; display labels belong to runner configuration.

    Unknown learner names/options fail explicitly. The supplied mechanism must
    contain only the known M and optional baseline labels, never oracle losses.
    """
    canonical = ALIASES.get(name, name)
    if canonical not in ALGORITHM_CLASSES:
        raise ValueError(f"Unknown algorithm {name!r}; choose {sorted(ALGORITHM_CLASSES)}")
    return ALGORITHM_CLASSES[canonical](mechanism, horizon, rng, **(options or {}))


__all__ = ["factory", "ALGORITHM_CLASSES", "IncompatibleMechanismError", "NumericalSolverError"]
