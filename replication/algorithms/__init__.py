"""Anonymous, standalone algorithm API for the paper replication.

Known-M and online plug-in learners share the decision/update interface.
For online estimation, call update_mechanism only after the completed loss
update. The runner owns pair counts; algorithms receive only the fitted M.
"""
from .mechanism import Mechanism
from .names import DISPLAY_NAMES
from . import registry
from .base import IncompatibleMechanismError, NumericalSolverError


def default_specs():
    return registry.algorithm_specs()


def _mechanism(matrix):
    return matrix if isinstance(matrix, Mechanism) else Mechanism(matrix)


def applicability(specification, M):
    return registry.applicability(specification, _mechanism(M))


def instantiate(specification, M, horizon, rng, online=False):
    specification = registry.resolve_spec(specification)
    mechanism = _mechanism(M)
    if online:
        from . import online as core_online
        from . import online_baselines
        module = core_online if specification["id"] in core_online.ALGORITHM_IDS else online_baselines
        learner = module.instantiate(specification, mechanism, horizon, rng)
    else:
        learner = registry.instantiate(specification, mechanism, horizon, rng)
    learner._replication_online = bool(online)
    return learner


def resolved_parameters(specification, learner, online=None):
    specification = registry.resolve_spec(specification)
    if online is None:
        online = getattr(learner, "_replication_online", False)
    if online:
        from . import online as core_online
        from . import online_baselines
        module = core_online if specification["id"] in core_online.ALGORITHM_IDS else online_baselines
        return module.resolved_parameters(specification, learner)
    return registry.resolved_parameters(specification, learner)


__all__ = ["default_specs", "instantiate", "applicability", "resolved_parameters",
           "Mechanism", "DISPLAY_NAMES", "IncompatibleMechanismError", "NumericalSolverError"]
