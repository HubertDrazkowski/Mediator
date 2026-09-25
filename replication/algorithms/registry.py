"""Fixed paper configurations and known-mechanism learner construction."""
from copy import deepcopy
import json
import math
from numbers import Real
from pathlib import Path
import numpy as np

from .base import factory as published_factory, IncompatibleMechanismError, NumericalSolverError
from .calibrated import instantiate as calibrated_factory
from .context_refinement import wrap_context_learner
from .exp4_rw import EXP4MFRV
from .names import DISPLAY_NAMES

CORE = ("Tsallis-INF", "CTsallis-Action", "CTsallis-Context",
        "CTsallis-Jensen-selected", "CTsallis-Jensen-rho0")
NO_ALPHA_ID = "CTsallis-Jensen-rho0"


def algorithm_specs():
    """Independent copies of the fixed configurations; legacy IDs are stable."""
    return json.loads(Path(__file__).with_name("defaults.json").read_text(encoding="utf-8"))


default_specs = specs = algorithm_specs


def resolve_spec(specification):
    if isinstance(specification, str):
        identifier = {name: identifier for identifier, name in DISPLAY_NAMES.items()}.get(
            specification, specification)
        for candidate in algorithm_specs():
            if candidate["id"] == identifier:
                specification = candidate
                break
        else:
            raise ValueError(f"Unknown algorithm {identifier!r}")
    else:
        specification = deepcopy(dict(specification))
    if specification.get("id") not in DISPLAY_NAMES:
        raise ValueError("Unknown algorithm identifier")
    identifier = specification["id"]
    if identifier in CORE:
        for field in ("c", "rho", "cutoff", "alpha"):
            value = specification.get(field)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{field} must be a finite real number")
        if specification["c"] <= 0 or specification["cutoff"] <= 0:
            raise ValueError("Require c>0 and cutoff>0")
        if not 0 < specification["alpha"] < 1:
            raise ValueError("Require 0<alpha<1")
        expected_method = "CTsallis-Jensen" if identifier == "CTsallis-Jensen-selected" else identifier
        if specification.get("method") != expected_method:
            raise ValueError("The algorithm identifier and objective method disagree")
        if identifier == "CTsallis-Jensen-selected":
            if specification["rho"] <= 0:
                raise ValueError("CTsallis-INF requires rho>0; use the explicit no-H-alpha ablation for zero")
        elif specification["rho"] != 0:
            raise ValueError("Native objectives and the no-H-alpha ablation require rho=0")
        if specification.get("cutoff_rule") not in ("fixed", "scaled"):
            raise ValueError("cutoff_rule must be fixed or scaled")
    return specification


def applicability(specification, mechanism):
    specification = resolve_spec(specification)
    if specification["id"] != "CUCB2":
        return None
    matrix = np.asarray(mechanism.M, dtype=float)
    reachable = np.any(matrix > 0, axis=1)
    unsupported = np.flatnonzero(reachable & (matrix.min(axis=1) <= 0))
    if unsupported.size:
        return (
            "CUCB2 requires identical nonzero context support under every action; "
            f"{unsupported.size} reachable contexts have zero probability under at "
            "least one action. Nair et al. Section 5 is inapplicable to this "
            "mechanism. Keep the instance and other algorithms; no support "
            "floor or replacement algorithm is used."
        )
    return None


def instantiate(specification, mechanism, horizon, rng):
    specification = resolve_spec(specification)
    reason = applicability(specification, mechanism)
    if reason is not None:
        raise IncompatibleMechanismError(reason)
    if specification["id"] == "EXP4MF-RV":
        return EXP4MFRV(mechanism, horizon, rng, **deepcopy(specification["options"]))
    if specification["backend"] == "calibrated_tsallis":
        learner = calibrated_factory(specification, mechanism, horizon, rng)
        return learner if specification["id"] == NO_ALPHA_ID else wrap_context_learner(learner)
    if specification["backend"] == "published":
        return published_factory(specification["method"], mechanism, horizon, rng,
                                 deepcopy(specification.get("options", {})))
    raise ValueError("Unknown learner backend")


def resolved_parameters(specification, learner):
    specification = resolve_spec(specification)
    result = dict(id=specification["id"], method=specification["method"],
        label=specification["label"], parameter_provenance=specification["parameter_provenance"],
        horizon=int(learner.horizon), reset_at_switch=False)
    if specification["backend"] == "calibrated_tsallis":
        result.update(c=specification["c"], rho=specification["rho"],
            alpha=specification["alpha"], cutoff=specification["cutoff"],
            cutoff_rule=specification["cutoff_rule"], estimator=learner.estimator,
            eta_scale=float(learner.eta_scale),
            baseline_probability_space="action" if specification["method"] == "Tsallis-INF" else "context")
        if specification["method"] == "CTsallis-Jensen":
            result.update(regularizer_scale=float(learner.regularizer_scale),
                backend_lambda_alpha=float(learner.lambda_alpha),
                effective_H_alpha_coefficient=float(learner.regularizer_scale * learner.lambda_alpha),
                normalized_objective_gap_tolerance=float(learner.fast_gap_tolerance))
        elif specification["id"] == NO_ALPHA_ID:
            if learner.lambda_alpha != 0:
                raise ValueError("Nonzero H_alpha coefficient in the no-alpha ablation")
            result.update(alpha_term_retained=False, regularizer_scale=float(learner.regularizer_scale),
                backend_lambda_alpha=0., effective_H_alpha_coefficient=0.,
                normalized_objective_gap_tolerance=float(learner.fast_gap_tolerance),
                noalpha_solver="replication.algorithms.noalpha_solver.FastNoAlphaJensen",
                probability_floor=None)
    else:
        result.update(options=deepcopy(specification["options"]), source=specification["source"])
        for key in ("confidence_delta", "prior_alpha", "prior_beta", "play_order",
                    "design_tolerance", "design_max_iterations", "capacity",
                    "capacity_is_exact", "capacity_source", "schedule"):
            if hasattr(learner, key):
                result[key] = getattr(learner, key)
        if specification["id"] == "EXP4MF-RV":
            result.update(estimator="rv", cutoff=float(learner.rv_cutoff), cutoff_rule="fixed",
                baseline_amplitude=.5, baseline_probability_space="context",
                published_guarantee_applies=False, experimental_extension=True)
    return result
