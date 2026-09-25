"""Reduced-variance EXP4MF estimator used for the explicitly labelled ablation."""
import numpy as np
from .base.exp4mf import EXP4MF

def mediator_rv_estimate(matrix, p, context, loss, t, cutoff=16.):
    """Context control variate lifted to action losses.

    B_z=.5*1{(M p)_z >= cutoff/t}; estimate=M.T B
    +M[z,:]*(loss-B_z)/(M p)_z. Conditional unbiasedness is with respect to
    the supplied mechanism. It is generally biased for true action losses
    when the supplied mechanism is an estimated, misspecified M_hat.
    No probability floors, clipping or baseline grouping are introduced.
    """
    matrix = np.asarray(matrix, dtype=float)
    p = np.asarray(p, dtype=float)
    if matrix.ndim != 2 or p.shape != (matrix.shape[1],):
        raise ValueError("matrix/p dimensions are inconsistent")
    if (not np.isfinite(p).all() or np.any(p < 0)
            or not np.isclose(p.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError("p must be the actual action sampling distribution")
    if int(t) != t or t < 1 or not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Require integer t>=1 and finite cutoff>0")
    if int(context) != context or not 0 <= context < matrix.shape[0]:
        raise ValueError("context is outside the mechanism")
    if not np.isfinite(loss) or not 0 <= loss <= 1:
        raise ValueError("loss must lie in [0, 1]")
    context = int(context)
    q = matrix @ p
    if q[context] <= 0:
        raise ValueError("an observed context has zero model sampling probability")
    baseline = .5 * (q >= float(cutoff) / int(t))
    estimate = matrix.T @ baseline
    residual = float(loss) - baseline[context]
    positive = matrix[context] > 0
    if residual != 0:
        estimate[positive] += (matrix[context, positive] / q[context]) * residual
    if not np.isfinite(estimate).all():
        raise FloatingPointError("EXP4MF-RV importance estimate exceeded numeric range")
    return estimate


class EXP4MFRV(EXP4MF):
    """Experimental RV estimator with EXP4MF's unchanged policy/rate rules."""
    name = "EXP4MF-RV"

    def __init__(self, mechanism, horizon, rng, **options):
        self.rv_cutoff = float(options.pop("rv_cutoff", 16.))
        if not np.isfinite(self.rv_cutoff) or self.rv_cutoff <= 0:
            raise ValueError("rv_cutoff must be finite and positive")
        super().__init__(mechanism, horizon, rng, **options)
        self.estimator = "rv"

    def update(self, t, action, context, loss, p):
        estimate = mediator_rv_estimate(self.M, p, context, loss, t, self.rv_cutoff)
        # A zero observation contributes no loss in the parent update. This
        # executes its *unchanged*, loss-independent entropy/information rate
        # bookkeeping and validation, then installs the RV estimate instead.
        super().update(t, action, context, 0., p)
        self.cumulative_loss += estimate
        self.cumulative_loss -= self.cumulative_loss.min()
        if not np.isfinite(self.cumulative_loss).all():
            raise FloatingPointError("EXP4MF-RV cumulative loss exceeded numeric range")

    def diagnostics(self):
        return {**super().diagnostics(), "estimator": "rv",
                "rv_cutoff": self.rv_cutoff, "baseline_amplitude": .5,
                "baseline_probability_space": "context",
                "published_guarantee_applies": False,
                "experimental_extension": True}

