# Algorithms and numerical conventions

The package contains eleven methods. The E1, random-graph, and STAR configurations use the ten-method paper comparison; the online-mechanism configuration also includes EXP4MF (RW). `replication/algorithms/defaults.json` records the exact numerical defaults. One configuration is used across environments; the runner does not tune methods to each instance or choose instances from their regret.

All policies are distributions over **actions**, including methods whose regularizer uses mediator probabilities. Write $A=|\mathcal A|$, $Z=|\mathcal Z|$, $M\in\mathbb R^{Z\times A}$, $p\in\Delta_A$, and $q=Mp$. A round supplies the chosen action, observed mediator and loss. Learners receive no loss means, gaps, optimal action or future switching schedule.

| Display name | Stable identifier | Default settings |
|---|---|---|
| Tsallis-INF | `Tsallis-INF` | $c=0.7$; action RV estimator; cutoff $\kappa=16/0.7^2$ |
| CTsallis-Action | `CTsallis-Action` | $c=0.7$; mediator RV estimator; $\kappa=16$; no stabilizer |
| CTsallis-Mediator | `CTsallis-Context` | $c=0.7$; raw mediator-entropy objective; mediator RV; $\kappa=16$; no stabilizer |
| CTsallis-INF | `CTsallis-Jensen-selected` | $c=0.7$, $\rho=0.015$, $\alpha=2/3$; minimum regularizer; mediator RV; $\kappa=16$ |
| CTsallis-INF (no Hα) | `CTsallis-Jensen-rho0` | Same minimum regularizer and $c=0.7$, with exactly $\rho=0$; mediator RV; $\kappa=16$ |
| EXP4MF (IW) | `EXP4MF` | Original mediator importance-weighted estimator; `schedule="bobw"` |
| EXP4MF (RW) | `EXP4MF-RV` | Experimental RV estimator; same EXP4MF policy/rate schedule; `options.rv_cutoff=16` |
| PE | `PECUCB` | Phase elimination; random play order; design tolerance $10^{-8}$; maximum 10,000 design iterations; $\delta=1/\max(2,T)$ |
| CUCB | `CUCB` | Mediator reward UCB; $\delta=1/\max(2,T)^2$ |
| CTS | `CTS` | Independent mediator reward priors $\operatorname{Beta}(1,1)$ |
| CUCB2 | `CUCB2` | Coverage-based reward UCB; no adjustable coefficient in the default implementation |

Current display names and stable identifiers refer to the same implementations. `RW` is the figure label for the reduced-variance EXP4MF variant; its stored identifier retains `RV`.

## Tsallis objectives and estimators

The regularizers used by the implementation are

\[
H_A(p)=\sum_a\sqrt{p_a}-1,\qquad
R_M(p)=\sum_z\sqrt{(Mp)_z},\qquad
J_M(p)=R_M(p)-\sum_a p_a\sum_z\sqrt{M_{z,a}},
\]
\[
H_\alpha(p)=\frac{\sum_a p_a^\alpha-1}{\alpha(1-\alpha)}.
\]

Tsallis-INF minimizes $\langle\widehat L,p\rangle-c\sqrt t\,H_A(p)$, using its action-only estimates. CTsallis-Action uses the same action objective with estimates obtained from mediator feedback. CTsallis-Mediator minimizes $\langle\widehat L,p\rangle-c\sqrt t\,R_M(p)$. Additive constants in these objectives do not change a minimizer.

CTsallis-INF minimizes

\[
\langle\widehat L,p\rangle-\sqrt t\left[
c\min\{H_A(p),J_M(p)\}+\rho\Lambda_\alpha H_\alpha(p)\right],
\]
\[
D_\alpha=\frac{A^{1-\alpha}-1}{\alpha(1-\alpha)},\qquad
\Lambda_\alpha=\min\left\{\sqrt{A^\alpha/D_\alpha},\frac{\sqrt Z-1}{D_\alpha}\right\}.
\]

The code sets $\Lambda_\alpha=0$ for the trivial $A=1$ case. Its internal convention is `regularizer_scale=c/2` and `lambda_alpha=(2*rho/c)*Lambda_alpha`; their product is the effective $H_\alpha$ coefficient. The no-Hα variant removes that term exactly, retaining the minimum regularizer. A theorem requiring a positive stabilizer should not be attributed to the zero-stabilizer ablation without a separate argument.

**The Mediator comparison uses raw mediator entropy.** It is not the Jensen branch alone on a general mechanism: $R_M-J_M$ is an action-dependent affine term. The two objectives coincide up to an irrelevant constant when all columns have the same value of $\sum_z\sqrt{M_{z,a}}$. This distinction matters for heterogeneous random graphs.

For a coordinate distribution $v_t$, the RV estimator is

\[
B_t(i)=\tfrac12\mathbf1\{v_t(i)\ge\kappa/t\},\qquad
\widehat\ell_t(i)=B_t(i)+
\frac{\mathbf1\{I_t=i\}(\ell_t-B_t(i))}{v_t(i)}.
\]

Tsallis-INF uses $v_t=p_t$. The causal Tsallis methods use $v_t=q_t$ and lift mediator estimates through $M^\top$. Default thresholds are per mediator; baseline labels do not merge coordinates in the default `fine` mode. The code's `eta_scale=sqrt(cutoff)` controls this estimator threshold, independently of the main objective coefficient $c$. No artificial denominator or action-probability floor is inserted.

## Other methods

EXP4MF (IW) estimates action losses by $\widehat\ell_t(a)=M_{Z_t,a}\ell_t/q_t(Z_t)$ and applies exponential weights. The default BOBW schedule uses a mechanism capacity, or its recorded upper bound. The recognized symmetric channel uses its exact capacity; otherwise the implementation uses

\[
C\leq \min\{A-1,Z-1,\sum_z\max_a M_{z,a}-1\}
\]

as a capacity upper bound in the rate calculation. With that chosen bound denoted by $\bar C$, $\gamma=\sqrt{e\bar C(1+\log T)/(2\log A)}$, $\beta_1=\gamma$, $\eta_t=\min(1,1/\beta_t)$, and

\[
\beta_{t+1}=\beta_t+\frac{\gamma}{\sqrt{1+\sum_{s\leq t}H_{\rm Shannon}(p_s)/\log A}}.
\]

Trivial single-action/zero-capacity cases use rate one. EXP4MF (RW) replaces only the loss estimator by $M^\top B_t+M_{Z_t,\cdot}(\ell_t-B_t(Z_t))/q_t(Z_t)$. This is an explicitly labelled experimental variant, not the estimator covered by the cited EXP4MF theorem.

The four stochastic baselines internally use reward $1-\ell$. For bounded non-binary STAR losses, the frozen CTS implementation draws a Bernoulli reward with mean $1-\ell$ using its independent learner RNG before updating its Beta posterior; the environment and regret retain the unchanged raw loss. CUCB chooses the action maximizing the lifted mediator score $\widehat r_z+\sqrt{2\log(1/\delta)/\max(1,N_z)}$, without clipping. CTS samples independent mediator reward means from their Beta posteriors and maximizes their lifted score. PE uses the mechanism columns as linear features, phase-specific reward regression and a certified sparse design; it does not use the realized mediator in that regression.

CUCB2 uses $c_z=\min_aM_{z,a}$, $\zeta_a=\sum_zM_{z,a}/c_z$, and an action bonus $\zeta_a\sqrt{\log(Zs^2/2)/s}$, where $s=t-1$, after initially playing each action once. Reachable mediators must have strictly positive probability under **every** action. Incompatible sparse graphs are retained, with CUCB2 recorded as N/A; their support zeros are not smoothed to make this method applicable. See [REFERENCES.md](REFERENCES.md) for the corresponding algorithms and assumptions.

## Online mechanism estimates

The initial data use uniformly sampled actions followed by $Z\mid A=a\sim M_{\cdot,a}$; joint pairs are not uniform over all $A Z$ cells. A single 2,000-pair sample supplies both the nested 500-pair prefix and the 2,000-pair estimate. Calibration samples are shared across methods and the two loss regimes for each paired replicate.

With prior strength $\nu=1$, the pre-round estimate is

\[
\widehat M_t(z,a)=
\frac{N^{\rm initial}_{z,a}+N^{\rm online}_{<t,z,a}+\nu/Z}
{N^{\rm initial}_a+N^{\rm online}_{<t,a}+\nu}.
\]

The runner first completes the loss-estimator update with the pre-round model, then adds the observed pair and supplies the next model. Only naturally chosen online actions provide new observations; there is no extra uniform exploration. This declared count prior differs from an arbitrary floor on an importance denominator.

Online causal Tsallis methods retain the action-space history $\sum_{s<t}\widehat M_s^\top\widehat g_s$; they never recompute old losses using a later matrix. Current regularization uses the current estimate. EXP4MF variants refresh capacity and $\gamma$, retaining $\beta$, accumulated entropy/information and losses. CUCB/CTS retain counts/posteriors, and CUCB2 refreshes its coverage factors. PE fits the model continuously but adopts it only at phase boundaries, keeping each phase's features and regression consistent. No method resets its learning history at a loss switch.

These are specified plug-in robustness extensions. With estimated $M$, importance weighting is generally biased for true action losses, and adaptively neglected actions need not have accurately estimated columns. Fixed-known-$M$ or stationary guarantees are not asserted for these extensions. In particular, the stationary assumptions of the stochastic baselines do not survive merely because a common-baseline switch preserves action gaps.

## Numerical optimization and configuration

Tsallis-INF and CTsallis-Action solve a scalar KKT equation. The mediator and minimum objectives use warm-started working-set/Newton methods, exact reductions when available, and certified conic fallbacks through CVXPY/Clarabel. The minimum solver searches its scalar branch-mixture dual. The no-Hα solver explicitly permits action-simplex boundary optima; reachable mediator probabilities must remain positive.

Accepted mediator/minimum decisions have a checked global gap bound of at most $10^{-6}$ **for the normalized objective**. This is an objective certificate, not a claimed parameter-error tolerance. Independent tangent/dual certificates, fallback counts and numerical failures are recorded. Failed runs retain their valid prefixes and are not silently replaced by a different policy. PE's separate design optimization checks its leverage certificate.

To change a method, use the explicit `algorithm_overrides` configuration rather than editing numerical solvers. Valid Tsallis coefficients are finite $c>0$, finite cutoff $\kappa>0$ and $0<\alpha<1$. Native objectives/no-Hα require $\rho=0$; the positive-stabilizer CTsallis-INF requires $\rho>0$. The stored numerical cutoff is the actual threshold numerator. The `scaled` configuration convention derives $16/c^2$ when no numerical cutoff is explicitly supplied; `fixed` retains its declared numerator. Changing a cutoff convention or a coefficient does not itself establish a theorem for the modified method. $\alpha$ changes the stabilizer, not the native half-Tsallis objectives. EXP4MF (RW)'s active cutoff is `options.rv_cutoff`.

The public API is:

```python
from replication.algorithms import (
    default_specs, instantiate, applicability, resolved_parameters,
    Mechanism, DISPLAY_NAMES,
)

spec = default_specs()[0]             # independent specification dictionary
learner = instantiate(spec, M, T, rng, online=False)
p = learner.probabilities(t)          # t starts at 1
learner.update(t, action, mediator, loss, p)
settings = resolved_parameters(spec, learner)
```

`M` may be an array or `Mechanism(M, baseline_groups)`. With `online=True`, call `learner.update_mechanism(M_next)` after the completed round's loss update. `applicability(spec, M)` returns an incompatibility reason or `None`. All eleven stable identifiers and current display names are accepted by `instantiate`.
