# Paper-to-code alignment

This package reproduces the implementations and environment definitions used in
the completed studies. No benchmark outcomes were used to change method defaults
while packaging them. No full experiments are launched when opening this folder,
importing a module, or installing requirements.

| Paper figure | Configuration | Plot | Population |
|---|---|---|---|
| Figure 2: main synthetic comparison | `e1.json` | `E1_focus` | A80/Z30, stationary and deterministic switching,30 seeds |
| Figure 3: extended nonstationarity | `e1.json` | `E1_all` | Three cardinality pairs and three regimes,30 seeds |
| Figure 4: adaptation | `transpose.json` | `Transpose_boxplots_t50000` | Six complete-graph settings,30 seeds,three methods |
| Figure 5: random graphs | `random.json` | `Random_stationary_switching` |70 stationary graphs and 30 preselected switching graphs,one seed/method |
| Figure 6: mechanism estimation | `online.json` | `Online_mechanism` | True M / online500 / online2,000;30 seeds per box |
| Figure 7: STAR | `star.json` | `STAR_main_log_t20000` | Five seeds,ten methods,4,298-record empirical cohort |

The numerical horizon is explicit in every configuration. Default methods are
frozen, with source hashes, before a run starts. The package does not launch a
hyperparameter search. Their prior calibration is described as preliminary
empirical calibration followed by fixed settings across reported environments;
this is not a claim of performance-blind selection or an independent held-out
validation set. See ALGORITHMS.md for exact objectives and estimator cutoffs.

## Clarifications needed when aligning manuscript and code

- **E1 denominator.** Executed code uses `epsilon/Z` for every matrix entry plus
  `1-epsilon` on the dominant mediator. The draft's `0.3/|Z-1|` does not give
  column sums of one when added to0.7 on the dominant entry. The replicated
  mechanism is `0.7*indicator + 0.3/Z`, and positive action gaps are0.07.
- **Random sample size.** The first50 stationary graphs were extended by 20,
  generated as a separate batch. The main stationary panel therefore has70
  graphs; the switching panel has30 selected without reference to regret.
  Describing both as 50 graphs omits the extension and switching subset. The
  additional matched30-versus-30 plot permits a paired regime comparison.
- **Initial mechanism data.** An initial observation samples A uniformly and
  then Z conditionally from M[:,A]. It is not uniform over all pairs (a,z).
- **Online seed count.** This requested code-only package uses30 seeds uniformly
  and omits the later 40-seed A30/Z80 preset. A paper figure using that extension
  will not have exactly the same A30/Z80 box summaries as this30-seed default.
- **Mediator regularizer.** The executed CTsallis-Mediator objective uses raw
  half-Tsallis entropy of Mp. INF uses the calibrated Jensen term inside its
  minimum. These differ by an action-dependent affine term on heterogeneous
  columns; they coincide up to a constant when column entropies are equal.
  The package preserves this implemented distinction instead of changing old
  experiments to fit a different interpretation of an ablation.
- **STAR assumptions.** STAR is an empirical treatment-arm replay benchmark
  under potentially imperfect mediation, not a validation that A and loss are
  conditionally independent given Z. Whole-record sampling retains that
  imperfection. Regret uses raw arm means, not projected mediator means.
  Missing early scores form a mediator category; missing later outcomes are
  excluded. Bounded rank losses are non-binary, on finite empirical support.
- **EXP4MF variants.** IW is the main baseline. RW is included only in the
  historical all-method online configuration and is explicitly experimental.
  It is absent from the E1, random, transpose, and STAR paper comparisons.

The package contains no generated inputs or existing results. Running a selected
configuration creates local inputs and result files for reproducibility. STAR
has a separate public-data acquisition step. Earlier unrelated studies, hidden
mediator variants, tuning sweeps, and rejected designs are excluded.
