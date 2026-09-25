# Causal Tsallis: paper replication

Anonymous, standalone code for the experiments in the accompanying paper.
This update includes the synthetic comparisons and the Project STAR benchmark.
No datasets, saved results, author metadata, machine paths, or experiment history
are distributed. Synthetic inputs are regenerated from declared seeds; STAR is
obtained separately from its public source.

## Start in PyCharm

1. Open this folder as the project. Use Python 3.12 with a virtual environment.
2. Run `python -m pip install -r requirements.txt` in the project terminal.
3. Open `run_experiments.py`. Set `EXPERIMENT` to one name from the table below.
4. Edit that experiment's JSON file in `configs/` to select methods, seeds,
   dimensions, horizon, parameters, workers, and relative output folder.
5. Run `run_experiments.py`. Only that configuration runs.

The paths are relative to this package, even if the terminal has another working
directory. Set `SMOKE = True` for a separate 200-round installation check; smoke
results are not paper results. Preparing STAR requires its source file first;
see [STAR.md](STAR.md).

## Paper figures and configurations

| Config / `EXPERIMENT` | Design | Replicates | Horizon | Methods |
|---|---|---|---|---|
| `e1.json` / `e1` | Three dimension pairs; stationary, deterministic switching, random piecewise baseline | 30 per dimension and regime | 30,000 | 10 |
| `transpose.json` / `transpose` | Complete-graph incidence, two orientations, n=8,12,16 | 30 per setting | 50,000 | 3 CTsallis variants |
| `random.json` / `random` | Original 50 plus 20 additional stationary graphs; deterministic switching on a fixed subset of 30 | One trajectory per method and graph/regime | 30,000 | 10 |
| `online.json` / `online` | E1 stationary/deterministic switching; true M, online Mhat initialized with 500 or 2,000 pairs | 30 per condition | 30,000 | 11 |
| `star.json` / `star` | Empirical STAR, observed post-treatment mediator | Five paired seeds | 20,000 | 10 |

Figures 2 and 3 come from `e1`: `E1_focus` selects A=80, Z=30 and stationary versus
deterministic switching; `E1_all` includes three regimes and three cardinality
pairs. The transpose plot is Figure 4, the stationary/switching random comparison
is Figure 5, the online comparison is Figure 6, and STAR's log-scale plot is
Figure 7. Additional outputs include an aggregate stationary random plot and a
matched 30-versus-30 random comparison, so populations can be compared directly.

E1, random, and STAR use Tsallis-INF, CTsallis-Action, CTsallis-Mediator,
CTsallis-INF, CTsallis-INF (no Hα), EXP4MF (IW), PE, CUCB, CTS, and CUCB2.
The online figure retains its historical EXP4MF (RW) comparison as an eleventh
method. Transpose uses only Action, Mediator, and INF. Removing a name from
`algorithms` excludes it; no hyperparameter is retuned by the runner.
CUCB2 receives N/A when its common-support assumption fails, with the graph
and all other methods retained.

The online default intentionally uses 30 seeds everywhere. It has no separate
40-seed A=30, Z=80 preset. A figure incorporating that later ten-seed extension
will have different boxes from this requested 30-seed replication. See
[PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) for this and manuscript clarifications.

## Run, resume, and replot

```sh
# Freeze the design, without running learners.
python run_experiments.py --config configs/e1.json --action prepare

# Choose ONE experiment to run. These are alternatives, not a sweep script.
python run_experiments.py --config configs/e1.json --jobs 4
python run_experiments.py --config configs/transpose.json --jobs 4
python run_experiments.py --config configs/random.json --jobs 4
python run_experiments.py --config configs/online.json --jobs 4
python run_experiments.py --config configs/star.json --jobs 4

# Regenerate figures from local results; no learners are run.
python run_experiments.py --config configs/random.json --action plot

# Short installation check or a bounded number of unfinished tasks.
python run_experiments.py --config configs/transpose.json --smoke
python run_experiments.py --config configs/e1.json --max-tasks 10
python -m pytest -q
```

Repeat the same run command to resume an experiment. Completed, failed, and N/A
cells are terminal and are never silently rerun. A stopped trajectory replays
its saved seed and verifies its prefix before continuing. Numerical failures
retain valid checkpoints and appear in CSVs; failed checkpoints are marked with
an asterisk. No replacement runs or environment selection based on regret occur.

Create a `STOP` file inside the chosen output folder to stop dispatching work;
workers save their prefixes and stop at checks every 250 rounds. Remove that
file deliberately to resume. A hard-killed process can leave `RUNNING.lock`;
confirm that its workers have exited before deleting a stale lock. There is no
background scheduler or automatic launch of other experiments.

## Change parameters

Shared fields: `horizon`, `checkpoints`, `seeds`, `algorithms`, `jobs`, `output`,
`save_full_trajectories`, and `policy_snapshot_interval`.

- E1/online: `dimensions` contains **[A,Z]** pairs. Choose one epsilon per config.
  `modes` selects `stationary`, `baseline_switching`, and (E1 only)
  `random_baseline_piecewise`. `e1` specifies gap, initial segment length, and
  geometric growth. Random baseline draws are independent across trajectory
  seeds and shared across methods and dimensions for a seed.
- Transpose: `n_values`, `orientations`, `baseline`, `minimum_action_gap`, and
  `environment_seed` in `transpose`.
- Random: original `seed` and `n_graphs`, plus `extension_seed` and
  `extension_n_graphs`, preserve the two separate samples. Setting the extension
  size to zero omits it. Set `switching_count` to zero for stationary only.
  `dimensions`, `densities`, `structures`, `kappas`, `lambdas`, and
  `losses_per_graph` control generation; `selection_seed`, `L0`, and `gamma`
  control the switching subset/schedule. Changing a batch size changes its
  balanced assignments: a single 70-graph batch is not the paper's 50+20 sample.
- Online: `initial_pairs` and total `prior_strength` per action. Initial actions
  are uniform; mediators are drawn conditionally from true M, not uniformly
  over the Cartesian product.
- STAR: the relative `source_path` in `star`; source/cohort conventions are fixed
  to the reported benchmark. See [STAR.md](STAR.md).
- Plotting: `plot.checkpoint`, `font_size`, `show_headers`, and `formats`
  (`png`, `pdf`, `svg`). The log-scale STAR plot keeps all positive values;
  missing values are never represented as zero.

Use a new `output`, e.g. `results/random_small`, when changing the scientific
configuration. Frozen manifests reject parameter or implementation changes.
Worker count and presentation choices can change. Algorithms are unchanged
from the executed studies; their defaults, objectives, estimators, and numerical
certificates are specified in [ALGORITHMS.md](ALGORITHMS.md).

Explicit overrides are accepted, for example:

```json
"algorithm_overrides": {
  "CTsallis-INF": {"c": 0.7, "rho": 0.015, "cutoff": 16.0, "cutoff_rule": "fixed"}
}
```

Changing these values defines a new experiment. The fixed paper settings are
c=0.7 for the CTsallis family, rho=0.015 and alpha=2/3 for INF, and cutoff16/t.
The no-Hα ablation has rho=0. Action and Mediator have no stabilizer. Classical
Tsallis-INF uses its separate action RV cutoff16/(0.7²t).

## Generated outputs

```text
results/<experiment>/
  manifest.json          frozen parameters, source hashes, tasks, seed rules
  instances/             generated mechanisms/schedules and evaluator diagnostics
  calibration/           generated initial mechanism samples, online study only
  raw/                   one atomic record per trajectory, including failures/N/A
  estimates/             online count/model diagnostics
  traces/                per-round arrays if save_full_trajectories=true
  results.csv            checkpoints for every planned cell
  summary.csv            per-setting means, medians, quartiles and sample counts
  plotting_data.csv      exact values and status used in each panel
  panel_summary.csv      per-panel availability and failure counts
  plots/                 paper layouts in requested formats
  REPORT.html            local figure gallery
```

These files are created locally when requested; they are not bundled.
STAR input files are likewise ignored by Git. E1 and online are self-contained
and therefore each compute their own true-M references if both are run.
No saved results from an older package are silently imported into a new plan.

## Reproducibility and scope

The package regenerates the executed mechanisms, loss schedules, and RNG streams.
Floating-point optimization can vary across numerical-library/platform versions.
`requirements-tested.txt` lists versions used for verification; `requirements.txt`
provides compatible ranges. Tests and short prefix checks validate implementations
and data plumbing; they do not re-establish the paper's statistical conclusions.

See [VALIDATION.md](VALIDATION.md) for the parity checks and test coverage.

Only these five experiment families are included. Earlier tuning sweeps, ACTG,
mediator-hiding analyses, rejected adaptivity designs, and unrelated cardinality
sweeps are absent. [EXPERIMENTS.md](EXPERIMENTS.md) gives exact definitions,
[REFERENCES.md](REFERENCES.md) gives method sources, and
[PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) explains figure-to-code mapping and
limitations. The scripts never upload or publish anything.
