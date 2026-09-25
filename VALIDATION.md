# Validation of this update

Validation checks preserve the executed algorithm definitions; they do not
retune methods or rerun the paper's full simulations.

- All 18 learner implementation/default files are byte-identical to the previous
  anonymous package. Only experiment support, orchestration, presentation,
  documentation, and tests changed.
- Regenerated matrices, loss vectors, segment starts, and baseline groups are
  exactly equal to saved inputs for 90 random-piecewise E1 schedules, six
  transpose settings, 70 stationary random graphs, and 30 switching graphs.
- Sixty synthetic verification prefixes of 100 rounds match saved actions,
  mediators, losses, both regret traces, policy snapshots, and random-stream IDs
  exactly. Coverage includes all three E1 regimes, both transpose orientations,
  both random graph batches, and online mechanism estimation.
- All ten STAR methods match their original first 200 rounds exactly, preserving
  the intended learner horizon of 20,000. Matched quantities include sampled
  row indices, bounded losses, policies, and empirical-arm pseudo-regret.
  An interrupted STAR prefix also resumes with an identical verified prefix.
- STAR preparation reproduces all relevant saved cohort arrays exactly. The
  explicit public-source export and source/semantic checksums were verified.
- Default design preparation gives E1: 2,700 tasks; transpose: 540; random:
  1,000 (900 applicable and 100 CUCB2 N/A); online: 5,940; STAR: 50. These are
  planned task counts, not new full runs performed during packaging.
- Saved-endpoint rendering checks cover every paper layout. STAR uses empirical
  arm means for its evaluator; random plots normalize within each instance.
  Missing values and structural N/A remain explicit, without invented zeros.
- The automated tests exercise matrix/gap invariants, jointly attainable
  complexity certificates, paired RNG identity, seed-specific schedule routing,
  online pre-round estimates, STAR row replay and rank losses, failure retention,
  and deterministic resume. Run `python -m pytest -q` from this folder.

Data used for local parity checks are not included. The archive contains code,
configuration, documentation, tests, and empty local data/result directories.
Numerical library changes can affect floating-point optimization and exact
trajectory parity. See requirements-tested.txt for the verified environment.
