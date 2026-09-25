"""Edit the settings below and Run in PyCharm, or use command-line options."""
# Choose exactly one experiment. Edit its JSON file to change parameters.
EXPERIMENT = 'e1'             # 'e1', 'transpose', 'random', 'online', or 'star'
ACTION = 'run'                 # 'prepare', 'run', or 'plot'
SMOKE = False                 # True: separate 200-round pipeline check
JOBS = None                   # None: use the config; e.g. 4 parallel workers

# Avoid nested BLAS threads inside each process; set before importing NumPy.
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'

from pathlib import Path
import argparse
from replication.configuration import load_config,smoke_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs')/(EXPERIMENT+'.json'))
    parser.add_argument('--action',choices=['prepare','run','plot'],default=ACTION)
    parser.add_argument('--smoke',action='store_true',default=SMOKE)
    parser.add_argument('--jobs',type=int,default=JOBS)
    parser.add_argument('--max-tasks',type=int,help='Run at most this many unfinished tasks in this invocation')
    parser.add_argument('--limit-rounds',type=int,help='Verification prefix only; learner horizon stays unchanged')
    args=parser.parse_args()
    config=load_config(args.config)
    if args.smoke:config=smoke_config(config)
    if args.jobs is not None:
        if args.jobs<1:parser.error('--jobs must be positive')
        config['jobs']=args.jobs
    if args.max_tasks is not None and args.max_tasks<1:parser.error('--max-tasks must be positive')
    if args.limit_rounds is not None and args.limit_rounds<1:parser.error('--limit-rounds must be positive')
    if args.action=='plot':
        from replication.plotting import plot_results
        from replication.io import safe_relative
        plot_results(safe_relative(config['output']),config['plot']);return
    from replication.runner import prepare,run_plan
    output=prepare(config)
    if args.action=='run':
        run_plan(output,jobs=config['jobs'],max_tasks=args.max_tasks,limit_rounds=args.limit_rounds)
        from replication.plotting import plot_results
        plot_results(output)


if __name__=='__main__':main()
