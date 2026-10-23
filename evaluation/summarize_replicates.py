import sys, glob, os, argparse
import pandas as pd
from scipy.stats import wilcoxon

# Combine per-run csvs from evaluation/hpc/run_replicates.sh (one per config x CV fold x seed) and compare configs
# Usage: python evaluation/summarize_replicates.py results/MUX6_QF_500                (writes all_replicates.csv and summary.csv there)
#        python evaluation/summarize_replicates.py results/my_run/runs --out-dir results/my_run --expected 600 --remove-runs
parser = argparse.ArgumentParser()
parser.add_argument('results_dir', nargs='?', default='results/MUX6_QF_500', help='folder holding the per-run csvs')
parser.add_argument('--out-dir', default=None, help='where to write all_replicates.csv and summary.csv (default: results_dir)')
parser.add_argument('--expected', type=int, default=None, help='number of per-run csvs a complete experiment has')
parser.add_argument('--remove-runs', action='store_true', help='delete the per-run csvs after combining, only if all --expected runs are present')
args = parser.parse_args()
results_dir = args.results_dir
out_dir = args.out_dir or results_dir
files = glob.glob(os.path.join(results_dir, '*_fold*_seed*.csv'))
if not files:
    sys.exit(f'No result files found in {results_dir}')
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
os.makedirs(out_dir, exist_ok=True)
df.to_csv(os.path.join(out_dir, 'all_replicates.csv'), index=False)

metrics = ['rule set size', 'train balanced acc', 'test balanced acc', 'test coverage', 'rule pop size', 'run time (s)']
counts = df.groupby('config').agg(runs=('seed', 'size'), folds=('fold', 'nunique'), seeds=('seed', 'nunique'))
print(f"Completed runs per config:\n{counts.to_string()}\n")
summary = df.groupby('config')[metrics].mean().add_prefix('mean ')
summary['min rule set size'] = df.groupby('config')['rule set size'].min()
summary['max rule set size'] = df.groupby('config')['rule set size'].max()
summary = summary.round(3)
summary.to_csv(os.path.join(out_dir, 'summary.csv'))
print("Mean over all fold x seed runs (rule set size min/max across runs):")
print(summary.T.to_string(), '\n')

# Runs sharing a fold reuse the same training data, so they are not independent. Average each seed over its folds
# (a CV estimate per seed), then compare configs with a paired Wilcoxon signed-rank test across seeds.
cv_means = df.groupby(['config', 'seed'])[metrics].mean()
if 'baseline' in df['config'].values:
    base = cv_means.loc['baseline']
    for config in sorted(set(df['config']) - {'baseline'}):
        other = cv_means.loc[config]
        seeds = base.index.intersection(other.index)
        print(f"{config} vs baseline (CV mean per seed, {len(seeds)} paired seeds):")
        for m in metrics:
            diff = other.loc[seeds, m] - base.loc[seeds, m]
            p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
            print(f"  {m:20s} mean diff {diff.mean():+.3f}   p = {p:.4g}")

# Keep the per-run csvs if any run is missing, so failed tasks can be rerun and the summary redone
if args.expected is not None and len(files) != args.expected:
    print(f"\nWARNING: found {len(files)} of {args.expected} expected runs; per-run csvs kept in {results_dir}")
elif args.remove_runs:
    for f in files:
        os.remove(f)
    if not os.listdir(results_dir):
        os.rmdir(results_dir)
    print(f"\nAll {len(files)} runs combined; removed the per-run csvs")
