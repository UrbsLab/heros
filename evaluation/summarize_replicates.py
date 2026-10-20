import sys, glob, os
import pandas as pd
from scipy.stats import wilcoxon

# Combine per-run csvs from evaluation/hpc/run_replicates.sh (one per config x CV fold x seed) and compare configs
# Usage: python evaluation/summarize_replicates.py results/replicates/MUX6_QF_500
results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results/replicates/MUX6_QF_500'
files = glob.glob(os.path.join(results_dir, '*_fold*_seed*.csv'))
if not files:
    sys.exit(f'No result files found in {results_dir}')
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(os.path.join(results_dir, 'all_replicates.csv'), index=False)

metrics = ['rule set size', 'train balanced acc', 'test balanced acc', 'test coverage', 'rule pop size', 'run time (s)']
counts = df.groupby('config').agg(runs=('seed', 'size'), folds=('fold', 'nunique'), seeds=('seed', 'nunique'))
print(f"Completed runs per config:\n{counts.to_string()}\n")
print("Mean (std) over all fold x seed runs:")
print(df.groupby('config')[metrics].agg(['mean', 'std']).round(3).T.to_string(), '\n')

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
