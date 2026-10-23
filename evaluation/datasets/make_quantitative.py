import os, glob
import numpy as np
import pandas as pd

# Convert the categorical multiplexer and GAMETES datasets into quantitative-feature versions.
# Each feature value is replaced by a uniform random value from its own non-overlapping range:
#   multiplexer (bits):      0 -> [0, 0.5),  1 -> [0.5, 1)
#   GAMETES (genotypes):     0 -> [0, 1),    1 -> [1, 2),    2 -> [2, 3)
# Class, Group, InstanceID and Model columns and missing values are left as they are.
# Values are rounded down to 4 decimals (as in the existing QF datasets), so they never cross into the next range.
# Output: evaluation/datasets/<folder>_quant/<name>_QF.txt
# Usage (from the repo root): python evaluation/datasets/make_quantitative.py
SEED = 42
DECIMALS = 4
NON_FEATURES = ['Class', 'Group', 'InstanceID', 'Model']
RANGES = {'multiplexer_data': {0: (0.0, 0.5), 1: (0.5, 1.0)},
          'gametes_data': {0: (0.0, 1.0), 1: (1.0, 2.0), 2: (2.0, 3.0)}}

rng = np.random.default_rng(SEED)
for folder, ranges in RANGES.items():
    out_folder = f'evaluation/datasets/{folder}_quant'
    os.makedirs(out_folder, exist_ok=True)
    for path in sorted(glob.glob(f'evaluation/datasets/{folder}/*.txt')):
        df = pd.read_csv(path, sep='\t')
        # Lines ending in a tab read as an empty 'Unnamed' column; keep it nameless so the output lines end the same way
        df.columns = ['' if c.startswith('Unnamed:') else c for c in df.columns]
        features = [c for c in df.columns if c not in NON_FEATURES and c != '']
        values = df[features].to_numpy(dtype=float)
        unexpected = set(np.unique(values[~np.isnan(values)])) - set(ranges)
        if unexpected:
            raise ValueError(f'{path}: unexpected feature values {sorted(unexpected)}')
        quant = np.full(values.shape, np.nan)
        for value, (low, high) in ranges.items():
            mask = values == value
            draws = rng.uniform(low, high, mask.sum())
            quant[mask] = np.floor(draws * 10 ** DECIMALS) / 10 ** DECIMALS
        df[features] = quant
        out_path = os.path.join(out_folder, os.path.splitext(os.path.basename(path))[0] + '_QF.txt')
        df.to_csv(out_path, sep='\t', index=False, float_format=f'%.{DECIMALS}f')
        print(f'{out_path}: {len(df)} rows, {len(features)} features converted')
