import sys, os, time, argparse; sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from skrebate import MultiSWRFDB
from src.skheros.heros import HEROS

# Baseline vs local_quantitative_optimization using the dataset setup and HEROS parameters from HEROS_Demo_Notebook.ipynb
# Usage: python evaluation/run_notebook_comparison.py MUX6     (or MUX6_QF, MUX6_QF_SPLIT, MUX6_QF_500, or a *_QUANT dataset from QUANT_DATASETS below)
# Replicate run (one CV fold, one seed, one config, results saved to csv), e.g. as a Slurm array task (see evaluation/hpc/run_replicates.sh):
#        python evaluation/run_notebook_comparison.py MUX6_QF_500 --configs local_quant_opt_gini --cv 10 --fold 3 --seed 7 --out results/local_quant_opt_gini_fold3_seed7.csv
parser = argparse.ArgumentParser()
parser.add_argument('dataset', nargs='?', default='MUX6')
parser.add_argument('--configs', default='baseline,local_quant_opt_tree,local_quant_opt_gini', help='comma-separated configs to run')
parser.add_argument('--seed', type=int, default=42, help='HEROS random_state (data partitions do not depend on it)')
parser.add_argument('--cv', type=int, default=None, help='number of stratified CV folds to split the pooled train+test data into')
parser.add_argument('--fold', type=int, default=None, help='CV fold (0 to cv-1) to hold out as testing data; requires --cv')
parser.add_argument('--out', default=None, help='optional csv path to save the summary table')
parser.add_argument('--iterations', type=int, default=50000, help='HEROS rule-learning iterations')
parser.add_argument('--vacuous_tolerance', type=float, default=0.0, help='max total fraction of training instances excluded by quantitative conditions dropped after optimization')
args = parser.parse_args()
if (args.cv is None) != (args.fold is None) or (args.cv is not None and not 0 <= args.fold < args.cv):
    parser.error('--cv and --fold must be given together, with 0 <= fold < cv')
example_dataset = args.dataset
split_data = False # True: make a stratified train/test split from the single training file

# Fully quantitative versions of the multiplexer and GAMETES datasets (made by evaluation/datasets/make_quantitative.py).
# One file each: use --cv to split it into folds, or leave --cv out for a stratified 75/25 train/test split.
QUANT_DATASETS = {'MUX6_QUANT': 'multiplexer_data_quant/A_multiplexer_6_bit_500_inst_QF.txt',
                  'MUX11_QUANT': 'multiplexer_data_quant/B_multiplexer_11_bit_5000_inst_QF.txt',
                  'MUX20_QUANT': 'multiplexer_data_quant/C_multiplexer_20_bit_10000_inst_QF.txt',
                  'MUX37_QUANT': 'multiplexer_data_quant/D_multiplexer_37_bit_10000_inst_QF.txt',
                  'MUX70_QUANT': 'multiplexer_data_quant/E_multiplexer_70_bit_20000_inst_QF.txt',
                  'MUX135_QUANT': 'multiplexer_data_quant/F_multiplexer_135_bit_20000_inst_QF.txt',
                  'GAMETES_uni_4add_QUANT': 'gametes_data_quant/A_uni_4add_QF.txt',
                  'GAMETES_univariate_QUANT': 'gametes_data_quant/B_univariate_QF.txt',
                  'GAMETES_2way_epistasis_QUANT': 'gametes_data_quant/C_2way_epistasis_QF.txt',
                  'GAMETES_2way_epi_2het_QUANT': 'gametes_data_quant/D_2way_epi_2het_QF.txt',
                  'GAMETES_uni_4het_QUANT': 'gametes_data_quant/E_uni_4het_QF.txt',
                  'GAMETES_3way_epistasis_QUANT': 'gametes_data_quant/F_3way_epistasis_QF.txt'}

# Dataset setup (as in notebook)
outcome_label = 'Class'
instanceID_label = 'InstanceID'
excluded_columns = [] # non-feature columns to drop (besides the instance ID and outcome)
if example_dataset in QUANT_DATASETS:
    train_data_path = 'evaluation/datasets/' + QUANT_DATASETS[example_dataset]
    test_data_path = None
    excluded_columns = ['Group', 'Model'] # multiplexer address group / GAMETES heterogeneity model labels (would leak the answer)
    cat_feat_indexes = []
    split_data = True
elif example_dataset == 'MUX6':
    train_data_path = 'evaluation/datasets/partitioned/multiplexer/A_multiplexer_6_bit_500_inst_CV_Train_1.txt'
    test_data_path = 'evaluation/datasets/partitioned/multiplexer/A_multiplexer_6_bit_500_inst_CV_Test_1.txt'
    excluded_columns = ['Group']
    cat_feat_indexes = list(range(6))
elif example_dataset == 'MUX6_QF': # notebook trains and tests on the same file for this dataset
    train_data_path = 'evaluation/datasets/unit_testing/Multiplexer6_feature_type_mix_decimals.csv'
    test_data_path = 'evaluation/datasets/unit_testing/Multiplexer6_feature_type_mix_decimals.csv'
    outcome_label = 'outcome'
    cat_feat_indexes = [0, 1, 2, 3]
elif example_dataset == 'MUX6_QF_SPLIT': # same file as MUX6_QF, but with separate training and testing instances
    train_data_path = 'evaluation/datasets/unit_testing/Multiplexer6_feature_type_mix_decimals.csv'
    test_data_path = None
    outcome_label = 'outcome'
    cat_feat_indexes = [0, 1, 2, 3]
    split_data = True
elif example_dataset == 'MUX6_QF_500': # MUX6 500 instance CV partition with R_2 and R_3 made quantitative
    train_data_path = 'evaluation/datasets/unit_testing/A_multiplexer_6_bit_500_inst_QF_CV_Train_1.txt'
    test_data_path = 'evaluation/datasets/unit_testing/A_multiplexer_6_bit_500_inst_QF_CV_Test_1.txt'
    excluded_columns = ['Group']
    cat_feat_indexes = [0, 1, 2, 3]
else:
    raise ValueError('Dataset must be MUX6, MUX6_QF, MUX6_QF_SPLIT, MUX6_QF_500, or one of ' + ', '.join(QUANT_DATASETS))
sep = '\t' if train_data_path.endswith('.txt') else ','

# HEROS parameters (as in notebook) 
stored_rule_iterations = ','.join(str(i) for i in sorted({500, 1000, 5000, 10000, args.iterations}) if i <= args.iterations)
params = dict(outcome_type='class', iterations=args.iterations, pop_size=500, cross_prob=0.8, mut_prob=0.04, nu=1, beta=0.2, theta_sel=0.5,
              fitness_function='pareto', subsumption='both', rsl=0, feat_track='end', model_iterations=100,
              model_pop_size=100, model_pop_init='target_acc', new_gen=1.0, merge_prob=0.1, rule_pop_init='dt_bstrap', compaction='sub',
              track_performance=1000, model_tracking=True, stored_rule_iterations=stored_rule_iterations, stored_model_iterations='10,50,100',
              random_state=args.seed, verbose=False, alternate=5, alternate_mode='equal', feedback=False, vacuous_tolerance=args.vacuous_tolerance)
max_instances = 2000

def prepare(df):
    """Drop excluded, instance ID and outcome columns (as in notebook), and empty 'Unnamed' columns read from trailing tabs"""
    X = df.drop(columns=[c for c in df.columns if c in excluded_columns or c == instanceID_label or str(c).startswith('Unnamed:')])
    X = X.drop(outcome_label, axis=1)
    row_id = df[instanceID_label].values if instanceID_label in df.columns else None
    return X, df[outcome_label].values, row_id

def balanced_sampling(df, outcome_label, max_instances):
    """Sample up to max_instances rows, retaining class balance if possible (as in notebook)"""
    df_class_0 = df[df[outcome_label] == 0]
    df_class_1 = df[df[outcome_label] == 1]
    n_class_0 = min(max_instances // 2, len(df_class_0))
    n_class_1 = min(max_instances - max_instances // 2, len(df_class_1))
    return pd.concat([df_class_0.sample(n=n_class_0, replace=False), df_class_1.sample(n=n_class_1, replace=False)]).reset_index(drop=True)

# Load data
if args.cv is not None: # pool all instances, then hold out one fold of a fixed stratified split (same folds for every seed and config)
    full_df = pd.concat([pd.read_csv(p, sep=sep) for p in dict.fromkeys([train_data_path, test_data_path]) if p], ignore_index=True)
    folds = list(StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42).split(full_df, full_df[outcome_label]))
    train_idx, test_idx = folds[args.fold]
    train_df, test_df = full_df.iloc[train_idx], full_df.iloc[test_idx]
elif split_data: # stratified 75/25 split, so testing instances are never seen in training (or in expert knowledge scoring)
    full_df = pd.read_csv(train_data_path, sep=sep)
    train_df, test_df = train_test_split(full_df, test_size=0.25, stratify=full_df[outcome_label], random_state=42)
else:
    train_df = pd.read_csv(train_data_path, sep=sep)
    test_df = pd.read_csv(test_data_path, sep=sep)
X_df, y, row_id = prepare(train_df)
feature_names = list(X_df.columns)
X = X_df.values
X_test_df, y_test, _ = prepare(test_df)
X_test = X_test_df.values
quant_feat_indexes = [i for i in range(X.shape[1]) if i not in cat_feat_indexes]

# Expert knowledge: load the notebook's cached MultiSWRFDB scores, or generate and cache them as the notebook does
output_path = './output/' + os.path.splitext(os.path.basename(train_data_path))[0] + ('_train_split' if split_data else '')
if args.cv is not None: # scores from this fold's training data only
    output_path = f'./output/{example_dataset}_cv{args.cv}/fold_{args.fold}'
score_path_name = output_path + '/MultiSWRFDB_Scores.csv'
if not os.path.isfile(score_path_name):
    print('Generating MultiSWRFDB Scores')
    os.makedirs(output_path, exist_ok=True)
    fs_X, fs_y, _ = prepare(balanced_sampling(train_df, outcome_label, max_instances))
    ek = MultiSWRFDB(n_jobs=None, categorical_features=cat_feat_indexes).fit(fs_X.values, fs_y).feature_importances_
    tmp_path = f'{score_path_name}.{os.getpid()}.tmp' # write then rename, so parallel array tasks never read a half-written file
    pd.DataFrame({'Feature': feature_names, 'Score': ek}).to_csv(tmp_path, index=False)
    os.replace(tmp_path, score_path_name)
else:
    print('Loading MultiSWRFDB Scores')
ek = list(pd.read_csv(score_path_name)['Score'])
print(dict(zip(feature_names, np.round(ek, 4))), flush=True)

def describe_conditions(rule):
    """Readable condition string, e.g. A_0=1 AND R_2 in [0.54, inf]"""
    parts = []
    for feat, value in zip(rule.condition_indexes, rule.condition_values):
        if isinstance(value, list):
            parts.append(f"{feature_names[feat]} in [{value[0]:.2f}, {value[1]:.2f}]")
        else:
            parts.append(f"{feature_names[feat]}={value:g}")
    return ' AND '.join(parts) if parts else '(no conditions)'

# Run comparison 
configs = [('baseline', None),
           ('local_quant_opt_tree', 'local_quantitative_optimization'), 
           ('local_quant_opt_gini', 'local_quantitative_optimization_v3')]

unknown = set(args.configs.split(',')) - {c[0] for c in configs}
if unknown:
    parser.error(f"unknown config(s) {sorted(unknown)}; choose from {[c[0] for c in configs]}")
configs = [c for c in configs if c[0] in args.configs.split(',')]

results = []
for name, method in configs:
    print(f"\n{'=' * 30} {example_dataset}: {name} {'=' * 30}", flush=True)
    t = time.time()
    h = HEROS(**params, optimization_method=method)
    h.fit(X, y, row_id, cat_feat_indexes=cat_feat_indexes, ek=None)
    run_time = time.time() - t

    # Model metrics (top model selected on training data)
    best = h.auto_select_top_model(X, y)
    model_rules = h.get_model_rules(best)
    train_bal_acc = balanced_accuracy_score(y, h.predict(X, whole_rule_pop=False, target_model=best))
    test_bal_acc = balanced_accuracy_score(y_test, h.predict(X_test, whole_rule_pop=False, target_model=best))
    test_coverage = h.predict_covered(X_test, whole_rule_pop=False, target_model=best).mean()

    # Rule population investigation
    pop = h.rule_population.pop_set
    print(f"Rule population: {len(pop)} unique rules, {sum(r.numerosity for r in pop)} total numerosity")
    if quant_feat_indexes:
        uses_quant = sum(any(f in quant_feat_indexes for f in r.condition_indexes) for r in pop)
        print(f"  rules using quantitative features: {uses_quant} ({uses_quant / max(len(pop), 1):.0%})")
    print(f"  mean conditions per rule: {np.mean([len(r.condition_indexes) for r in pop]):.2f}")
    print(f"  mean rule accuracy: {np.mean([r.accuracy for r in pop]):.3f}, "
          f"rules with accuracy 1.0: {sum(r.accuracy == 1.0 for r in pop)}")
    print(f"  rule pareto front (useful accuracy, useful coverage): {h.rule_pareto.rule_front}")
    print(f"Top model rule set ({len(model_rules)} rules):")
    for r in sorted(h.model_population.pop_set[best].rule_set, key=lambda r: -r.fitness):
        print(f"  {describe_conditions(r):60s} -> class {r.action}  acc={r.accuracy:.2f} "
              f" cover={r.useful_coverage} fitness={r.fitness:.3f}")

    results.append({'config': name, 'fold': args.fold, 'seed': args.seed, 'rule set size': len(model_rules),
                    'train balanced acc': round(train_bal_acc, 3), 'test balanced acc': round(test_bal_acc, 3),
                    'test coverage': round(test_coverage, 3), 'rule pop size': len(pop), 'run time (s)': round(run_time, 1)})

print(f"\n{'=' * 30} {example_dataset} summary ({len(y)} train / {len(y_test)} test rows) {'=' * 30}")
print(pd.DataFrame(results).to_string(index=False))
if args.out:
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    pd.DataFrame(results).assign(dataset=example_dataset, iterations=args.iterations, vacuous_tolerance=args.vacuous_tolerance).to_csv(args.out, index=False)
