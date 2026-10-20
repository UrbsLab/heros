import sys, time; sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from src.skheros.heros import HEROS

# Baseline vs local_quantitative_optimization on the mixed feature type 6-bit multiplexer,
# evaluated with stratified 5-fold cross-validation (A_0..R_1 categorical, R_2 and R_3 quantitative)
df = pd.read_csv('evaluation/datasets/unit_testing/Multiplexer6_feature_type_mix.csv')
X, y = df.drop(columns=['outcome']).values, df['outcome'].values
cat_feat_indexes = [0, 1, 2, 3]
quant_feat_indexes = [4, 5]
iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 10000 # e.g. python evaluation/run_mux6_feature_type_mix_cv.py 1000
n_folds = 5

configs = [('baseline', None),
           ('local_quant_opt', 'local_quantitative_optimization')]
folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(X, y))
results = []
for name, method in configs:
    for fold, (train_idx, test_idx) in enumerate(folds):
        X_tr, X_te, y_tr, y_te = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        t = time.time()
        h = HEROS(outcome_type='class', iterations=iterations, pop_size=500, cross_prob=0.8, mut_prob=0.04, nu=1, beta=0.2,
                  theta_sel=0.5, fitness_function='pareto', subsumption='both', rsl=0, model_iterations=40,
                  model_pop_size=100, model_pop_init='target_acc', new_gen=1.0, merge_prob=0.1, compaction='sub',
                  random_state=42, verbose=False, optimization_method=method)
        h.fit(X_tr, y_tr, None, cat_feat_indexes=cat_feat_indexes)
        run_time = time.time() - t

        best = h.auto_select_top_model(X_tr, y_tr)
        pop = h.rule_population.pop_set
        row = {'config': name, 'fold': fold + 1,
               'rule set size': len(h.get_model_rules(best)),
               'train bal acc': balanced_accuracy_score(y_tr, h.predict(X_tr, whole_rule_pop=False, target_model=best)),
               'test bal acc': balanced_accuracy_score(y_te, h.predict(X_te, whole_rule_pop=False, target_model=best)),
               'test coverage': h.predict_covered(X_te, whole_rule_pop=False, target_model=best).mean(),
               'rule pop size': len(pop),
               'pop % using R_2/R_3': 100 * np.mean([any(f in quant_feat_indexes for f in r.condition_indexes) for r in pop]),
               'pop mean rule acc': np.mean([r.accuracy for r in pop]),
               'run time (s)': run_time}
        results.append(row)
        print(f"{name:16s} fold {fold + 1}: " + ', '.join(f"{k}={v:.3f}" for k, v in row.items() if k not in ('config', 'fold')), flush=True)

res = pd.DataFrame(results)
print(f"\n{'=' * 20} per-fold results ({iterations} iterations, {n_folds}-fold CV, {len(y)} rows) {'=' * 20}")
print(res.round(3).to_string(index=False))
print(f"\n{'=' * 20} mean (std) across folds {'=' * 20}")
summary = res.drop(columns='fold').groupby('config', sort=False).agg(['mean', 'std'])
print(pd.DataFrame({col: summary[col]['mean'].round(3).astype(str) + ' (' + summary[col]['std'].round(3).astype(str) + ')'
                    for col in summary.columns.levels[0] if col in summary}).T.to_string())
