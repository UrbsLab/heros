import sys, time; sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from src.skheros.heros import HEROS

# 6-bit multiplexer with mixed feature types: A_0..R_1 categorical, R_2 and R_3 quantitative
# (64 rows and no provided split, so a stratified 75/25 train/test split is made here)
df = pd.read_csv('evaluation/datasets/unit_testing/Multiplexer6_feature_type_mix.csv')
X, y = df.drop(columns=['outcome']).values, df['outcome'].values
feature_names = list(df.drop(columns=['outcome']).columns)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
cat_feat_indexes = [0, 1, 2, 3]
quant_feat_indexes = [4, 5]
iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 10000 # e.g. python evaluation/run_mux6_feature_type_mix.py 1000

def describe_conditions(rule):
    """Readable condition string, e.g. A_0=1 AND R_2 in [0.54, inf]"""
    parts = []
    for feat, value in zip(rule.condition_indexes, rule.condition_values):
        if isinstance(value, list):
            parts.append(f"{feature_names[feat]} in [{value[0]:.2f}, {value[1]:.2f}]")
        else:
            parts.append(f"{feature_names[feat]}={value:g}")
    return ' AND '.join(parts) if parts else '(no conditions)'

configs = [('baseline', None),
           ('local_quant_opt', 'local_quantitative_optimization')]
results = []
for name, method in configs:
    print(f"\n{'=' * 30} {name} {'=' * 30}", flush=True)
    t = time.time()
    h = HEROS(outcome_type='class', iterations=iterations, pop_size=500, cross_prob=0.8, mut_prob=0.04, nu=1, beta=0.2,
              theta_sel=0.5, fitness_function='pareto', subsumption='both', rsl=0, model_iterations=40,
              model_pop_size=100, model_pop_init='target_acc', new_gen=1.0, merge_prob=0.1, compaction='sub',
              random_state=42, verbose=False, optimization_method=method)
    h.fit(X_tr, y_tr, None, cat_feat_indexes=cat_feat_indexes)
    run_time = time.time() - t

    # Model metrics (top model selected on training data)
    best = h.auto_select_top_model(X_tr, y_tr)
    model_rules = h.get_model_rules(best)
    train_bal_acc = balanced_accuracy_score(y_tr, h.predict(X_tr, whole_rule_pop=False, target_model=best))
    test_bal_acc = balanced_accuracy_score(y_te, h.predict(X_te, whole_rule_pop=False, target_model=best))
    test_coverage = h.predict_covered(X_te, whole_rule_pop=False, target_model=best).mean()

    # Rule population investigation
    pop = h.rule_population.pop_set
    uses_quant = [r for r in pop if any(f in quant_feat_indexes for f in r.condition_indexes)]
    print(f"Rule population: {len(pop)} unique rules, {sum(r.numerosity for r in pop)} total numerosity")
    print(f"  rules using R_2/R_3: {len(uses_quant)} ({len(uses_quant) / max(len(pop), 1):.0%})")
    print(f"  mean conditions per rule: {np.mean([len(r.condition_indexes) for r in pop]):.2f}")
    print(f"  mean rule accuracy: {np.mean([r.accuracy for r in pop]):.3f}, "
          f"rules with accuracy 1.0: {sum(r.accuracy == 1.0 for r in pop)}")
    print(f"  rule pareto front (useful accuracy, useful coverage): {h.rule_pareto.rule_front}")
    print(f"Top model rule set ({len(model_rules)} rules):")
    for r in sorted(h.model_population.pop_set[best].rule_set, key=lambda r: -r.fitness):
        print(f"  {describe_conditions(r):60s} -> class {r.action}  acc={r.accuracy:.2f} "
              f"cover={r.match_cover} fitness={r.fitness:.3f}")

    results.append({'config': name, 'rule set size': len(model_rules),
                    'train balanced acc': round(train_bal_acc, 3), 'test balanced acc': round(test_bal_acc, 3),
                    'test coverage': round(test_coverage, 3), 'run time (s)': round(run_time, 1)})

print(f"\n{'=' * 30} summary ({iterations} iterations, {len(y_tr)} train / {len(y_te)} test) {'=' * 30}")
print(pd.DataFrame(results).to_string(index=False))
