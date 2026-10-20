import sys, time; sys.path.insert(0, '.')
import pandas as pd
from sklearn.metrics import accuracy_score
from src.skheros.heros import HEROS

d = 'evaluation/datasets/partitioned/multiplexer/A_multiplexer_6_bit_500_inst_CV_'
train = pd.read_csv(d + 'Train_1.txt', sep='\t')
test = pd.read_csv(d + 'Test_1.txt', sep='\t')
drop = ['Class', 'Group', 'InstanceID']

X_tr, y_tr = train.drop(columns=drop).values, train['Class'].values
X_te, y_te = test.drop(columns=drop).values, test['Class'].values
row_id = train['InstanceID'].values

iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 10000 # e.g. python evaluation/run_mux6_quant.py 1000

configs_all = [('baseline, categorical', None, True),
           ('local_quant_opt, categorical', 'local_quantitative_optimization', True),
           ('local_quant_opt, quantitative', 'local_quantitative_optimization', False)]

configs = configs_all[2:]

for name, method, cat in configs:
    t = time.time()
    h = HEROS(outcome_type='class', iterations=iterations, pop_size=500, cross_prob=0.8, mut_prob=0.04, nu=1, beta=0.2,
              theta_sel=0.5, fitness_function='pareto', subsumption='both', rsl=0, model_iterations=40,
              model_pop_size=100, model_pop_init='target_acc', new_gen=1.0, merge_prob=0.1, compaction='sub',
              random_state=42, verbose=False, optimization_method=method)
    
    h.fit(X_tr, y_tr, row_id, cat_feat_indexes=list(range(X_tr.shape[1])) if cat else [])
    best = h.auto_select_top_model(X_tr, y_tr)
    tr = accuracy_score(y_tr, h.predict(X_tr, whole_rule_pop=False, target_model=best))
    te = accuracy_score(y_te, h.predict(X_te, whole_rule_pop=False, target_model=best))
    rules = h.get_model_rules(best)
    print(f"{name:32s} train={tr:.3f} test={te:.3f} rules={len(rules)} time={time.time()-t:.0f}s", flush=True)
    if method and not cat:
        print(rules.to_string())
