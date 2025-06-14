import os
import pandas as pd
from skheros.heros import HEROS
from sklearn.metrics import classification_report

# To test using pytest run `pytest --log-cli-level=DEBUG` from the root folder`

def test_general():
    df = pd.read_csv('test/datasets/Multiplexer6.csv')
    outcome_label = 'class'
    X = df.drop(outcome_label, axis=1)
    #feature_names = X.columns # 6-bit multiplexer feature names are ['A_0','A_1','R_0', 'R_1', 'R_2','R_3']
    cat_feat_indexes = list(range(X.shape[1])) #all feature are categorical so provide indexes 0-5 in this list for 6-bit multiplexer dataset
    X = X.values
    y = df[outcome_label].values #outcome values
    ek = [0.05271707, 0.05271707, 0.00982932, 0.00982932, 0.00982932, 0.00982932]
    #ek = [0.05271707, 0.05271707, 0.05271707, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932]
    #ek = [0.05271707, 0.05271707, 0.05271707, 0.05271707, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932]
    #ek = [0.05271707, 0.05271707, 0.05271707, 0.05271707, 0.05271707, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932, 0.00982932]
    heros = HEROS(outcome_type='class',iterations=10000, pop_size=1000, cross_prob=0.8, mut_prob=0.04, nu=10, beta=0.2, theta_sel=0.5,fitness_function='pareto',
                  subsumption='both', use_ek=True, rsl=None, feat_track='add', pop_init_type=None, compaction='sub',
                  track_performance=1000,random_state=42, verbose=True)
    heros.fit(X, y, cat_feat_indexes=cat_feat_indexes, ek=ek)
    heros.env.report_data_logging()
    pop_df = heros.get_pop()
    pop_df.to_csv('output/demo_rule_pop.csv', index=False)
    get_ft = heros.get_ft()
    get_ft .to_csv('output/demo_ft.csv', index=False)

def test_load_general():
    df = pd.read_csv('test/datasets/Multiplexer6.csv')
    outcome_label = 'class'
    X = df.drop(outcome_label, axis=1)
    cat_feat_indexes = list(range(X.shape[1])) #all feature are categorical so provide indexes 0-5 in this list for 6-bit multiplexer dataset
    X = X.values
    y = df[outcome_label].values #outcome values
    ek = [0.05271707, 0.05271707, 0.00982932, 0.00982932, 0.00982932, 0.00982932]
    pop_df = pd.read_csv('output/demo_rule_pop.csv')
    heros = HEROS(outcome_type='class',iterations=10000, pop_size=500, cross_prob=0.8, mut_prob=0.04, nu=10, beta=0.2, theta_sel=0.5,fitness_function='pareto',
                  subsumption='both', use_ek=True, rsl=None, feat_track='add', pop_init_type='load', compaction='sub',
                  track_performance=1000,random_state=42, verbose=True)
    heros.fit(X, y, cat_feat_indexes=cat_feat_indexes, pop_df=pop_df, ek=ek)
    heros.env.report_data_logging()
    pop_df = heros.get_pop()
    pop_df.to_csv('output/demo_load_init_rule_pop.csv', index=False)
    get_ft = heros.get_ft()
    get_ft .to_csv('output/demo_load_init_ft.csv', index=False)

def test_mixed_feature_types():
    df = pd.read_csv('test/datasets/Multiplexer6_quantitative.csv')
    outcome_label = 'class'
    X = df.drop(outcome_label, axis=1)
    cat_feat_indexes = [0,1,2,3] #all feature are categorical so provide indexes 0-5 in this list for 6-bit multiplexer dataset
    X = X.values
    y = df[outcome_label].values #outcome values
    ek = [0.05271707, 0.05271707, 0.00982932, 0.00982932, 0.00982932, 0.00982932]
    heros = HEROS(outcome_type='class',iterations=10000, pop_size=500, cross_prob=0.8, mut_prob=0.04, nu=10, beta=0.2, theta_sel=0.5,fitness_function='pareto',
                  subsumption='both', use_ek=True, rsl=None, feat_track='add', pop_init_type=None, compaction='sub',
                  track_performance=1000,random_state=42, verbose=True)
    heros.fit(X, y, cat_feat_indexes=cat_feat_indexes, ek=ek)
    heros.env.report_data_logging()
    pop_df = heros.get_pop()
    pop_df.to_csv('output/demo_rule_pop.csv', index=False)
    get_ft = heros.get_ft()
    get_ft .to_csv('output/demo_ft.csv', index=False)

def test_quantitative_outcome():
    df = pd.read_csv('test/datasets/Multiplexer6_quantitative_outcome.csv')
    outcome_label = 'outcome'
    X = df.drop(outcome_label, axis=1)
    cat_feat_indexes = [0,1,2,3] #all feature are categorical so provide indexes 0-5 in this list for 6-bit multiplexer dataset
    X = X.values
    y = df[outcome_label].values #outcome values
    ek = [0.05271707, 0.05271707, 0.00982932, 0.00982932, 0.00982932, 0.00982932]
    heros = HEROS(outcome_type='quant',iterations=10000, pop_size=500, cross_prob=0.8, mut_prob=0.04, nu=10, beta=0.2, theta_sel=0.5,fitness_function='pareto',
                  subsumption='both', use_ek=True, rsl=None, feat_track='add', pop_init_type=None, compaction='sub',
                  track_performance=1000,random_state=42, verbose=True)
    heros.fit(X, y, cat_feat_indexes=cat_feat_indexes, ek=ek)
    heros.env.report_data_logging()
    pop_df = heros.get_pop()
    pop_df.to_csv('output/demo_rule_pop.csv', index=False)
    get_ft = heros.get_ft()
    get_ft .to_csv('output/demo_ft.csv', index=False)

def test_na():
    df = pd.read_csv('test/datasets/Multiplexer6_NA.csv')
    outcome_label = 'class'
    X = df.drop(outcome_label, axis=1)
    cat_feat_indexes = list(range(X.shape[1])) #all feature are categorical so provide indexes 0-5 in this list for 6-bit multiplexer dataset
    X = X.values
    y = df[outcome_label].values #outcome values
    ek = [0.05271707, 0.05271707, 0.00982932, 0.00982932, 0.00982932, 0.00982932]
    heros = HEROS(outcome_type='class',iterations=10000, pop_size=500, cross_prob=0.8, mut_prob=0.04, nu=10, beta=0.2, theta_sel=0.5,fitness_function='pareto',
                  subsumption='both', use_ek=True, rsl=None, feat_track='add', pop_init_type=None, compaction='sub',
                  track_performance=1000,random_state=42, verbose=True)
    heros.fit(X, y, cat_feat_indexes=cat_feat_indexes, ek=ek)
    heros.env.report_data_logging()
    pop_df = heros.get_pop()
    pop_df.to_csv('output/demo_rule_pop.csv', index=False)
    get_ft = heros.get_ft()
    get_ft .to_csv('output/demo_ft.csv', index=False)

if __name__ == "__main__":
    test_general()
    #test_load_general()
    #test_mixed_feature_types()
    #test_quantitative_outcome()
    #test_na()

