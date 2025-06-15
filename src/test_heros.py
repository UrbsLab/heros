import os
import pandas as pd
from skheros.heros import HEROS
from sklearn.metrics import classification_report
from skrebate import MultiSURF # Install using: pip install skrebate==0.7

# To test using pytest run `pytest --log-cli-level=DEBUG` from the root folder`

def get_EK(X,y,data_name,feature_names):
    """ Calculates or loads expert knowledge (EK) scores for the given unit testing dataset """
    # Further data preparation for expert knowldge generation (i.e. balanced subsampling of training instances for faster run times)
    score_path_name = 'evaluation/datasets/unit_testing/unit_EK/'+str(data_name)+'_MultiSURF_Scores.csv' #No need to change
    # Calculate or load feature importance estimates with MultiSURF ------------------------------------
    if not os.path.isfile(score_path_name):
        print("Generating MultiSURF Scores:")
        clf = MultiSURF(n_jobs=1).fit(X, y)
        ek = clf.feature_importances_
        score_data = pd.DataFrame({'Feature':feature_names,'Score':ek})
        score_data.to_csv(score_path_name,index=False)
    else: #load previously trained scores
        print("Loading MultiSURF Scores")
        loaded_data = pd.read_csv(score_path_name)
        ek = loaded_data['Score'].tolist()
    return ek

def test_6mux():
    print("------------------------------------------------------")
    print("Test: 6-bit MUX - binary features and outcome, No NA's")
    # Load and prepare data
    data_name = 'Multiplexer6'
    df = pd.read_csv('evaluation/datasets/unit_testing/'+str(data_name)+'.csv')
    outcome_label = 'outcome'
    X = df.drop(outcome_label, axis=1)
    feature_names = X.columns 
    cat_feat_indexes = list(range(X.shape[1])) #all feature are categorical so provide indexes 0-5 in this list for 6-bit multiplexer dataset
    X = X.values
    y = df[outcome_label].values #outcome values
    ek = get_EK(X,y,data_name,feature_names)
    print(ek)
    heros = HEROS(outcome_type='class',iterations=10000,pop_size=500,cross_prob=0.8,mut_prob=0.04,nu=1,beta=0.2,theta_sel=0.5,
                fitness_function='pareto',subsumption='both',rsl=0,feat_track=None, model_iterations=40,
                model_pop_size=100,model_pop_init='target_acc',new_gen=1.0,merge_prob=0.1,rule_pop_init=None,compaction='sub',
                track_performance=1000,model_tracking=True,stored_rule_iterations=None,stored_model_iterations=None,random_state=42,verbose=True)
    heros = heros.fit(X, y, None, cat_feat_indexes=cat_feat_indexes, ek=ek)
    #Select best model from the front
    best_model_index = heros.auto_select_top_model(X, y,verbose=True)
    set_df = heros.get_model_rules(best_model_index)
    print(set_df) #Print all rules of the top model
    #Evaluate best model (here just on same training data)
    predictions = heros.predict(X, whole_rule_pop=False, target_model=best_model_index)
    print("HEROS Top Model Training Data Performance Report:")
    print(classification_report(predictions, y, digits=8))


def test_na():
    print("------------------------------------------------------")
    print("Test: 6-bit MUX with NAs - binary features and outcome")
    # Load and prepare data
    data_name = 'Multiplexer6_NA'
    df = pd.read_csv('evaluation/datasets/unit_testing/'+str(data_name)+'.csv')
    outcome_label = 'outcome'
    X = df.drop(outcome_label, axis=1)
    feature_names = X.columns 
    cat_feat_indexes = list(range(X.shape[1])) #all feature are categorical so provide indexes 0-5 in this list for 6-bit multiplexer dataset
    X = X.values
    y = df[outcome_label].values #outcome values
    ek = [0.0527170745920746, 0.0527170745920746, 0.00982931998557, 0.0098293199855699, 0.0098293199855699, 0.00982931998557] #temporary overide until Rebate fixed
    #ek = get_EK(X,y,data_name,feature_names)
    print(ek)
    heros = HEROS(outcome_type='class',iterations=10000,pop_size=500,cross_prob=0.8,mut_prob=0.04,nu=1,beta=0.2,theta_sel=0.5,
                fitness_function='pareto',subsumption='both',rsl=0,feat_track=None, model_iterations=40,
                model_pop_size=100,model_pop_init='target_acc',new_gen=1.0,merge_prob=0.1,rule_pop_init=None,compaction='sub',
                track_performance=1000,model_tracking=True,stored_rule_iterations=None,stored_model_iterations=None,random_state=42,verbose=True)
    heros = heros.fit(X, y, None, cat_feat_indexes=cat_feat_indexes, ek=ek)
    #Select best model from the front
    best_model_index = heros.auto_select_top_model(X, y,verbose=True)
    set_df = heros.get_model_rules(best_model_index)
    print(set_df) #Print all rules of the top model
    #Evaluate best model (here just on same training data)
    predictions = heros.predict(X, whole_rule_pop=False, target_model=best_model_index)
    print("HEROS Top Model Training Data Performance Report:")
    print(classification_report(predictions, y, digits=8))


def test_mixed_feature_types():
    print("------------------------------------------------------")
    print("Test: 6-bit MUX with mixed feature types - binary and quantitative features and binary outcome, No NA's")
    # Load and prepare data
    data_name = 'Multiplexer6_feature_type_mix'
    df = pd.read_csv('evaluation/datasets/unit_testing/'+str(data_name)+'.csv')
    outcome_label = 'outcome'
    X = df.drop(outcome_label, axis=1)
    feature_names = X.columns 
    cat_feat_indexes = [0,1,2,3] #all feature are categorical so provide indexes 0-5 in this list for 6-bit multiplexer dataset
    X = X.values
    y = df[outcome_label].values #outcome values
    ek = get_EK(X,y,data_name,feature_names)
    print(ek)
    heros = HEROS(outcome_type='class',iterations=20000,pop_size=500,cross_prob=0.8,mut_prob=0.04,nu=1,beta=0.2,theta_sel=0.5,
                fitness_function='pareto',subsumption='both',rsl=0,feat_track=None, model_iterations=40,
                model_pop_size=100,model_pop_init='target_acc',new_gen=1.0,merge_prob=0.1,rule_pop_init=None,compaction='sub',
                track_performance=1000,model_tracking=True,stored_rule_iterations=None,stored_model_iterations=None,random_state=42,verbose=True)
    heros = heros.fit(X, y, None, cat_feat_indexes=cat_feat_indexes, ek=ek)
    #Select best model from the front
    best_model_index = heros.auto_select_top_model(X, y,verbose=True)
    set_df = heros.get_model_rules(best_model_index)
    print(set_df) #Print all rules of the top model
    #Evaluate best model (here just on same training data)
    predictions = heros.predict(X, whole_rule_pop=False, target_model=best_model_index)
    print("HEROS Top Model Training Data Performance Report:")
    print(classification_report(predictions, y, digits=8))


def test_mixed_feature_types_na():
    print("------------------------------------------------------")
    print("Test: 6-bit MUX with mixed feature types and NAs - binary and quantitative features and binary outcome")
    # Load and prepare data
    data_name = 'Multiplexer6_NA'
    df = pd.read_csv('evaluation/datasets/unit_testing/'+str(data_name)+'.csv')
    outcome_label = 'outcome'
    X = df.drop(outcome_label, axis=1)
    feature_names = X.columns 
    cat_feat_indexes = [0,1,2,3] #all feature are categorical so provide indexes 0-5 in this list for 6-bit multiplexer dataset
    X = X.values
    y = df[outcome_label].values #outcome values
    ek = [0.0527170745920746, 0.0527170745920746, 0.00982931998557, 0.0098293199855699, 0.0098293199855699, 0.00982931998557] #temporary overide until Rebate fixed
    #ek = get_EK(X,y,data_name,feature_names)
    print(ek)
    heros = HEROS(outcome_type='class',iterations=20000,pop_size=500,cross_prob=0.8,mut_prob=0.04,nu=1,beta=0.2,theta_sel=0.5,
                fitness_function='pareto',subsumption='both',rsl=0,feat_track=None, model_iterations=40,
                model_pop_size=100,model_pop_init='target_acc',new_gen=1.0,merge_prob=0.1,rule_pop_init=None,compaction='sub',
                track_performance=1000,model_tracking=True,stored_rule_iterations=None,stored_model_iterations=None,random_state=42,verbose=True)
    heros = heros.fit(X, y, None, cat_feat_indexes=cat_feat_indexes, ek=ek)
    #Select best model from the front
    best_model_index = heros.auto_select_top_model(X, y,verbose=True)
    set_df = heros.get_model_rules(best_model_index)
    print(set_df) #Print all rules of the top model
    #Evaluate best model (here just on same training data)
    predictions = heros.predict(X, whole_rule_pop=False, target_model=best_model_index)
    print("HEROS Top Model Training Data Performance Report:")
    print(classification_report(predictions, y, digits=8))


def test_multiclass():
    print("------------------------------------------------------")
    print("Test: 6-bit MUX as a Multi class outcome problem - binary features and 8 class outcome, No NA's")
    # Load and prepare data
    data_name = 'Multiplexer6_multiclass'
    df = pd.read_csv('evaluation/datasets/unit_testing/'+str(data_name)+'.csv')
    outcome_label = 'outcome'
    X = df.drop(outcome_label, axis=1)
    feature_names = X.columns 
    cat_feat_indexes = list(range(X.shape[1])) #all feature are categorical so provide indexes 0-5 in this list for 6-bit multiplexer dataset
    X = X.values
    y = df[outcome_label].values #outcome values
    ek = get_EK(X,y,data_name,feature_names)
    print(ek)
    heros = HEROS(outcome_type='class',iterations=10000,pop_size=500,cross_prob=0.8,mut_prob=0.04,nu=1,beta=0.2,theta_sel=0.5,
                fitness_function='pareto',subsumption='both',rsl=0,feat_track=None, model_iterations=40,
                model_pop_size=100,model_pop_init='target_acc',new_gen=1.0,merge_prob=0.1,rule_pop_init=None,compaction='sub',
                track_performance=1000,model_tracking=True,stored_rule_iterations=None,stored_model_iterations=None,random_state=42,verbose=True)
    heros = heros.fit(X, y, None, cat_feat_indexes=cat_feat_indexes, ek=ek)
    #Select best model from the front
    best_model_index = heros.auto_select_top_model(X, y,verbose=True)
    set_df = heros.get_model_rules(best_model_index)
    print(set_df) #Print all rules of the top model
    #Evaluate best model (here just on same training data)
    predictions = heros.predict(X, whole_rule_pop=False, target_model=best_model_index)
    print("HEROS Top Model Training Data Performance Report:")
    print(classification_report(predictions, y, digits=8))


def test_quantitative_outcome():
    print("------------------------------------------------------")
    print("Test: 6-bit MUX as quantiative outcome problem - binary features, No NA's")
    # Load and prepare data
    data_name = 'Multiplexer6_quant_outcome'
    df = pd.read_csv('evaluation/datasets/unit_testing/'+str(data_name)+'.csv')
    outcome_label = 'outcome'
    X = df.drop(outcome_label, axis=1)
    feature_names = X.columns 
    cat_feat_indexes = list(range(X.shape[1])) #all feature are categorical so provide indexes 0-5 in this list for 6-bit multiplexer dataset
    X = X.values
    y = df[outcome_label].values #outcome values
    ek = get_EK(X,y,data_name,feature_names)
    print(ek)
    heros = HEROS(outcome_type='quant',iterations=10000,pop_size=500,cross_prob=0.8,mut_prob=0.04,nu=1,beta=0.2,theta_sel=0.5,
                fitness_function='pareto',subsumption='both',rsl=0,feat_track=None, model_iterations=40,
                model_pop_size=100,model_pop_init='target_acc',new_gen=1.0,merge_prob=0.1,rule_pop_init=None,compaction='sub',
                track_performance=1000,model_tracking=True,stored_rule_iterations=None,stored_model_iterations=None,random_state=42,verbose=True)
    heros = heros.fit(X, y, None, cat_feat_indexes=cat_feat_indexes, ek=ek)
    #Incomplete implementation


if __name__ == "__main__":
    test_6mux()
    test_na()
    test_mixed_feature_types()
    test_mixed_feature_types_na()
    test_multiclass()
    #test_quantitative_outcome() #incomplete


