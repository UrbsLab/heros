import math
import logging
#import pandas as pd
import numpy as np
import math
        
class DATA_MANAGE:
    def __init__(self, X, y, row_id, cat_feat_indexes, ek, heros):
        np.random.seed(heros.random_state)
        # DATA HANDLING AND CHARACTERISTICS ***************************************************
        # Store number of each feature type (quantiative and categorical) 
        if cat_feat_indexes is None or cat_feat_indexes == 'None' or len(cat_feat_indexes) == 0: #all features are quantitative
            self.num_q_feat = X.shape[1]
            self.num_c_feat = 0
        else:
            self.num_q_feat = X.shape[1] - len(cat_feat_indexes)
            self.num_c_feat = len(cat_feat_indexes)
        #Identify feature types
        self.num_feat = X.shape[1] # total number of features in the dataset
        self.num_instances = X.shape[0] # total number of instances in the dataset
        # Specify feature types reference
        self.feat_types = []
        for i in range(self.num_feat): # 0 for quantitative, 1 for categorical
            if i in cat_feat_indexes:
                self.feat_types.append(1)
            else:
                self.feat_types.append(0)
        self.missing_values = np.isnan(X).sum()
        # Initialize class outcome attributes
        self.classes = [] #for class outcomes - stores unique class values in dataset
        self.class_counts = {} #for class outcomes - dictionary of classes with their respective instance counts
        self.class_weights = {} #for class outcomes - dictionary of classes with their respective instance weights (i.e. low class count = high weight)
        self.majority_class = None #for class outcomes - identifies the class with the most instances (if there's a tie, one is arbitrarily chosen)
        # Initialize quantitative outcome attributes
        self.outcome_range = [np.inf, -np.inf] #for quantitative outcomes - stores min and max quant outcome values in dataset
        self.outcome_ranked = None # for quantitative outcomes
        self.outcome_sd = None
        # Initialize feature attributes
        self.feat_q_range = [[np.inf, -np.inf] for _ in range(self.num_feat)] #for quantitative features - stores min and max quant feature values in dataset
        self.feat_c_values = [[] for _ in range(self.num_feat)] #for categorical features - stores unique feature values in dataset
        self.avg_feat_states = 0 #for calculating rule specificity limit - stores average number of unique feature values in each feature (quantitative features are assigned 2 unique values by default)
        # Update data attributes
        if heros.outcome_type == 'class':
            self.update_class_attributes(y)
        elif heros.outcome_type == 'quant': 
            self.outcome_range[0] = y.min()
            self.outcome_range[1] = y.max()
            self.outcome_ranked = sorted(y)
            self.outcome_sd = np.std(self.outcome_ranked, ddof=1) #calculates the 'sample' sample standard deviation # NEEDED?????
        else:
            pass
        self.update_feature_attributes(X)
        # Set rule specificity limit
        self.set_rule_specificity_limit(heros)
       # Format data (convert to numpy array, shuffle instances, and set NaN values to None)
        self.train_data = self.format_data(X, y, row_id) #stores formatted dataset used in training as numpy array (self.train_data[0] = feature values, and self.train_data[1] = outcome values)
        self.instance_index = 0 #instance index identifying what dataset instance is being focused on on a given iteration of the algorithm
        self.instance_state = self.train_data[0][self.instance_index] #gives the list of feature values for the current instance (i.e. instance 'state')
        self.outcome_state = self.train_data[1][self.instance_index] #gives the outcome value for the current instance (i.e. outcome 'state)
        self.instance_ids = self.train_data[2] #stores all instance ids 
        #EXPERT KNOWLEGE HANDLING *******************************************************
        self.ek_index_rank = None
        self.ek_weights = None
        if not ek is None:
            if len(ek) != self.num_feat:
                raise Exception("Length of expert knowledge list (i.e. 'ek' param) must be equal to the total number of features in the dataset and have the same feature order")
            else:
                heros.use_ek = True # Expert knowledge in the form of feature weights were provided
            # Generate EK index ranking ******************************************
            self.ek_index_rank =sorted(range(len(ek)), key=lambda x: ek[x], reverse=True)
            # Generate EK weights *************************************************
            self.transform_ek_to_weights(ek)
        if heros.verbose:
            self.report_data(heros) # Debug


    def clear_data_from_memory(self):
        self.train_data = None


    def transform_ek_to_weights(self,ek):
        """ Transform expert knowledge scores into rule specification probability weights"""
        self.ek_weights = ek
        # Scale weights within a specified value range
        new_min = 1.0
        new_max = 2.0
        min_val = min(self.ek_weights)
        max_val = max(self.ek_weights)
        self.ek_weights = [new_min + ((x - min_val) / (max_val - min_val)) * (new_max - new_min) for x in self.ek_weights]
        # Prevent zero division edge case
        if sum(self.ek_weights) == 0:
            self.ek_weights = self.ek_weights + 1
        # Convert from scores to probability/weights
        ek_sum = sum(self.ek_weights)
        self.ek_weights = [x / ek_sum for x in self.ek_weights]


    def update_class_attributes(self, y):
        """ Update classification outcome attributes for algorithm reference """
        for i in y:
            if i not in self.classes:
                self.classes.append(i)
                self.class_counts[i] = 1
                self.class_weights[i] = 1
            else:
                self.class_counts[i] += 1
                self.class_weights[i] += 1
        self.majority_class = max(self.class_counts, key=self.class_counts.get)
        for key in self.class_weights:
            self.class_weights[key] = 1 - (self.class_weights[key] / self.num_instances) # Classes with lower instance counts have higher weights


    def update_feature_attributes(self, X):
        """ """
        unique_state_count = 0
        for feat in range(self.num_feat):
            if self.feat_types[feat] == 1: #categorical
                for instance in range(self.num_instances):
                    value = X[instance,feat]
                    if not np.isnan(value) and value not in self.feat_c_values[feat]:  #.distinct_values:
                        self.feat_c_values[feat].append(value)
                        unique_state_count += 1
            else: #quantitative
                self.feat_q_range[feat][0] = np.min(X[:, feat])
                self.feat_q_range[feat][1] = np.max(X[:, feat])
                unique_state_count += 2 #Assumption/Estimate made here for quantitative features for calculating rule specificity limit

        self.avg_feat_states = unique_state_count / self.num_feat


    def set_rule_specificity_limit(self, heros):
        """Determine and set the rule specificity limit."""
        if heros.rsl == 0:
            limit = 1
            unique_combinations = math.pow(self.avg_feat_states, limit)
            while unique_combinations < self.num_instances:
                limit += 1
                unique_combinations = math.pow(self.avg_feat_states, limit)
            heros.rsl = min(limit, self.num_feat)


    def format_data(self, X, y, row_id):
        """Format training data: convert to np array, shuffle instances, and set NaNs as None."""
        if row_id is None:
            instance_ids = np.arange(0, self.num_instances) #create instance IDs for rows/instances - used to identify corresponding feature tracking scores
        else:
            instance_ids = row_id
        formatted_data = np.insert(X, self.num_feat, y, axis=1)
        formatted_data = np.insert(formatted_data, self.num_feat+1, instance_ids, axis=1)
        shuffle_order = np.random.permutation(self.num_instances)
        shuffled_data = formatted_data[shuffle_order]
        features = shuffled_data[:, :-2].tolist()
        labels = shuffled_data[:, -2].tolist()
        id = shuffled_data[:, -1].tolist()
        for i in range(len(features)):
            features[i] = [None if np.isnan(x) else x for x in features[i]]
        return [features, labels, id]


    def get_instance(self):
        return (self.instance_state,self.outcome_state)
        

    def next_instance(self):
        if self.instance_index < self.num_instances-1:
            self.instance_index += 1
            self.instance_state = self.train_data[0][self.instance_index]
            self.outcome_state = self.train_data[1][self.instance_index]
        else:
            self.reset_instance()


    def reset_instance(self):
        self.instance_index = 0
        self.instance_state = self.train_data[0][self.instance_index]
        self.outcome_state = self.train_data[1][self.instance_index]


    def report_data(self, heros):
        if heros.outcome_type == 'class':
            print("Data Manage Summary: ------------------------------------------------")
            print("Number of quantitative features: "+str(self.num_q_feat))
            print("Number of categorical features: "+str(self.num_c_feat))
            print("Total Features: "+str(self.num_feat))
            print("Total Instances: "+str(self.num_instances))
            print("Feature Types: "+str(self.feat_types))
            print("Missing Values: "+str(self.missing_values))
            print("Classes: "+str(self.classes))
            print("Class Counts: "+str(self.class_counts))
            print("Class Weights: "+str(self.class_weights))
            print("Majority Class: "+str(self.majority_class))
            print("Quantiative Feature Range: "+str(self.feat_q_range))
            print("Categorical Feature Values: "+str(self.feat_c_values))
            print("Average States: "+str(self.avg_feat_states))
            print("Rule Specificity Limit: "+str(heros.rsl))
            print("Expert Knowledge Weights Used: "+str(heros.use_ek))
            print("--------------------------------------------------------------------")
        elif heros.outcome_type == 'quant':
            print("Data Manage Summary: ------------------------------------------------")
            print("Number of quantitative features: "+str(self.num_q_feat))
            print("Number of categorical features: "+str(self.num_c_feat))
            print("Total Features: "+str(self.num_feat))
            print("Total Instances: "+str(self.num_instances))
            print("Feature Types: "+str(self.feat_types))
            print("Missing Values: "+str(self.missing_values))
            print("Outcome Range: "+str(self.outcome_range)) 
            print("Outcome Standard Deviation: "+str(self.outcome_sd)) 
            print("Quantiative Feature Range: "+str(self.feat_q_range))
            print("Categorical Feature Values: "+str(self.feat_c_values))
            print("Average States: "+str(self.avg_feat_states))
            print("Rule Specificity Limit: "+str(heros.rsl))
            print("Expert Knowledge Weights Used: "+str(heros.use_ek))
            print("Outcome Ranked: "+str(self.outcome_ranked)) 
            print("--------------------------------------------------------------------")
        else:
            pass
