import random
import numpy as np
import pandas as pd
import collections.abc
from sklearn.base import BaseEstimator, TransformerMixin
from .methods.time_tracking import TIME_TRACK
from .methods.data_mange import DATA_MANAGE
from .methods.rule_population import RULE_POP
from .methods.rule_pareto_fitness import RULE_PARETO
from .methods.feature_tracking import FEAT_TRACK
from .methods.rule_tracking import RULE_TRACK
from .methods.rule_prediction import RULE_PREDICTION
from .methods.rule_compaction import COMPACT

from .methods.model_population import MODEL_POP
from .methods.model_pareto_fitness import MODEL_PARETO
from .methods.model_prediction import MODEL_PREDICTION
import pickle #temporary testing
import inspect #temporary testing

class HEROS(BaseEstimator, TransformerMixin):
    def __init__(self, outcome_type='class',iterations=100000,pop_size=1000,cross_prob=0.8,mut_prob=0.04,nu=1,beta=0.2,theta_sel=0.5,fitness_function='pareto',
                 subsumption='both',rsl=0,feat_track='add',model_iterations=500,model_pop_size=100, model_pop_init = 'target_acc', new_gen=1.0,merge_prob=0.1,rule_pop_init=None,compaction='sub',track_performance=0,stored_rule_iterations=None,stored_model_iterations=None,random_state=None,verbose=False):
        """
        A Scikit-Learn compatible implementation of the 'Heuristic Evolutionary Rule Optimization System' (HEROS) Algorithm.
        ..
            General Parameters
        :param outcome_type: Defines the outcome type (Must be 'class' for classification, or 'quant' for quantiative outcome)
        :param iterations: The number of rule training cycles to run. (Must be nonnegative integer)
        :param pop_size: Maximum 'micro' rule population size, i.e. sum of rule numerosities. (Must be nonnegative integer)
        :param cross_prob: The probability of applying crossover in rule discovery with the genetic algorithm. (Must be float from 0 - 1)
        :param mut_prob: The probability of mutating a position within an offspring rule. (Must be float from 0 - 1)
        :param nu: Power parameter used to determine the importance of high rule-accuracy when calculating fitness. (must be non-negative)
        :param beta: Learning parameter - used in calculating average match set size (must be float from 0 - 1) 
        :param theta_sel: The fraction of the correct set to be included in tournament selection (must be float from 0 - 1)
        :param fitness_function: The fitness function used to globally evaluate rules (must be 'accuracy' or 'pareto') 
        :param subsumption: Specify subsumption strategy(s) to apply (must be 'ga', 'c', 'both', or None)
        :param rsl: Rule specificity limit, automatically determined when 0, or specified as a positive integer (Must be 0 or a positive integer)
        :param feat_track: Activates a specified feature tracking mechanism which tracks estimated feature importance for individual instances (Must be None or 'add' or 'wh' or 'end')
        :param model_iterations: The number of model training cycles to run (Must 0 or a nonnegative integer)
        :param model_pop_size: Maximum model population size. (Must be nonnegative integer)
        :param model_pop_init: model population initialization method (Must be 'random', 'probabilistic', 'bootstrap', or 'target_acc')
        :param new_gen: Proportion of maximum pop size used to generate an model offspring population each generation (must be float from 0 - 1)
        :param merge_prob: The probablity of the merge operator being used during model offspring generation (must be float from 0 - 1)
        :param rule_pop_init: Specifies type of population initiailzation (if any) (Must be 'load' or 'dt', or None)
        :param compaction: Specifies type of rule-compaciton to apply at end of rule population training (if any) (Must be 'sub' or None)
        :param track_performance: Activates performance tracking when > 0. Value indicates how many iteration steps to wait to gather tracking data (Must be 0 or a positive integer)
        :param stored_rule_iterations: specifies iterations where a copy of the rule population is stored (Must be positive integers separated by commas or None)
        :param stored_model_iterations: specifies iterations where a copy of the model population is stored (Must be positive integers separated by commas or None)
        :param random_state: the seed value needed to generate a random number
        :param verbose: Boolean flag to run in 'verbose' mode - display run details
        :param init: model population initialization method 
        """
        # Basic run parameter checks
        if not outcome_type == 'class' and not outcome_type == 'quant':
            raise Exception("'outcome' param must be 'class' or 'quant'")

        if not self.check_is_int(iterations) or iterations < 10:
            raise Exception("'iterations' param must be non-negative integer >= 10")
        
        if not self.check_is_int(pop_size) or pop_size < 50:
            raise Exception("'pop_size' param must be non-negative integer >= 50")
        
        if not self.check_is_float(cross_prob) or cross_prob < 0 or cross_prob > 1:
            raise Exception("'cross_prob' param must be float from 0 - 1")
        
        if not self.check_is_float(mut_prob) or mut_prob < 0 or mut_prob > 1:
            raise Exception("'mut_prob' param must be float from 0 - 1")
        
        if not self.check_is_float(nu) and not self.check_is_int(nu):
            raise Exception("'nu' param must be an int or float")
        if nu < 0:
            raise Exception("'nu' param must be > 0")

        if not self.check_is_float(beta) or beta < 0 or beta > 1:
            raise Exception("'beta' param must be float from 0 - 1")
        
        if not self.check_is_float(theta_sel) or theta_sel < 0 or theta_sel > 1:
            raise Exception("'theta_sel' param must be float from 0 - 1")
        
        if not fitness_function == 'accuracy' and not fitness_function == 'pareto':
            raise Exception("'fitness_function' param must be 'accuracy', or 'pareto'")

        if not subsumption == 'ga' and not subsumption == 'c' and not subsumption == 'both' and not subsumption is None and not subsumption == 'None':
            raise Exception("'subsumption' param must be 'ga', or 'c', or 'both', or None")

        if not self.check_is_int(rsl) and not rsl == 0:
            raise Exception("'rsl' param must be zero or a positive int")

        if not feat_track == 'add' and not feat_track == 'wh' and not feat_track == 'end' and not feat_track is None and not feat_track == 'None':
            raise Exception("'feat_track' param must be 'add', or 'wh', or 'end', or None")
        
        if not model_iterations == 0 and (not self.check_is_int(model_iterations) or model_iterations < 5):
            raise Exception("'model_iterations' param must be non-negative integer >= 5, or 0")

        if not self.check_is_int(model_pop_size) or model_pop_size < 20:
            raise Exception("'model_pop_size' param must be non-negative integer >= 20")
        
        if not self.check_is_float(new_gen) or new_gen < 0 or new_gen > 1:
            raise Exception("'new_gen' param must be float from 0 - 1")

        if not self.check_is_float(merge_prob) or merge_prob < 0 or merge_prob > 1:
            raise Exception("'merge_prob' param must be float from 0 - 1")

        if not rule_pop_init == 'load' and not rule_pop_init == 'dt' and not rule_pop_init is None and not rule_pop_init == 'None':
            raise Exception("'rule_pop_init' param must be 'load', 'dt', or None")

        if not compaction == 'sub' and not compaction is None and not compaction == 'None':
            raise Exception("'compaction' param must be 'sub', or None")
        
        if not self.check_is_int(track_performance) or track_performance < 0:
            raise Exception("'track_performance' param must be non-negative integer")
        
        if not self.check_is_int(random_state) and not random_state is None and not random_state == 'None':
            raise Exception("'random_state' param must be an int or None")

        if not verbose == True and not verbose == False and not verbose == 'True' and not verbose == 'False':
            raise Exception("'verbose' param must be a boolean, i.e. True or False")
        
        #Initialize global variables
        self.outcome_type = str(outcome_type)
        self.iterations = int(iterations)
        self.pop_size = int(pop_size)
        self.cross_prob = float(cross_prob)
        self.mut_prob = float(mut_prob)
        self.nu = float(nu)
        self.beta = float(beta)
        self.theta_sel = float(theta_sel)
        self.fitness_function = str(fitness_function)
        self.subsumption = str(subsumption)
        self.rsl = int(rsl)
        if feat_track == 'None' or feat_track is None:
            self.feat_track = None
        else:
            self.feat_track = str(feat_track)
        self.model_iterations = int(model_iterations)
        self.model_pop_size = int(model_pop_size)
        self.model_pop_init = model_pop_init
        self.new_gen = float(new_gen)
        self.merge_prob = float(merge_prob)
        self.rule_pop_init = str(rule_pop_init)
        if compaction == 'None' or compaction is None:
            self.compaction = None
        else:
            self.compaction = str(compaction)
        self.track_performance = int(track_performance)
        if stored_rule_iterations == 'None' or stored_rule_iterations is None:
            self.stored_rule_iterations = None
        else:
            self.stored_rule_iterations = [int(value) for value in stored_rule_iterations.split(',')]
        if stored_model_iterations == 'None' or stored_model_iterations is None:
            self.stored_model_iterations = None
        else:
            self.stored_model_iterations = [int(value) for value in stored_model_iterations.split(',')]
        if random_state == 'None' or random_state is None:
            self.random_state = None
        else:
            self.random_state = int(random_state)
        if verbose == 'True' or verbose == True:
            self.verbose = True
        if verbose == 'False' or verbose == False:
            self.verbose = False

        self.use_ek = False #internal parameter - set to False by default, but switched to true of ek scores passed to fit()
        self.y_encoding = None
        self.top_models = [] #for tracking model performance increase over iterations

    @staticmethod
    def check_is_int(num):
        """
        :meta private:
        """
        return isinstance(num, int)

    @staticmethod
    def check_is_float(num):
        """
        :meta private:
        """
        return isinstance(num, float)

    @staticmethod
    def check_is_list(num):
        """
        :meta private:
        """
        return isinstance(num, list)
    
    def check_inputs(self, X, y, row_id, cat_feat_indexes, pop_df, ek): 
        """
        Function to check if X, y, pop_df, and ek inputs to fit are valid.
        
        :param X: None or array-like {n_samples, n_features} Training instances of quantitative features.
        :param y: array-like {n_samples} Training labels of the outcome variable.
        :param row_id: array-like{n_samples} Unique instance lables for each row
        :param cat_feat_indexes: array-like max({n_features}) A list of feature indexes 
                in 'X' that should be treated as categorical variables (all others treated 
                as quantitative). An empty list or None indicates all features should be 
                treated as quantitative.
        :param pop_df: None or pandas data frame of HEROS-formatted rule population
        :param ek: None or np.ndarray or list
        """
        # Validate list of instance ID's 

        self.row_id = row_id
        if not self.check_is_list(self.row_id) and not self.row_id is None:
            unique_count = len(set(row_id))
            if len(row_id) != unique_count:
                raise Exception("'row_id' param must be a list of unique row/instance identifiers or None")
            
        # Validate list of feature indexes to treat as categorical
        if cat_feat_indexes == 'None' or cat_feat_indexes is None:
            self.cat_feature_indexes = None
        else:
            self.cat_feature_indexes = cat_feat_indexes
        if not self.check_is_list(cat_feat_indexes) and not cat_feat_indexes is None:
            raise Exception("'cat_feat_indexes' param must be a list of integer column indexes (for 'X') or None")
        if self.check_is_list(cat_feat_indexes):
            for each in cat_feat_indexes:
                if not self.check_is_int(each):
                    raise Exception("All values in 'cat_feat_indexes' must be an integer that is a column index in 'X'")
        # Validate the prior HEROS rule population for algorithm initialization
        if not isinstance(pop_df, pd.DataFrame) and not pop_df is None:
            raise Exception("'pop_df' param must be either None or a DataFame that is formatted to store a HEROS rule population")
        if self.rule_pop_init == 'load' and not isinstance(pop_df, pd.DataFrame):
            raise Exception("'pop_df' must be provided to fit() when rule_pop_init = 'load'")
        if self.rule_pop_init != 'load' and not pop_df is None:
            raise Exception("'pop_df' provided but rule_pop_init was not set to 'load'")
        # Validate expert knowledge scores (if specified)
        if not (isinstance(ek, np.ndarray)) and not (isinstance(ek, list)) and not ek is None:
            raise Exception("'ek' param must be None or list/ndarray")
        if isinstance(ek,np.ndarray):
            ek = ek.tolist()
        # Validate the feature data (X)
        if X is not None and X is not isinstance(X, (collections.abc.Sequence, np.ndarray)):
           pass # FIX
           #raise TypeError("X must be a numpy arraylike.")
        if y is not isinstance(y, (collections.abc.Sequence, np.ndarray)):
           pass #FIX
           #raise TypeError("y must be included and be arraylike")
        # Ensure numerical data in X
        if X is not None: # TO FIX - this should also allow missing data!
            X = np.array(X)
            if not np.isreal(X).all():
                raise ValueError("All values in X must be numeric.")
        # Ensure numerical data in y and handle categorical text values
        if y is not None:
            y = np.array(y, dtype=object)  # Use object dtype to handle mixed types
            if isinstance(y[0], str):
                unique_categories, encoded = np.unique(y, return_inverse=True)
                self.y_encoding = dict(enumerate(unique_categories))
                y = encoded
            else:
                if self.outcome_type == 'class':
                    y = y.astype(int)
                elif self.outcome_type == 'quant':
                    y = y.astype(float)
                else:
                    pass
            if not np.isreal(y).all():
                raise ValueError("All values in y must be numeric after encoding.")
        return X, y, row_id, cat_feat_indexes, pop_df, ek

    def fit(self, X, y, row_id=None, cat_feat_indexes=None, pop_df=None, ek=None):
        """
        Scikit-learn required function for supervised training of HEROS

        :param X: None or array-like {n_samples, n_features} Training instances.
                ALL INSTANCE FEATURES MUST BE NUMERIC OR NAN
        :param y: array-like {n_samples} Training labels (outcome). 
                ALL INSTANCE OUTCOME LABELS MUST BE NUMERIC NOT NAN OR OTHER TYPE
        :param row_id: array-like {n_samples} Row/instance labels
                ALL INSTANCE LABELS MUST BE NUMERIC OR STRING NOT NAN OR OTHER TYPE
        :param cat_feat_indexes: array-like max({n_features}) A list of feature indexes 
                in 'X' that should be treated as categorical variables (all others treated 
                as quantitative). An empty list or None indicates all features should be 
                treated as quantitative.
                ALL FEATURE INDEXES MUST BE NUMERIC INTEGERS

        :param pop_df: None or dataframe of HEROS-formatted rule population
                DATAFRAME MUST CONFORM TO HEROS RULE POPULATION FORMAT
        :param ek: None or np.ndarray or list. Feature expert knowlege weights.

        :return: self
        """  
        self.timer = TIME_TRACK()
        # (HEROS PHASE 1) ALGORITHM INITIALIZATION ***********************************************************
        self.timer.phase1_time_start()
        self.timer.init_time_start() #initialization time tracking
        random.seed(self.random_state) # Set random seed/state
        # Data Preparation
        X, y, row_id, cat_feat_indexes, pop_df, ek = self.check_inputs(X, y, row_id, cat_feat_indexes, pop_df, ek) #check loaded data
        self.env = DATA_MANAGE(X, y, row_id, cat_feat_indexes, ek, self) #initialize the data environment; data formatting, summary statistics, and expert knowledge preparation
        # Memory Cleanup
        X = None
        y = None
        # Initialize Objects
        self.iteration = 0
        self.rule_population = RULE_POP() # Initialize rule sets
        if self.fitness_function == 'pareto':
            self.rule_pareto = RULE_PARETO()
        else:
            self.rule_pareto = None
        if self.feat_track != None:
            self.FT = FEAT_TRACK(self)
        else:
            self.FT = None
        
        # Initialize Learning Performance Tracking Objects
        self.tracking = RULE_TRACK(self)

        # Initialize Rule Population (if specified)
        if self.rule_pop_init == 'load': # Initialize rule population based on loaded rule population
            self.rule_population.load_rule_population(pop_df,self)
        elif self.rule_pop_init == 'dt': # Train and utilize decision tree models to initialize rule population (based on individual tree 'branches')
            print("To implement")
            pass
        else: # No rule population initialization other than standard LCS-algorithm-style 'covering' mechanism.
            pass
        self.timer.init_time_stop() #initialization time tracking
        # RUN RULE-LEARNING TRAINING ITERATIONS **************************************************************
        while self.iteration < self.iterations:
            # Get current training instance
            instance = self.env.get_instance()
            #print('Iteration: '+str(self.iteration)+' RulePopSize: '+str(len(self.rule_population.pop_set)))
            # Run a single training iteration focused on the current training instance
            self.run_iteration(instance)
            # Evaluation tracking ***************************************************
            if (self.iteration + 1) % self.track_performance == 0:
                self.tracking.update_performance_tracking(self.iteration,self)
                if self.verbose:
                    self.tracking.print_tracking_entry()
            #Pause learning to conduct a complete evaluation of the current rule population
            if self.stored_rule_iterations != None and (self.iteration + 1) in self.stored_rule_iterations:
                #Archive current rule population
                print('archiving'+str(self.iteration+1))
                self.rule_population.archive_rule_pop(self.iteration+1)
                self.timer.archive_rule_pop(self.iteration+1)
            # Increment iteration and training instance
            self.iteration += 1
            self.env.next_instance()

        # RULE COMPACTION *********************************************
        self.timer.compaction_time_start()
        compact = COMPACT(self)
        compact.basic_rule_cleaning(self)
        if self.compaction == 'sub':
            compact.subsumption_compation(self)
        self.timer.compaction_time_stop()

        # BATCH FEATURE TRACKING **************************************
        if self.feat_track == 'end':
            self.FT.batch_calculate_ft_scores(self)
        self.timer.phase1_time_stop()
        print("HEROS (Phase 1) run complete!")

        # (HEROS PHASE 2) RUN RULE-SET-LEARNING TRAINING ITERATIONS  ********************************************************************
        self.timer.phase2_time_start()
        if self.model_iterations != 0:
            # Initialize model population and 
            self.model_population = MODEL_POP() # Initialize rule sets
            if self.fitness_function == 'pareto':
                self.model_pareto = MODEL_PARETO()
            else:
                self.model_pareto = None
            self.model_iteration = 0
            self.model_population.initialize_model_population(self,random,self.model_pop_init)

            # RUN MODEL-LEARNING TRAINING ITERATIONS **************************************************************
            while self.model_iteration < self.model_iterations:
                print("Iteration: "+str(self.model_iteration))
                # GENETIC ALGORITHM 
                target_offspring_count = int(self.model_pop_size*self.new_gen) #Determine number of offspring to generate
                front_updated_global = False
                while len(self.model_population.offspring_pop) < target_offspring_count: #Generate offspring until we hit the target number
                    # Parent Selection
                    parent_list = self.model_population.select_parent_pair(self.theta_sel,random)
                    # Generate Offspring - clone, crossover, mutation, evaluation, add to population
                    front_updated = self.model_population.generate_offspring(self.model_iteration,parent_list,random,self)
                    if not front_updated_global: # check if the front was updated ever during the generation of this offspring population
                        front_updated_global = front_updated
                # Add Offspring Models to Population
                self.model_population.add_offspring_into_pop()
                if self.fitness_function == 'pareto' and front_updated_global:
                    self.model_population.global_fitness_update(self)
                #Bin Deletion
                self.model_population.probabilistic_model_deletion(self,random)
                #Model Performance Tracking
                self.top_models.append(self.model_population.get_max())
                #Pause learning to conduct a complete evaluation of the current rule population
                if self.stored_model_iterations != None and (self.model_iteration + 1) in self.stored_model_iterations:
                    #Archive current rule population
                    self.model_population.sort_model_pop()
                    self.model_population.identify_models_on_front(self) #For evaluating all models on the front.
                    self.model_population.archive_model_pop(self.model_iteration+1)
                    self.timer.archive_model_pop(self.model_iteration+1)
                #Next Iteration
                self.model_iteration += 1
            #Sort the model population first by accuracy and then by number of rules in model.
            self.model_population.sort_model_pop()
            self.model_population.identify_models_on_front(self) #For evaluating all models on the front.
            self.model_population.get_target_model(0) #the 'model' object with the best accuracy, then coverage, then lowest rule count (assumes prior sorting)
            self.timer.phase2_time_stop()
            self.timer.update_global_time()
            print("HEROS (Phase 2) run complete!")
            print("Random Seed Check - End: "+ str(random.random()))
        self.env.clear_data_from_memory()
        return self


    def run_iteration(self,instance):
        # Make 'Match Set', {M}
        self.rule_population.make_match_set(instance,self,random)

        # Track Training Accuracy
        outcome_prediction = None
        if self.track_performance > 0:
            self.timer.prediction_time_start()
            prediction = RULE_PREDICTION(self, self.rule_population)
            outcome_prediction = prediction.get_prediction()
            self.tracking.update_prediction_list(outcome_prediction,instance[1],self)
            self.timer.prediction_time_stop()

        # Make 'Correct Set', {C}
        self.rule_population.make_correct_set(instance[1],self) #passed the instance outcome only

        # Update Rule Parameters
        self.timer.rule_eval_time_start()
        self.rule_population.update_rule_parameters(self)
        self.timer.rule_eval_time_stop()

        # Correct Set Subsumption (New implementation)
        if self.subsumption == 'c' or self.subsumption == 'both':
            self.timer.subsumption_time_start()
            self.rule_population.correct_set_subsumption(self)
            self.timer.subsumption_time_stop()

        # Update Feature Tracking
        if self.feat_track == 'add':
            self.timer.feature_track_time_start()
            self.FT.update_ft_scores(outcome_prediction,instance[1],self)
            self.timer.feature_track_time_stop()
        elif self.feat_track == 'wh':
            self.timer.feature_track_time_start()
            self.FT.update_ft_scores_wh(outcome_prediction,instance[1],self)
            self.timer.feature_track_time_stop()

        # Apply Genetic Algorithm To Generate Offspring Rules
        self.rule_population.genetic_algorithm(instance,self,random)

        # Apply Rule Deletion
        self.rule_population.deletion(self,random)

        #Clear Match and Correct Sets
        self.rule_population.clear_sets()


    def predict(self, X, whole_rule_pop=False, target_model=0, rule_pop_iter=None, model_pop_iter=None):
        """Scikit-learn required: Apply trained model to predict outcomes of instances. 
        Applicable to both classification and regression (i.e. quantitative outcome prediction).

        Parameters
        :param X: array-like {n_samples, n_features} Training or Testing instances upon which to make predictions. ALL INSTANCE 
            ATTRIBUTES MUST BE NUMERIC OR NAN AND MUST BE ORDERED IN THE SAME MANNER AS THE TRAINING DATA PASSED TO FIT
        :param rule_set: boolean (True or False) Determines if prediction is made by rule set (True) or the whole rule population (False)

        Returns
        :param y: array-like {n_samples} Outcome predictions. 
                ALL INSTANCE OUTCOME LABELS MUST BE NUMERIC NOT NAN OR OTHER TYPE
        """
        # Prediction Data Checks ************************
        for instance in X:
            for value in instance:
                if np.isnan(value):
                    value = None
                elif not self.check_is_float(value) and self.check_is_int(value):
                    raise Exception("X must be fully numeric")
        # Initialize Key Objects ************************
        num_instances = X.shape[0]
        prediction_list = []
        # Apply Prediction ******************************
        if whole_rule_pop:
            if rule_pop_iter != None:
                self.rule_population.change_rule_pop(rule_pop_iter)
            # Whole final rule population used to make predictions based on standard LCS voting scheme
            for instance in range(num_instances):
                instance_state = X[instance]
                self.rule_population.make_eval_match_set(instance_state,self)
                prediction = RULE_PREDICTION(self, self.rule_population)
                outcome_prediction = prediction.get_prediction()
                prediction_list.append(outcome_prediction)
                self.rule_population.clear_sets()
            if rule_pop_iter != None:    
                self.rule_population.restore_rule_pop()
        else:
            if model_pop_iter != None:
                self.model_population.change_model_pop(model_pop_iter)
            self.model_population.get_target_model(target_model)
            #Top performing model (i.e. rule-set) is used to make predictions based on standard LCS voting scheme
            for instance in range(num_instances):
                instance_state = X[instance]
                self.model_population.make_eval_match_set(instance_state,self)
                prediction = MODEL_PREDICTION(self, self.model_population)
                outcome_prediction = prediction.get_prediction()
                prediction_list.append(outcome_prediction)
                self.model_population.clear_sets()
            if model_pop_iter != None:    
                self.model_population.restore_model_pop()
        return np.array(prediction_list)
    

    def predict_proba(self, X, whole_rule_pop=False, target_model=0, rule_pop_iter=None, model_pop_iter=None):
        """Scikit-learn required: Apply trained model to get class prediction probabilities for instances. 
            Applicable to both classification and regression (i.e. quantitative outcome prediction).

        Parameters
        :param X: array-like {n_samples, n_features} Training or Testing instances upon which to make predictions. ALL INSTANCE 
            ATTRIBUTES MUST BE NUMERIC OR NAN AND MUST BE ORDERED IN THE SAME MANNER AS THE TRAINING DATA PASSED TO FIT
        :param rule_set: boolean (True or False) Determines if prediction is made by rule set (True) or the whole rule population (False)

        Returns
        :param y: array-like {n_samples, n_classes} Outcome class prediction probabilities. 
                ALL INSTANCE OUTCOME LABELS MUST BE NUMERIC NOT NAN OR OTHER TYPE
        """
        # Prediction Data Checks ************************
        for instance in X:
            for value in instance:
                if np.isnan(value):
                    value = None
                elif not self.check_is_float(value) and self.check_is_int(value):
                    raise Exception("X must be fully numeric")
        # Initialize Key Objects ************************
        num_instances = X.shape[0]
        prediction_proba_list = []
        # Apply Prediction ******************************
        if whole_rule_pop:
            if rule_pop_iter != None:
                self.rule_population.change_rule_pop(rule_pop_iter)
            # Whole final rule population used to make predictions based on standard LCS voting scheme
            for instance in range(num_instances):
                instance_state = X[instance]
                self.rule_population.make_eval_match_set(instance_state,self)
                prediction = RULE_PREDICTION(self, self.rule_population)
                outcome_proba = prediction.get_prediction_proba()
                prediction_proba_list.append(outcome_proba)
                self.rule_population.clear_sets()
            if rule_pop_iter != None:    
                self.rule_population.restore_rule_pop()
        else:
            if model_pop_iter != None:
                self.model_population.change_model_pop(model_pop_iter)
            self.model_population.get_target_model(target_model)
            #Top performing model (i.e. rule-set) is used to make predictions based on standard LCS voting scheme
            for instance in range(num_instances):
                instance_state = X[instance]
                self.model_population.make_eval_match_set(instance_state,self)
                prediction = MODEL_PREDICTION(self, self.model_population)
                outcome_proba = prediction.get_prediction_proba()
                prediction_proba_list.append(outcome_proba)
                self.model_population.clear_sets()
            if model_pop_iter != None:    
                self.model_population.restore_model_pop()
        return np.array(prediction_proba_list)

    def predict_ranges(self, X, whole_rule_pop=False, target_model=0, rule_pop_iter=None, model_pop_iter=None):
        """Scikit-learn required: Apply trained model to get prediction probabilities for instances.
        Applicable only to regression (i.e. quantitative outcome prediction).

        Parameters
        :param X: array-like {n_samples, n_features} Training or Testing instances upon which to make predictions. ALL INSTANCE 
            ATTRIBUTES MUST BE NUMERIC OR NAN AND MUST BE ORDERED IN THE SAME MANNER AS THE TRAINING DATA PASSED TO FIT
        :param rule_set: boolean (True or False) Determines if prediction is made by rule set (True) or the whole rule population (False)

        Returns
        :param y: array-like {n_samples, n_quantiative_outcome_ranges} Outcome range predictions. 
                ALL INSTANCE OUTCOME LABELS MUST BE NUMERIC NOT NAN OR OTHER TYPE
        """
        # Prediction Data Checks ************************
        for instance in X:
            for value in instance:
                if np.isnan(value):
                    value = None
                elif not self.check_is_float(value) and self.check_is_int(value):
                    raise Exception("X must be fully numeric")
        # Initialize Key Objects ************************
        num_instances = X.shape[0]
        prediction_range_list = []
        # Apply Prediction ******************************
        if whole_rule_pop:
            if rule_pop_iter != None:
                self.rule_population.change_rule_pop(rule_pop_iter)
            # Whole final rule population used to make predictions based on standard LCS voting scheme
            for instance in range(num_instances):
                instance_state = X[instance]
                self.rule_population.make_eval_match_set(instance_state,self)
                prediction = RULE_PREDICTION(self, self.rule_population)
                outcome_range = prediction.get_prediction_range()
                prediction_range_list.append(outcome_range)
                self.rule_population.clear_sets()
            if rule_pop_iter != None:    
                self.rule_population.restore_rule_pop()
        else:
            if model_pop_iter != None:
                self.model_population.change_model_pop(model_pop_iter)
            self.model_population.get_target_model(target_model)
            #Top performing model (i.e. rule-set) is used to make predictions based on standard LCS voting scheme
            for instance in range(num_instances):
                instance_state = X[instance]
                self.model_population.make_eval_match_set(instance_state,self)
                prediction = MODEL_PREDICTION(self, self.model_population)
                outcome_range = prediction.get_prediction_range()
                prediction_range_list.append(outcome_range)
                self.model_population.clear_sets()
            if model_pop_iter != None:    
                self.model_population.restore_model_pop()
        return np.array(prediction_range_list)

    def predict_covered(self, X, whole_rule_pop=False, target_model=0, rule_pop_iter=None, model_pop_iter=None):
        """Scikit-learn required: Apply trained model to get class prediction probabilities for instances.
        Applicable only to regression (i.e. quantitative outcome prediction).

        Parameters
        :param X: array-like {n_samples, n_features} Training or Testing instances upon which to make predictions. ALL INSTANCE 
            ATTRIBUTES MUST BE NUMERIC OR NAN AND MUST BE ORDERED IN THE SAME MANNER AS THE TRAINING DATA PASSED TO FIT
        :param rule_set: boolean (True or False) Determines if prediction is made by rule set (True) or the whole rule population (False)

        Returns
        :param y: array-like {n_samples} Boolean covering indicator (1 = instance covered by at least one rule, 0 = rule not covered). 
                ALL INSTANCE OUTCOME LABELS MUST BE NUMERIC NOT NAN OR OTHER TYPE
        """
        # Prediction Data Checks ************************
        for instance in X:
            for value in instance:
                if np.isnan(value):
                    value = None
                elif not self.check_is_float(value) and self.check_is_int(value):
                    raise Exception("X must be fully numeric")
        # Initialize Key Objects ************************
        num_instances = X.shape[0]
        prediction_covered_list = []
        # Apply Prediction ******************************
        if whole_rule_pop:
            if rule_pop_iter != None:
                self.rule_population.change_rule_pop(rule_pop_iter)
            # Whole final rule population used to make predictions based on standard LCS voting scheme
            for instance in range(num_instances):
                instance_state = X[instance]
                self.rule_population.make_eval_match_set(instance_state,self)
                prediction = RULE_PREDICTION(self, self.rule_population)
                outcome_coverage = prediction.get_if_covered()
                prediction_covered_list.append(outcome_coverage)
                self.rule_population.clear_sets()
            if rule_pop_iter != None:    
                self.rule_population.restore_rule_pop()
        else:
            if model_pop_iter != None:
                self.model_population.change_model_pop(model_pop_iter)
            self.model_population.get_target_model(target_model)
            #Top performing model (i.e. rule-set) is used to make predictions based on standard LCS voting scheme
            for instance in range(num_instances):
                instance_state = X[instance]
                self.model_population.make_eval_match_set(instance_state,self)
                prediction = MODEL_PREDICTION(self, self.model_population)
                outcome_coverage = prediction.get_if_covered()
                prediction_covered_list.append(outcome_coverage)
                self.model_population.clear_sets()
            if model_pop_iter != None:    
                self.model_population.restore_model_pop()
        return np.array(prediction_covered_list)

    def get_pop(self):
        """ Return a dataframe of the rule population. """
        self.rule_population.order_rule_conditions()
        pop_df = self.rule_population.export_rule_population()
        return pop_df
    
    def get_ft(self,feature_names):
        """ Return a dataframe of the ft scores. """
        ft_df = self.FT.export_ft_scores(self,feature_names)
        return ft_df 
    
    def get_model_pop(self):
        """ Return a dataframe of the model population. """
        pop_df = self.model_population.export_model_population()
        return pop_df
    
    def get_model_rules(self,index=0):
        """ Return a dataframe of the top model rule-set. """
        set_df = self.model_population.export_indexed_model(index)
        return set_df

    def get_rule_pop_heatmap(self,feature_names, weighting, specified_filter, display_micro, show, save, output_path):
        """ """
        self.rule_population.plot_rule_pop_heatmap(feature_names, self, weighting, specified_filter, display_micro, show, save, output_path)

    def get_rule_pop_network(self, feature_names, weighting, display_micro, node_size, edge_size, show, save, output_path):
        """ """
        self.rule_population.plot_rule_pop_network(feature_names, weighting, display_micro, node_size, edge_size, show, save, output_path)

    def get_rule_pareto_landscape(self,resolution, rule_population, plot_rules, color_rules, show, save, output_path):
        """ """
        self.rule_pareto.plot_pareto_landscape(resolution, rule_population, plot_rules, color_rules, self, show, save, output_path)

    def get_clustered_ft_heatmap(self,feature_names, show, save, output_path):
        ft_df = self.FT.export_ft_scores(self,feature_names)
        print(ft_df)
        ft_df = ft_df.drop('row_id', axis=1)
        self.FT.plot_clustered_ft_heatmap(ft_df, feature_names, show, save, output_path)

    def get_performance_tracking(self):
        return self.tracking.get_performance_tracking_df()
    
    def get_model_pareto_landscape(self,resolution, rule_population, plot_rules, show, save, output_path):
        """ """
        self.model_pareto.plot_pareto_landscape(resolution, rule_population, plot_rules, self, show, save, output_path)
    
    def get_runtimes(self):
        return self.timer.report_times()
    
    def export_model_growth(self):
        """ Prepares and exports a dataframe capturing the top models at each iteration. """
        pop_list = []
        column_names = ['Rule IDs', 
                        'Number of Rules',
                        'Fitness', 
                        'Accuracy',
                        'Coverage', 
                        'Birth Iteration', 
                        'Deletion Probability']
        for model in self.top_models: 
            model_list = [str(model.rule_IDs), 
                          len(model.rule_set), 
                          model.fitness, 
                          model.accuracy,
                        model.coverage, 
                        model.birth_iteration, 
                        model.deletion_prob]
            pop_list.append(model_list)
        pop_df = pd.DataFrame(pop_list, columns = column_names)
        return pop_df

    def save_run_params(self,filename):
        with open(filename, 'w') as file:
            file.write(f"outcome_type: {self.outcome_type}\n")
            file.write(f"iterations: {self.iterations}\n")
            file.write(f"pop_size: {self.pop_size}\n")
            file.write(f"cross_prob: {self.cross_prob}\n")
            file.write(f"mut_prob: {self.mut_prob}\n")
            file.write(f"nu: {self.nu}\n")
            file.write(f"beta: {self.beta}\n")
            file.write(f"theta_sel: {self.theta_sel}\n")
            file.write(f"fitness_function: {self.fitness_function}\n")
            file.write(f"subsumption: {self.subsumption}\n")
            file.write(f"use_ek: {self.use_ek}\n")
            file.write(f"rsl: {self.rsl}\n")
            file.write(f"feat_track: {self.feat_track}\n")
            file.write(f"model_iterations: {self.model_iterations}\n")
            file.write(f"model_pop_size: {self.model_pop_size}\n")
            file.write(f"model_pop_init: {self.model_pop_init}\n")
            file.write(f"new_gen: {self.new_gen}\n")
            file.write(f"merge_prob: {self.merge_prob}\n")
            file.write(f"rule_pop_init: {self.rule_pop_init}\n")
            file.write(f"compaction: {self.compaction}\n")
            file.write(f"track_performance: {self.track_performance}\n")
            file.write(f"stored_rule_iterations: {self.stored_rule_iterations}\n")
            file.write(f"stored_model_iterations: {self.stored_model_iterations}\n")
            file.write(f"random_state: {self.random_state}\n")
            file.write(f"verbose: {self.verbose}\n")
            if self.use_ek:
                file.write(f"ek_weights: {self.env.ek_weights}\n")
  