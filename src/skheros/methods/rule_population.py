import copy
import pandas as pd
import ast
from skheros.methods.rule import RULE
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage #, dendrogram, leaves_list
import networkx as nx
from collections import defaultdict
from itertools import combinations
import struct
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree, DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from sklearn import tree as sktree
from matplotlib.table import Table
from textwrap import fill
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

class RULE_POP:
    def __init__(self):
        """ Initializes rule population objects. """
        self.pop_set = []  # List rule objects making up the rule population
        self.match_set = []  # List of references to rules in population that make up a temporary match set (i.e. rules with 'IF' conditions matching current instance state)
        self.correct_set = []  # List of references to rules in population that make up correct set (i.e. rules with 'THEN' action matching current instance outcome)
        self.micro_pop_count = 0 # Number of rules in the population defined by the sum of individual rule numerosities (aka 'micro' population count)
        self.ID_counter = 0 # A unique id given to each new rule discovered (that isn't in the current rule population).
        self.pop_set_archive = {}
        self.pop_set_hold = None
        #Experimental
        self.explored_rules = {}
        self.archive_discovered_rules = False #True value is experimental


    def add_new_explored_rules(self,rule, heros):
        """Stores the unique and essential information to reconstitute an explored rule without re-evaluation."""
        #rule_entry = [rule.condition_indexes, rule.condition_values, rule.action, rule.instance_outcome_count, rule.ID, rule.birth_iteration]
        #self.explored_rules.append(rule_entry)
        self.explored_rules[rule.encoding] = rule.instance_outcome_count


    def clear_explored_rules(self):
        self.explored_rules = {}


    def rule_exists(self, target_rule, heros):
        """Checks the explored rules list to see if a given 'new' rule has been previously discovered and evaluated, returning that rule's reference in explored rules."""
        encoded = target_rule.encode_rule_binary(heros.env.num_feat)
        if encoded in self.explored_rules:
            return self.decode_rule_binary(encoded, heros.env.num_feat)
        return None
    

    def equals(self,target_rule,rule_summary):
        if sorted(target_rule.condition_indexes) == sorted(rule_summary[0]):
            for i in range(len(target_rule.condition_indexes)): #final check of rule equality (condition_values)
                position = rule_summary[0].index(target_rule.condition_indexes[i])
                if not (target_rule.condition_values[i] == rule_summary[1][position]):
                    return False
            return True
        return False
    

    def archive_rule_pop(self,iteration):
        self.pop_set_archive[int(iteration)] = copy.deepcopy(self.pop_set)


    def change_rule_pop(self,iteration):
        self.pop_set_hold = copy.deepcopy(self.pop_set)
        self.pop_set = self.pop_set_archive[int(iteration)]


    def restore_rule_pop(self):
        self.pop_set = self.pop_set_hold
        self.pop_set_hold = None


    def make_match_set(self, instance,heros,random,np):
        """ Makes a match set {M} and activates covering as needed to initialize the population. """
        # MATCHING ****************************************************
        heros.timer.matching_time_start() #matching time tracking
        instance_state = instance[0] #instance feature values
        outcome_state = instance[1] #instance outcome value
        do_covering = True 
        set_numerosity_sum = 0
        for i in range(len(self.pop_set)):
            rule = self.pop_set[i]
            if rule.match(instance_state,heros):
                self.match_set.append(i) #adds index to rule in pop_set
                set_numerosity_sum += rule.numerosity
                if heros.outcome_type == 'class':
                    if rule.action == outcome_state: #if at least one correct/matching rule is found in the population, covering not applied
                        do_covering = False
                elif heros.outcome_type == 'quant':
                    if rule.action[0] <= outcome_state <= rule.action[1]:
                        do_covering = False
        heros.timer.matching_time_stop() #matching time tracking
        # COVERING ****************************************************
        # While HEROS covering is not guaranteed to create a rule with the current instance class, it is activated whenever the correct set would be empty
        heros.timer.covering_time_start() #covering time tracking
        if do_covering:
            new_rule = RULE(heros)
            new_rule.initialize_by_covering(set_numerosity_sum+1,instance_state,outcome_state,heros,random,np)
            #self.debug_confirm_offspring_match(new_rule, instance,heros,'covering',None)
            if len(new_rule.condition_indexes) > 0: #prevents completely general rules from being added to the population
                #Check for duplicate rule in {P} - important since covering runs if {C} is empty, which can generate an existing rule in the match set
                if self.archive_discovered_rules:
                    rule_summary = self.rule_exists(new_rule,heros)
                    if rule_summary == None:
                        self.evaluate_covered_rule(new_rule,outcome_state,heros,random)
                    else:
                        new_rule.reestablish_rule(rule_summary,heros)
                else:
                    self.evaluate_covered_rule(new_rule,outcome_state,heros,random)
                if self.no_identical_rule_exists(new_rule,heros,'match_set'):
                    self.add_rule_to_pop(new_rule,heros)
                    self.match_set.append(len(self.pop_set)-1)
        heros.timer.covering_time_stop() #covering time tracking


    def make_eval_match_set(self,instance_state,heros):
        """ Makes a match set {M} given an instance state. Used by predict function."""
        for i in range(len(self.pop_set)):
            rule = self.pop_set[i]
            if rule.match(instance_state,heros):
                self.match_set.append(i)


    def global_fitness_update(self,heros):
        """ Relevant for pareto-front rule fitness. Updates the fitness of all rules in the population if the pareto front gets updated. """
        for rule in self.pop_set:
            rule.update_rule_fitness(heros)


    def correct_set_subsumption(self,heros):
        """ Applies correct set subsumption. The most general and accurate rule in the correct set is given the opportunity to subsume the others."""
        # Find highest rule accuracy in correct set
        candidate_subsumer = None
        rule_accuracy_list = []
        for rule_index in self.correct_set:
            rule = self.pop_set[rule_index]
            rule_accuracy_list.append(rule.accuracy)
        max_accuracy = max(rule_accuracy_list)
        # Identify the most accurate and general rule in the correct set
        for rule_index in self.correct_set:
            rule = self.pop_set[rule_index]
            if candidate_subsumer is None:
                if rule.accuracy == max_accuracy:
                    candidate_subsumer = rule
            else:
                if rule.accuracy == max_accuracy and rule.is_more_general(candidate_subsumer,heros):
                    if heros.outcome_type == 'quant': #additional 'more general' check for quantitative outcomes
                        if rule.action[0] <= candidate_subsumer.action[0] and rule.action[1] >= candidate_subsumer.action[1]:
                            candidate_subsumer = rule
                    else: #class outcome
                        candidate_subsumer = rule
        # Check if the target 'subsumer' subsumes any other 
        if candidate_subsumer != None:
            i = 0
            while i < len(self.correct_set):
                rule_index = self.correct_set[i]
                good_check = True
                if heros.outcome_type == 'quant':
                    if candidate_subsumer.action[0] > self.pop_set[rule_index].action[0] or candidate_subsumer.action[1] < self.pop_set[rule_index].action[1]:
                        good_check = False
                if good_check and candidate_subsumer.is_more_general(self.pop_set[rule_index],heros):
                    candidate_subsumer.update_numerosity(self.pop_set[rule_index].numerosity)
                    self.remove_macro_rule(rule_index)
                    self.remove_from_match_set(rule_index)
                    self.remove_from_correct_set(rule_index)
                    i -= 1
                i += 1


    def remove_from_match_set(self,rule_index):
        """ Delete reference to rule in population, contained in self.match_set."""
        if rule_index in self.match_set:
            self.match_set.remove(rule_index)
        for j in range(len(self.match_set)):
            ref = self.match_set[j]
            if ref > rule_index:
                self.match_set[j] -= 1


    def remove_from_correct_set(self,rule_index):
        """ Delete reference to rule in population, contained in self.correct_set."""
        if rule_index in self.correct_set:
            self.correct_set.remove(rule_index)
        for j in range(len(self.correct_set)):
            ref = self.correct_set[j]
            if ref > rule_index:
                self.correct_set[j] -= 1


    def debug_confirm_offspring_match(self, rule, instance,heros,step,parent_list):
        instance_state = instance[0] #instance feature values
        outcome_state = instance[1] #instance outcome value
        if not rule.match(instance_state,heros):
            print("------------------------------------------------")
            print("Generated rule failed to match current instance! "+str(step))
            print("Failed Offspring-----------------")
            print(rule.condition_indexes)
            print(rule.condition_values)
            print(rule.action)
            print("True instance states-----------------")
            temp_val_list = []
            for each in rule.condition_indexes:
                temp_val_list.append(instance_state[each])
            print(temp_val_list)
            print("Parents-----------------")
            print(parent_list[0].condition_indexes)
            print(parent_list[0].condition_values)
            print(parent_list[0].action)
            print(parent_list[1].condition_indexes)
            print(parent_list[1].condition_values)
            print(parent_list[1].action)
            print(1/0)


    def genetic_algorithm(self, instance, heros, random,np):
        instance_state = instance[0] #instance feature values
        outcome_state = instance[1] #instance outcome value
        # PARENT SELECTION *****************************************
        heros.timer.selection_time_start() #parent selection time tracking
        parent_list = self.tournament_selection(heros,random)
        heros.timer.selection_time_stop() #parent selection time tracking

        # INITIALIZE OFFSPRING *************************************
        heros.timer.mating_time_start() #mating time tracking
        offspring_list = []
        for parent_rule in parent_list:
            new_rule = RULE(heros)
            new_rule.initialize_by_parent(parent_rule,heros)
            offspring_list.append(new_rule)

        # CROSSOVER OPERATOR **************************************
        if len(offspring_list) > 1: #crossover only applied between two parent rules
            if random.random() < heros.cross_prob:
                offspring_list[0].uniform_crossover(offspring_list[1],heros,random,np)
        #for offspring in offspring_list: #debug
        #    self.debug_confirm_offspring_match(offspring, instance,heros,'crossover',parent_list)

        # MUTATION OPERATOR ***************************************
        for offspring in offspring_list:
            offspring.mutation(instance_state,outcome_state,heros,random,np)
        #for offspring in offspring_list: #debug
        #    self.debug_confirm_offspring_match(offspring, instance,heros,'mutation',parent_list)
        heros.timer.mating_time_stop() #mating time tracking

        #Check for offspring duplication
        if len(offspring_list) > 1:
            if offspring_list[0].equals(offspring_list[1]): 
                offspring_list.pop()
                #print('This happened')
                #print("Random Seed Check post GA - equal offspring: "+ str(random.random()))
                if len(offspring_list) > 1:
                    print("ERROR: More than 2 expected offspring in GA")

        # CHECK FOR DUPLICATE RULES IN {P} and EVALUATE Non-Duplicate Ruels
        front_updated = False
        final_offspring_list = []
        for offspring in offspring_list:
            if self.archive_discovered_rules:
                rule_summary = self.rule_exists(offspring,heros)
                if rule_summary == None:
                    heros.timer.rule_eval_time_start() #rule evaluation time tracking
                    front_changed = self.evaluate_offspring_rule(offspring,outcome_state,heros,random)
                    if front_changed:
                        front_updated = True
                    heros.timer.rule_eval_time_stop() #rule evaluation time tracking
                else:
                    offspring.reestablish_rule(rule_summary,heros)
            else:
                heros.timer.rule_eval_time_start() #rule evaluation time tracking
                front_changed = self.evaluate_offspring_rule(offspring,outcome_state,heros,random)
                if front_changed:
                    front_updated = True
                heros.timer.rule_eval_time_stop() #rule evaluation time tracking

            if self.no_identical_rule_exists(offspring,heros,'pop_set'):
                final_offspring_list.append(offspring)

        # Update all rule fitness values if one or both offspring rules updated the pareto front
        heros.timer.rule_eval_time_start() #rule evaluation time tracking
        if heros.fitness_function == 'pareto' and front_updated: #new 3/29/25
            self.global_fitness_update(heros) #Re-evaluates all rule fitness values in rule population
            #In update fitness of two offspring rules that are not yet in the population
            for offspring in final_offspring_list:
                offspring.update_rule_fitness(heros)
        heros.timer.rule_eval_time_stop() #rule evaluation time tracking

        # INSERT RULE(S) IN POPULATON (OPTIONAL GA SUBSUMPTION) ***************************
        return self.process_offspring(parent_list,final_offspring_list,heros)


    def tournament_selection(self,heros,random):
        """ Applies tournament selection to choose and return two parent rules. """
        parent_options = sorted(copy.deepcopy(self.match_set)) #extra code to ensure random seed reproducibility
        parent_list = []
        #print("length of match set: "+str(len(self.match_set)))
        #for rule_ref in parent_options: #debugging
        #    self.pop_set[rule_ref].show_rule()

        if len(parent_options) == 1: #only one rule in {M}
            #parent_list = [self.pop_set[self.match_set[0]]] #only one parent returned
            parent_list = [self.pop_set[parent_options[0]]]
        elif len(parent_options) == 2: #only two rules in {M}
            #parent_list = [self.pop_set[self.match_set[0]],self.pop_set[self.match_set[1]]]
            parent_list = [self.pop_set[parent_options[0]], self.pop_set[parent_options[1]]]
        else:
            while len(parent_list) < 2:
                tournament_size = max(2,int(len(parent_options)*heros.theta_sel))
                tournament_set = random.sample(parent_options,tournament_size)
                #best_fitness = 0
                #best_rule_index = self.match_set[0]
                best_rule_index = tournament_set[0]
                best_fitness = self.pop_set[best_rule_index].fitness

                for i in tournament_set:
                    if self.pop_set[i].fitness > best_fitness or (self.pop_set[i].fitness == best_fitness and self.pop_set[i].ID < self.pop_set[best_rule_index].ID): #extra code to ensure random seed reproducibility
                        best_fitness = self.pop_set[i].fitness
                        best_rule_index = i
                parent_list.append(self.pop_set[best_rule_index])
                parent_options.remove(best_rule_index)
        return parent_list


    def process_offspring(self,parent_list,offspring_list,heros):
        new_rules = []
        """ Activates GA subsumption (if used), and then inserts offpring rules into population as needed. """
        if heros.subsumption == 'ga' or heros.subsumption == 'both': #apply subsumption and insert rule(s) as needed
            heros.timer.subsumption_time_start()
            for offspring in offspring_list:
                new_rules.extend(self.ga_subsumption(offspring,parent_list,heros))
            heros.timer.subsumption_time_stop()
        else: #insert rule(s) as needed following rule equality check
            for offspring in offspring_list:
                self.add_rule_to_pop(offspring,heros)
                print("process")
            new_rules = offspring_list
        return new_rules


    def ga_subsumption(self,offspring,parent_list,heros):
        """ Applies GA subsumption. """
        new_rules = []
        offspring_subsumed = False
        for parent in parent_list:
            if not offspring_subsumed:
                if parent.subsumes(offspring,heros):
                    offspring_subsumed = True
                    self.micro_pop_count += 1
                    parent.update_numerosity(1)
        if not offspring_subsumed:
            self.add_rule_to_pop(offspring,heros)
            new_rules.append(offspring)
        return new_rules


    def evaluate_covered_rule(self,new_rule,outcome_state,heros,random):
        heros.timer.covering_time_stop() #covering time tracking
        heros.timer.rule_eval_time_start() #rule evaluation time tracking
        if heros.outcome_type == 'class':
            front_updated = new_rule.complete_rule_evaluation_class(heros,random,outcome_state) #only called if brand new rule being added to population
        elif heros.outcome_type == 'quant':
            front_updated = new_rule.complete_rule_evaluation_quant(heros) #only called if brand new rule being added to population
        else:
            pass
        if heros.fitness_function == 'pareto' and front_updated: 
            self.global_fitness_update(heros)
        heros.timer.rule_eval_time_stop() #rule evaluation time tracking
        heros.timer.covering_time_start() #covering time tracking


    def evaluate_offspring_rule(self,new_rule,outcome_state,heros,random):
        heros.timer.rule_eval_time_start() #rule evaluation time tracking
        front_updated = False
        if heros.outcome_type == 'class':
            front_updated = new_rule.complete_rule_evaluation_class(heros,random,outcome_state) #only called if brand new rule being added to population
        elif heros.outcome_type == 'quant':
            front_updated = new_rule.complete_rule_evaluation_quant(heros) #only called if brand new rule being added to population
        else:
            print("Error: Outcome type not found.")
        heros.timer.rule_eval_time_stop() #rule evaluation time tracking
        return front_updated

        
    def no_identical_rule_exists(self,new_rule,heros,where):
        identical_rule = None
        heros.timer.rule_equality_time_start() #rule equality time tracking
        if where == 'pop_set':
            identical_rule = self.search_pop_for_identical_rule(new_rule)
        elif where == 'match_set': 
            identical_rule = self.search_match_set_for_identical_rule(new_rule)
        else:
            print('Error: Location for identical rule search not found.')
        heros.timer.rule_equality_time_stop() #rule equality time tracking
        if identical_rule != None: #Identical rule found
            identical_rule.update_numerosity(1) #virtual copy of new rule added
            self.micro_pop_count += 1
            return False
        else:
            return True


    def search_pop_for_identical_rule(self,new_rule):
        """ Identifies if an identical rule already exists in the population. """
        for rule in self.pop_set:
            if new_rule.equals(rule): 
                return rule
        return None


    def search_match_set_for_identical_rule(self,new_rule):
        """ Identifies if an identical rule already exists in the current match set. """
        for each in self.match_set:
            if new_rule.equals(self.pop_set[each]):
                return self.pop_set[each]
        return None


    def add_rule_to_pop(self,new_rule,heros):
        """ Add new and novel rule to population, updating key relevant parameters. """
        new_rule.assign_ID(self.ID_counter)
        self.pop_set.append(new_rule)
        self.ID_counter += 1 #every time a new rule gets added to the pop (that isn't in the current pop) it is assigned a new unique ID
        self.micro_pop_count += 1
        if self.archive_discovered_rules:
            self.add_new_explored_rules(new_rule,heros)


    def make_correct_set(self,outcome_state,heros):
        """ Makes a correct set {C}"""
        for i in range(len(self.match_set)):
            rule_index = self.match_set[i]
            if heros.outcome_type == 'class':
                if self.pop_set[rule_index].action == outcome_state:
                    self.correct_set.append(rule_index)
            elif heros.outcome_type == 'quant':
                if self.pop_set[rule_index].action[0] <= outcome_state <= self.pop_set[rule_index].action[1]:
                    self.correct_set.append(rule_index)
            else:
                pass


    def update_rule_parameters(self,heros):
        """ Updates all relevant rule parameters for rules in the current match set. """
        match_set_numerosity_sum = 0
        for rule_index in self.match_set:
            match_set_numerosity_sum += self.pop_set[rule_index].numerosity
        for rule_index in self.match_set:
            self.pop_set[rule_index].update_ave_match_set_size(match_set_numerosity_sum,heros)


    def deletion(self,heros,random):
        """ Applies probabalistic deletion to the rule population to maintain maximum population size."""
        heros.timer.deletion_time_start()
        while self.micro_pop_count > heros.pop_size:
            self.delete_rule(random,heros)
        heros.timer.deletion_time_stop()
    

    def delete_rule(self,random,heros):
        """ Probabilistically identifies a rule to delete with roulette wheel selection, and deletes it at the micro-rule level."""
        vote_sum = 0.0
        vote_list = []
        for rule in self.pop_set:
            vote = rule.get_deletion_vote(heros)
            vote_sum += vote
            vote_list.append(vote)
        i = 0
        for rule in self.pop_set:
            rule.deletion_prob = vote_list[i] / vote_sum 
            i += 1
        choicePoint = vote_sum  * random.random()  # Determine the choice point
        new_sum = 0.0
        for i in range(len(vote_list)): 
            rule = self.pop_set[i]
            new_sum = new_sum + vote_list[i]
            if new_sum > choicePoint:  # Select classifier for deletion
                # Delete classifier----------------------------------
                self.micro_pop_count -= 1
                if rule.numerosity == 1: # When all micro-classifiers for a given classifier have been depleted.
                    self.remove_macro_rule(i)
                else:
                    rule.update_numerosity(-1)
                return


    def get_pop_fitness_sum(self):
        """ Returns the sum of the fitnesses of all rules in the population. """
        fitness_sum = 0.0
        for rule in self.pop_set:
            fitness_sum += rule.fitness *rule.numerosity
        return fitness_sum 
    

    def remove_macro_rule(self,rule_index):
        """ Removes the given (macro-) rule from the population. """
        self.pop_set.pop(rule_index)
    

    def clear_sets(self):
        """ Clears out references in the match and correct sets for the next learning iteration. """
        self.match_set = []
        self.correct_set = []


    def order_all_rule_conditions(self):
        """ Order the rule conditions by increasing feature index; keeping the ordering consistent between condition_indexes and condition_values."""
        for rule in self.pop_set:
            rule.order_rule_conditions()


    def load_rule_population(self, pop_df, heros, random, np):
        """ Load a HEROS rule population data frame, then instantiates and evaluates all rules.
            Each specified rule must have a condition and action at minimum. """
        self.ID_counter  = pop_df['ID'].astype(int).max() + 1
        if heros.verbose:
            print("Initializing Rule Population via Loaded File!")
            print('Max Rule ID in Loaded Population: '+str(self.ID_counter))
        for index, row in pop_df.iterrows():
            # Initialize the rule
            loaded_rule = RULE(heros)
            # Set the rule condition
            loaded_rule.condition_indexes = ast.literal_eval(row['Condition Indexes'])
            safe_globals = {"__builtins__": {}, "inf": np.inf, "-inf": -np.inf}
            loaded_rule.condition_values = eval(row['Condition Values'], safe_globals)
            # Set the rule action
            if heros.outcome_type =='class':
                loaded_rule.action = int(row['Action'])
            elif heros.outcome_type =='quant':
                loaded_rule.action = ast.literal_eval(row['Action'])
            else:
                pass
            # Set the rule ID
            loaded_rule.ID = int(row['ID'])
            # Set the rule numerosity
            if loaded_rule.numerosity is None:
                loaded_rule.numerosity = 1
            else:
                loaded_rule.numerosity = int(row['Numerosity'])
            # Set the rule average match set size
            if loaded_rule.ave_match_set_size is None:
                loaded_rule.ave_match_set_size = 1
            else:
                loaded_rule.ave_match_set_size = float(row['Average Match Set Size'])
            # Set the rule birth iteration (Currently we simplify by resetting the birth iteration to zero)
            loaded_rule.birth_iteration = 0
            # Evaluate loaded rule
            if heros.outcome_type == 'class':
                front_updated = loaded_rule.complete_rule_evaluation_class(heros,random,None) #only called if brand new rule being added to population
            elif heros.outcome_type == 'quant':
                front_updated = loaded_rule.complete_rule_evaluation_quant(heros) #only called if brand new rule being added to population
            # Add rule to the population
            self.pop_set.append(loaded_rule)
            self.micro_pop_count += loaded_rule.numerosity

        # Update all rule fitness values (if pareto front rule fitness used)
        if heros.fitness_function == 'pareto': #new 3/29/25
            self.global_fitness_update(heros)
        if heros.verbose:
            print('Loading Rule Population Complete: '+str(len(self.pop_set))+' unique rules and '+str(self.micro_pop_count)+' total rules loaded.')


    def tree_init_population(self, X, y, heros, random, np, verbose = False, bstrap = False):
        """ Trains a set of decision trees using random forest classifier and then extracts rules from tree branches, deduplicates the candidate rule population, converts rules to HEROS format and lastly evaluates all unique rules and updates remaining rule parameters. Initial exploration of methodology by Harsh Bandhey, and early contributions to method development by following UPenn students: Akshita Islam, Khoi Dinh, and Gabe Gabe Lipschutz-Villa. """

        # STEP 1: Hard Coded Random Forest Hyperparameters for Tree Training and Rule Extraction --------------------------------
        RF_INIT_SHARED = {
            "n_estimators": 10,
            "bootstrap": bstrap,
            "oob_score": False,
            "n_jobs": -1,
            "random_state": heros.random_state,
            "max_features": "sqrt",
        }

        max_depth_values = [1, 2, 3, 4, 5, 6, 7, None]

        rf_settings = [
            {
                **RF_INIT_SHARED,
                "max_depth": depth
            }
            for depth in max_depth_values
        ]
        # -----------------------------------------------------------------------------------------------------------------
        print('Beginning Decision Tree Rule Inititialization...')
        #--------------------------------------------------------------------------------------------------------------------
        # STEP 2: One Hot Encode the categorical features for decision tree training (since random forest classifier expects quantitative features) 
        #original_X = X.copy() if hasattr(X, 'copy') else X
        onehot_mapping = {}  # Maps one-hot encoded feature index -> (original_feat_idx, categorical_value)
        reverse_onehot_mapping = {}  # Maps (original_feat_idx, categorical_value) -> one-hot encoded feature index
        quant_feat_mapping = {}  # Maps encoded quantitative feature index -> original feature index
        
        if heros.cat_feature_indexes is not None and len(heros.cat_feature_indexes) > 0:
            print(f"\nOne-hot encoding {len(heros.cat_feature_indexes)} categorical features...")
            
            # Convert X to numpy array if it's a DataFrame
            if hasattr(X, 'values'):
                X_array = X.values
                X_is_dataframe = True
                X_columns = list(X.columns)
            else:
                X_array = np.array(X)
                X_is_dataframe = False
                X_columns = None
            
            # Separate categorical and quantitative features
            cat_feat_indexes = sorted(heros.cat_feature_indexes)
            quant_feat_indexes = sorted([i for i in range(X_array.shape[1]) if i not in cat_feat_indexes])

            # Extract categorical and quantitative columns
            cat_data = X_array[:, cat_feat_indexes]
            quant_data = X_array[:, quant_feat_indexes] if quant_feat_indexes else None
    
            # One-hot encode categorical features
            onehot_encoder = OneHotEncoder(sparse_output=False, drop=None, handle_unknown='ignore')
            cat_onehot = onehot_encoder.fit_transform(cat_data)
            
            # Build mapping: one-hot encoded feature index -> (original_feat_idx, categorical_value)
            # Also build mapping for quantitative features: encoded_idx -> original_idx
            if quant_feat_indexes:
                for encoded_idx, orig_idx in enumerate(quant_feat_indexes):
                    quant_feat_mapping[encoded_idx] = orig_idx
            
            # Build one-hot mapping
            #current_onehot_idx = 0
            num_quant = len(quant_feat_indexes) if quant_feat_indexes else 0
            
            for cat_col_idx, orig_cat_idx in enumerate(cat_feat_indexes):
                # Get the categories from the encoder (in order)
                if hasattr(onehot_encoder, 'categories_'):
                    encoder_categories = onehot_encoder.categories_[cat_col_idx]
                else:
                    # Fallback: use unique values from data
                    encoder_categories = np.unique(cat_data[:, cat_col_idx])
                
                # Find the start index for this categorical feature's one-hot columns and count how many one-hot columns come before this feature
                onehot_start_idx = 0
                for prev_cat_idx in cat_feat_indexes:
                    if prev_cat_idx == orig_cat_idx:
                        break
                    prev_cat_col_idx = cat_feat_indexes.index(prev_cat_idx)
                    if hasattr(onehot_encoder, 'categories_'):
                        onehot_start_idx += len(onehot_encoder.categories_[prev_cat_col_idx])
                    else:
                        onehot_start_idx += len(np.unique(cat_data[:, prev_cat_col_idx]))
                
                # Map each one-hot column for this categorical feature
                for cat_val_idx, cat_val in enumerate(encoder_categories):
                    onehot_feat_idx = num_quant + onehot_start_idx + cat_val_idx
                    onehot_mapping[onehot_feat_idx] = (orig_cat_idx, cat_val)
                    reverse_onehot_mapping[(orig_cat_idx, cat_val)] = onehot_feat_idx
            
            # Combine quantitative and one-hot encoded features
            if quant_data is not None:
                X_encoded = np.hstack([quant_data, cat_onehot])
            else:
                X_encoded = cat_onehot
            
            # Convert back to DataFrame if original was DataFrame
            if X_is_dataframe:
                # Create new column names
                new_columns = []
                if quant_feat_indexes:
                    new_columns.extend([X_columns[i] for i in quant_feat_indexes])
                for orig_cat_idx in cat_feat_indexes:
                    cat_values = heros.env.feat_c_values[orig_cat_idx]
                    if hasattr(onehot_encoder, 'categories_'):
                        encoder_categories = onehot_encoder.categories_[cat_feat_indexes.index(orig_cat_idx)]
                    else:
                        encoder_categories = np.unique(cat_data[:, cat_feat_indexes.index(orig_cat_idx)])
                    for cat_val in encoder_categories:
                        new_columns.append(f"{X_columns[orig_cat_idx]}_{cat_val}")
                X = pd.DataFrame(X_encoded, columns=new_columns, index=X.index if hasattr(X, 'index') else None)
            else:
                X = X_encoded
            
            print(f"  Original features: {X_array.shape[1]}, After one-hot encoding: {X.shape[1]}")
            print(f"  One-hot mapping created for {len(onehot_mapping)} encoded features")
            print(f"  Quantitative feature mapping: {len(quant_feat_mapping)} features")
        else:
            print("\nNo categorical features to encode.")

        #--------------------------------------------------------------------------------------------------------------------
        # STEP 3: Train multiple random forest classifiers with varying hyperparameters to create a diverse set of decision trees for rule extraction 
        rf_models = []
        tree_depths_by_rf = []
        for idx, params in enumerate(rf_settings):
            rf = RandomForestClassifier(
                **params
            )
            rf.fit(X, y)
            rf_models.append(rf)
            tree_depths = [estimator.tree_.max_depth for estimator in rf.estimators_]
            tree_depths_by_rf.append(tree_depths)

        # Save the first RF for fidelity proof and visualization - FOR DEBUGGING ONLY, CAN BE REMOVED LATER
        self.rf_model = rf_models[0]

        print("Random Seed Check After RF: "+ str(random.random()))

        def print_rf_training_summary(rf_models):
            print("\nSummary: Trained {} random forests with varying hyperparameters.".format(len(rf_models)))
            for i, rf in enumerate(rf_models):
                depths = [estimator.tree_.max_depth for estimator in rf.estimators_]
                print(f"  RF {i+1}: n_estimators={len(rf.estimators_)}, tree depths={depths}")
        
        if verbose: 
            print_rf_training_summary(rf_models)

        # STEP 4: Extract rules from all trees in all forests
        print("\nExtracting rules from all decision tree branches in all forests...")
        all_rules = []
        branch_paths = []

        def recurse_tree(tree, node_id, path, rules, branch_paths=None, onehot_mapping=None): #consider updating to only store condition (no action) 
            if tree.children_left[node_id] == _tree.TREE_LEAF:
                condition_indexes = []
                condition_values = []
                for feat_idx, threshold, direction in path:
                    condition_indexes.append(feat_idx)
                    condition_values.append((direction, threshold))
                values = tree.value[node_id][0]
                action = np.argmax(values)
                rules.append([condition_indexes, condition_values, action])
                if branch_paths is not None:
                    branch_paths.append((list(condition_indexes), list(condition_values), action, list(path)))
                return
            left_id = tree.children_left[node_id]
            feat_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            recurse_tree(tree, left_id, path + [(feat_idx, threshold, 'leq')], rules, branch_paths, onehot_mapping)
            right_id = tree.children_right[node_id]
            recurse_tree(tree, right_id, path + [(feat_idx, threshold, 'gt')], rules, branch_paths, onehot_mapping)

        for rf in rf_models:
            for estimator in rf.estimators_:
                recurse_tree(estimator.tree_, 0, [], all_rules, branch_paths, onehot_mapping)

        if verbose: 
            # Visual: Tree Depth vs. Number of Rules Produced
            rf_depths = [max([est.tree_.max_depth for est in rf.estimators_]) for rf in rf_models]
            rules_per_rf = []
            for rf in rf_models:
                rf_rules = []
                for estimator in rf.estimators_:
                    local_rules = []
                    recurse_tree(estimator.tree_, 0, [], local_rules)
                    rf_rules.extend(local_rules)
                rules_per_rf.append(len(rf_rules))
            plt.figure(figsize=(7, 4))
            plt.scatter(rf_depths, rules_per_rf, c='#81c784', s=80)
            plt.xlabel("Max Tree Depth in RF")
            plt.ylabel("Number of Rules Extracted")
            plt.title("Tree Depth vs. Number of Rules Extracted")
            plt.grid(True)
            plt.savefig("output/tree_depth_vs_num_rules.png", bbox_inches="tight")
            plt.show()
        print("Random Seed Check After Rule Extract: "+ str(random.random()))

        # STEP 5: Deduplicate rules (based on rule's condition and action)
        print("Deduplicating rules...")
        #rule_tuples = [tuple((tuple(r[0]), tuple(r[1]), r[2])) for r in all_rules]
        #unique_rules_tuples = list(set(rule_tuples))
        #unique_rules = [[list(r[0]), list(r[1]), r[2]] for r in unique_rules_tuples]
        
        # 1. Convert to tuples so they are hashable for the set
        rule_tuples = [tuple((tuple(r[0]), tuple(r[1]), r[2])) for r in all_rules]

        # 2. Use set to get unique items, but IMMEDIATELY sort the resulting list
        # Sorting ensures that the order is identical across every run
        unique_rules_tuples = sorted(list(set(rule_tuples)))

        # 3. Convert back to the original list-of-lists format
        unique_rules = [[list(r[0]), list(r[1]), r[2]] for r in unique_rules_tuples]

        if verbose: 
            print("\nSummary: Extracted {} branch-rules from all trees.".format(len(all_rules)))
            print("After deduplication, {} unique rules remain.".format(len(unique_rules)))

        print("Random Seed Check After Deduplication: "+ str(random.random()))
        # STEP 5: Convert rules to HEROS format, check for redundancy, and add to population
        print("\nConverting extracted rules to HEROS format and checking for redundancy...")

        # --------------------------------------------------------------------------------------------------------
        def convert_path_to_minmax(condition_indexes, condition_values, onehot_mapping, quant_feat_mapping):
            """Convert a list of (direction, threshold) for each feature into HEROS format.
            Handles both quantitative features (min/max ranges) and categorical features (equality checks).
            Maps one-hot encoded features back to original categorical features."""
            minmax_dict = {}  # For quantitative features: {orig_feat_idx: [min, max]}
            categorical_dict = {}  # For categorical features: {orig_feat_idx: set of values}
            
            for idx, (direction, threshold) in zip(condition_indexes, condition_values):
                # Check if this is a one-hot encoded feature
                if idx in onehot_mapping:
                    # This is a one-hot encoded categorical feature
                    orig_feat_idx, cat_value = onehot_mapping[idx]
                    
                    # For one-hot encoding: features are binary (0 or 1) - Threshold is typically 0.5
                    # If direction is 'gt' and threshold <= 0.5, it means the one-hot feature is 1 (category IS present)
                    # If direction is 'leq' and threshold < 0.5, it means the one-hot feature is 0 (category NOT present)
                    if direction == 'gt' and threshold <= 0.5:
                        # This branch means the one-hot feature is 1, so the category IS present
                        if orig_feat_idx not in categorical_dict:
                            categorical_dict[orig_feat_idx] = set()
                        categorical_dict[orig_feat_idx].add(cat_value)
                    # If direction is 'leq' and threshold < 0.5, the category is NOT present (we ignore it)
                    # Note: We only add categories that are explicitly present (value = 1)
                else:
                    # This is a quantitative feature
                    # Map encoded index back to original index
                    if quant_feat_mapping and idx in quant_feat_mapping:
                        orig_idx = quant_feat_mapping[idx]
                    else:
                        # No one-hot encoding was done, so index is already original
                        orig_idx = idx
                    
                    if orig_idx not in minmax_dict:
                        minmax_dict[orig_idx] = [float('-inf'), float('inf')]
                    if direction == 'leq':
                        minmax_dict[orig_idx][1] = min(minmax_dict[orig_idx][1], threshold)
                    elif direction == 'gt':
                        minmax_dict[orig_idx][0] = max(minmax_dict[orig_idx][0], np.nextafter(threshold, threshold+1))
            
            # Build final condition lists
            clean_indexes = []
            clean_values = []

            # Add quantitative features
            for idx in sorted(minmax_dict.keys()):
                min_val, max_val = minmax_dict[idx]
                if min_val <= max_val:
                    clean_indexes.append(idx)
                    clean_values.append([min_val, max_val])
            
            # Add categorical features - we need to check if all one-hot conditions for a feature point to the same value
            for orig_feat_idx in sorted(categorical_dict.keys()):
                cat_values = categorical_dict[orig_feat_idx]
                # If only one value is in the set, that's the categorical condition
                if len(cat_values) == 1:
                    clean_indexes.append(orig_feat_idx)
                    clean_values.append(list(cat_values)[0])  # Single categorical value, not a range
                # If multiple values, we might need to handle this differently - For now, we'll take the first one (though this might not be correct)
                elif len(cat_values) > 1:
                    # Multiple categories for same feature - this shouldn't happen in a valid tree path - But if it does, we'll use the first one
                    clean_indexes.append(orig_feat_idx)
                    clean_values.append(list(cat_values)[0])
            return clean_indexes, clean_values
        # --------------------------------------------------------------------------------------------------------

        for rule_data in unique_rules:
            raw_condition_indexes, raw_condition_values, action = rule_data
            condition_indexes, condition_values = convert_path_to_minmax(raw_condition_indexes, raw_condition_values, onehot_mapping, quant_feat_mapping)
            if len(condition_indexes) == 0:
                continue

            # Create a new RULE object
            rule_obj = RULE(heros)
            #rule_obj.condition_indexes = list(condition_indexes)
            rule_obj.condition_indexes = [x.item() if hasattr(x, 'item') else x for x in condition_indexes]
            #print(type(rule_obj.condition_indexes[0]))
            #rule_obj.condition_values = list(condition_values)
            rule_obj.condition_values = [x.item() if hasattr(x, 'item') else x for x in condition_values]  #CHECK THIS STILL WORKS FOR Quantitative features
            #print(type(rule_obj.condition_values[0]))
            rule_obj.action = None
            rule_obj.numerosity = 1
            rule_obj.birth_iteration = 0
            try: #FUTURE EXPANSION TO QUANTITATIVE OUTCOMES NEEDED
                if hasattr(rule_obj, 'complete_rule_evaluation_class'): #Evaluates rules and assignes best outcome.
                    front_updated = rule_obj.complete_rule_evaluation_class(heros, random, None)
                else:
                    continue
                if not hasattr(rule_obj, 'match_cover') or rule_obj.match_cover == 0:
                    continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                print(f"Type of unexpected exception: {type(e)}")
                continue
            if heros.fitness_function == 'pareto':  #Needs expansion for non-pareto option
                rule_obj.update_rule_fitness(heros)
            identical_rule = self.search_pop_for_identical_rule(rule_obj)
            if identical_rule is not None:
                pass # NEW 4/18/26 only a single copy of each rule is used for initialization (i.e. numerosity = 1)
                #identical_rule.update_numerosity(1)
                #self.micro_pop_count += 1
            else:
                rule_obj.assign_ID(self.ID_counter)
                self.pop_set.append(rule_obj)
                self.ID_counter += 1
                self.micro_pop_count += 1

        #Global Fitness update
        if heros.fitness_function == 'pareto':  #Needs expansion for non-pareto option
            self.global_fitness_update(heros)
        print("Random Seed Check After Convert to HEROS rules: "+ str(random.random()))

        def print_rule_conversion_summary(pop_set, micro_pop_count):
            print("\nSummary: Converted rules to HEROS format and added to population.")
            print(f"Total Population Numerosity: {micro_pop_count}")
            print(f"Unique HEROS Rules: {len(pop_set)}")

        print_rule_conversion_summary(self.pop_set, self.micro_pop_count)

        self.order_all_rule_conditions() #New potential random seed reproducibitliy fix

        """
        if verbose: 


            # STEP 8: Visualize the first decision tree in the random forest
            print("\nVisualizing the first decision tree in the random forest...")
            try:
                estimator = self.rf_model.estimators_[0]
                plt.figure(figsize=(20, 10))
                sktree.plot_tree(
                    estimator,
                    feature_names=list(X.columns) if hasattr(X, 'columns') else None,
                    class_names=[str(c) for c in np.unique(y)],
                    filled=True, rounded=True
                )
                plt.title("First Decision Tree in Random Forest")
                plt.savefig("output/decision_tree_visualization_matplotlib.png", bbox_inches="tight")
                plt.show()
                print("Decision tree visualization saved as 'decision_tree_visualization_matplotlib.png'")
            except Exception as e:
                print(f"Could not visualize decision tree: {e}")

            def print_tree_visualization_summary():
                print("\nSummary: Plotted and saved the first decision tree from the random forest.")
            print_tree_visualization_summary()

            # STEP 9: Visualize a single branch and its conversion to a HEROS rule
            print("\nVisualizing a single branch and its corresponding rule...")
            try:
                branch = None
                for rf in rf_models:
                    if len(rf.estimators_) > 0:
                        tree = rf.estimators_[0].tree_
                        branch_paths_local = []
                        recurse_tree(tree, 0, [], [], branch_paths_local)
                        if branch_paths_local:
                            branch = branch_paths_local[0]
                            break
                if branch is not None:
                    cond_indexes, cond_values, action, path = branch
                    print("Example branch path:")
                    for step in path:
                        feat_idx, threshold, direction = step
                        feat_name = X.columns[feat_idx] if hasattr(X, 'columns') else f"f{feat_idx}"
                        print(f"  If {feat_name} {'<=' if direction == 'leq' else '>'} {threshold:.4f}")
                    print(f"  --> Predict class: {action}")
                    heros_indexes, heros_values = convert_path_to_minmax(cond_indexes, cond_values)
                    print("Converted to HEROS rule format:")
                    for idx, (minv, maxv) in zip(heros_indexes, heros_values):
                        feat_name = X.columns[idx] if hasattr(X, 'columns') else f"f{idx}"
                        print(f"  {feat_name}: [{minv:.4f}, {maxv:.4f}]")
                    print(f"  Action: {action}")
                    # Visualize the branch as a path in the tree
                    print("Visualizing the branch as a path in the tree...")
                    try:
                        estimator = self.rf_model.estimators_[0]
                        plt.figure(figsize=(20, 10))
                        sktree.plot_tree(
                            estimator,
                            feature_names=list(X.columns) if hasattr(X, 'columns') else None,
                            class_names=[str(c) for c in np.unique(y)],
                            filled=True, rounded=True,
                            impurity=False,
                            proportion=False,
                            precision=2
                        )
                        # Highlight the branch path (not trivial in matplotlib, so just print info)
                        plt.title("First Decision Tree with Example Branch (see printed path)")
                        plt.savefig("output/decision_tree_with_branch.png", bbox_inches="tight")
                        plt.show()
                        print("Decision tree with branch visualization saved as 'decision_tree_with_branch.png'")
                    except Exception as e:
                        print(f"Could not visualize branch in tree: {e}")
                else:
                    print("No branch found for visualization.")
            except Exception as e:
                print(f"Could not visualize branch-to-rule conversion: {e}")

            def print_branch_visualization_summary():
                print("\nSummary: Printed a single branch from a tree and its conversion to a HEROS rule.")
            print_branch_visualization_summary()
        """



    def export_rule_population(self,rsl='Unspecified'):
        """ Prepares and exports a dataframe capturing the rule population."""
        pop_list = []
        column_names = ['ID',
                        'Condition Indexes',
                        'Condition Values',
                        'Action',
                        'Numerosity',
                        'Fitness',
                        'Useful Accuracy',
                        'Useful Coverage',
                        'Accuracy',
                        'Match Cover',
                        'Correct Cover',
                        'Mean Absolute Error',
                        'Prediction',
                        'Outcome Range Probability',
                        'Birth Iteration',
                        'Specified Count (RSL='+str(rsl)+')',
                        'Average Match Set Size',
                        'Deletion Probabiilty']
        for rule in self.pop_set:
            rule_list = [rule.ID,
                         rule.condition_indexes,
                         rule.condition_values,
                         rule.action,
                         rule.numerosity,
                         rule.fitness,
                         rule.useful_accuracy,
                         rule.useful_coverage,
                         rule.accuracy,
                         rule.match_cover,
                         rule.correct_cover,
                         rule.mean_absolute_error,
                         rule.prediction,
                         rule.outcome_range_prob,
                         rule.birth_iteration,
                         len(rule.condition_indexes),
                         rule.ave_match_set_size,
                         rule.deletion_prob]
            pop_list.append(rule_list)
        pop_df = pd.DataFrame(pop_list,columns=column_names)
        return pop_df


    def show_rules(self,rule_list,name):
        """ Print condition of rules for debugging."""
        for target_rule in rule_list:
            target_rule.show_rule_short(name)


    def multiplex6_delete_test(self):
        temp_pop = []
        #delete all rules 
        for rule in self.pop_set:
            if rule.useful_accuracy == 1.0:
                if len(rule.condition_indexes) == 3:
                    if 0 in rule.condition_indexes and 1 in rule.condition_indexes:
                        temp_pop.append(rule)
        self.pop_set = temp_pop


    def plot_rule_pop_heatmap(self, feature_names, heros, weighting='useful_accuracy', specified_filter=None, display_micro=False, show=True, save=False, output_path=None):
        """ Plots a clustered heatmap of the rule population based on what features are specified vs. generalized in each rule.
            Hierarchical clustering is applied to rows (i.e. across rules), with feature order preserved. 

            Parameters:
            :param feature_names: a list of feature names for the entire training dataset (given in original dataset order)
            :param weighting: indicates what (if any) weighting is applied to individual rules for the plot ('useful_accuracy', 'fitness', None)
            :param specified_filter: the number of times a given feature must be specified in rules of the population to be included in the plot (must be a positive integer or None)
            :param display_micro: controls whether or not additional copies of rules (based on rule numerosity) should be included in the heatmap (True or False) 
            :param show: indicates whether or not to show the plot (True or False)
            :param save: indicates whether or not to save the plot to a specified path/filename (True or False)
            :param output_path: a valid folder path within which to save the plot (str of folder path)
            :param data_name: a unique name precursor to give to the plot (str)
        """
        if display_micro:
            rule_spec_df = pd.DataFrame([[0.0] * heros.env.num_feat for _ in range(self.micro_pop_count)])
            rule_weight_df = pd.DataFrame([[0.0] * heros.env.num_feat for _ in range(self.micro_pop_count)])
        else:
            rule_spec_df = pd.DataFrame([[0.0] * heros.env.num_feat for _ in range(len(self.pop_set))])
            rule_weight_df = pd.DataFrame([[0.0] * heros.env.num_feat for _ in range(len(self.pop_set))])
        # Add feature names as the dataframe columns
        rule_weight_df.columns = feature_names
        rule_spec_df.columns = feature_names
        # Add feature specificities (and weights if selected) to this dataframe
        row = 0
        for rule in self.pop_set:
            if display_micro: #include copies of rules based on rule numerosity
                for copy in range(rule.numerosity):
                    feat_index = 0 #feature index
                    for feat in feature_names:
                        if feat_index in rule.condition_indexes: #feature is specified in given rule
                            rule_spec_df.at[row,feat] = 1.0
                            if weighting is None or weighting == 'None':
                                rule_weight_df.at[row,feat] = 1.0
                            elif weighting == 'useful_accuracy':
                                rule_weight_df.at[row,feat] = float(rule.useful_accuracy)
                            elif weighting == 'fitness':
                                rule_weight_df.at[row,feat] = float(rule.fitness)
                            else:
                                print("Warning: Rule pop heatmap weighting must be 'useful_accuracy', 'fitness' or None. " )
                        feat_index += 1
                    row += 1
            else: #include each rule only once (i.e. ignore rule numerosity)
                feat_index = 0 #feature index
                for feat in feature_names:
                    if feat_index in rule.condition_indexes: #feature is specified in given rule
                        rule_spec_df.at[row,feat] = 1.0
                        if weighting is None or weighting == 'None':
                            rule_weight_df.at[row,feat] = 1.0
                        elif weighting == 'useful_accuracy':
                            rule_weight_df.at[row,feat] = float(rule.useful_accuracy)
                        elif weighting == 'fitness':
                            rule_weight_df.at[row,feat] = float(rule.fitness)
                        else:
                            print("Warning: Rule pop heatmap weighting must be 'useful_accuracy', 'fitness' or None. " )
                    feat_index += 1
                row += 1      
        # Apply optional filtering to the dataframe to remove features that are specified with a lower frequency
        if specified_filter != None and specified_filter != 'None':
            cols_to_keep = (rule_spec_df != 0.0).sum(axis=0) >= specified_filter
            rule_weight_df = rule_weight_df.loc[:, cols_to_keep]
            rule_spec_df = rule_spec_df.loc[:, cols_to_keep]
        # Perform hierarchical clustering on columns
        col_linkage = linkage(rule_spec_df.T, method='average', metric='euclidean', optimal_ordering=False)
        # Perform hierarchical clustering on rows
        row_linkage = linkage(rule_spec_df.values, method='average', metric='euclidean', optimal_ordering=True)
        # Create a seaborn clustermap
        #clustermap = sns.clustermap(rule_weight_df, row_linkage=row_linkage, col_cluster=False, cmap='viridis', figsize=(10, 10))
        clustermap = sns.clustermap(rule_weight_df, row_linkage=row_linkage, col_linkage=col_linkage, cmap='viridis', figsize=(10, 10))
        clustermap.ax_heatmap.set_xlabel('Features', fontsize=12)
        clustermap.ax_heatmap.set_ylabel('Rules', fontsize=12)
        clustermap.ax_heatmap.set_yticks([])
        # Dynamicaly update x-tick label text size based on number of features in the dataset (up to a minimum )
        num_features = rule_weight_df.shape[1]
        min_text_size = 4
        max_text_size = 12
        font_size = max(min_text_size, max_text_size - num_features // min_text_size)  # Adjust font size based on the number of features
        clustermap.ax_heatmap.set_xticklabels(clustermap.ax_heatmap.get_xticklabels(), rotation=90, fontsize=font_size)
        if save:
            plt.savefig(output_path+'/clustered_rule_pop_heatmap.png', bbox_inches="tight")
        if show:
            plt.show()


    def plot_rule_pop_network(self, feature_names, weighting='useful_accuracy', display_micro=False, node_size=1000, edge_size=10, show=True, save=False, output_path=None):
        """ Plots a network visualization of the rule population with feature specificity across rules as node size and feature co-specificity 
            across rules in the population as edge size.
        """
        # Initialize dictionaries to count the number of times each feature is specified in rules of the population and how often feature combinations are cospecified
        feat_spec_count = defaultdict(int)
        feat_cooccurrence_count = defaultdict(int)
        #Create dictionaries of specificity counts
        for rule in self.pop_set:
            # Count appearances of each integer
            base_score = 1.0
            if display_micro:
                base_score = base_score * rule.numerosity
            for feature_index in rule.condition_indexes:
                if weighting is None or weighting == 'None':
                    feat_spec_count[feature_index] += base_score
                elif weighting == 'useful_accuracy':
                    feat_spec_count[feature_index] += base_score * rule.useful_accuracy
                elif weighting == 'fitness':
                    feat_spec_count[feature_index] += base_score * rule.fitness
                else:
                    print("Warning: Rule pop network weighting must be 'useful_accuracy', 'fitness' or None. " )
            # Count appearances of each unique pair
            for pair in combinations(rule.condition_indexes, 2):
                # Ensure pairs are in sorted order to avoid duplicate pairs (e.g., (1, 2) and (2, 1))
                pair = tuple(sorted(pair))
                if weighting is None or weighting == 'None':
                    feat_cooccurrence_count[pair] += base_score
                elif weighting == 'useful_accuracy':
                    feat_cooccurrence_count[pair] += base_score * rule.useful_accuracy
                elif weighting == 'fitness':
                    feat_cooccurrence_count[pair] += base_score * rule.fitness
                else:
                    print("Warning: Rule pop network weighting must be 'useful_accuracy', 'fitness' or None. " )
        # Convert defaultdicts to regular dictionaries
        feat_spec_count = dict(feat_spec_count)
        feat_cooccurrence_count = dict(feat_cooccurrence_count)
        # Scale all node weights to a max of 1
        max_value = max(feat_spec_count.values())
        feat_spec_count = {key: value / max_value for key, value in feat_spec_count.items()}
        # Scale all edge weights to a max of 1
        max_value = max(feat_cooccurrence_count.values())
        feat_cooccurrence_count = {key: value / max_value for key, value in feat_cooccurrence_count.items()}
        # Create a graph
        G = nx.Graph()
        # Add nodes with their weights
        for feature, weight in feat_spec_count.items():
            G.add_node(feature_names[feature], size=weight)
        # Add edges with their weights
        for (feature1, feature2), weight in feat_cooccurrence_count.items():
            G.add_edge(feature_names[feature1], feature_names[feature2], weight=weight)
        # Get positions for the nodes
        pos = nx.circular_layout(G)
        # Draw nodes with sizes proportional to their weights
        node_sizes = [G.nodes[node]['size'] * node_size for node in G.nodes]  # Scale factor for visibility
        # Set node colors proportional to normalized weights
        node_colors = [G.nodes[node]['size'] for node in G.nodes] 
        #nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.9)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap='viridis', alpha=0.9)
        # Draw edges with widths proportional to their weights
        edge_widths = [G.edges[edge]['weight'] * edge_size for edge in G.edges]
        edge_colors = [G.edges[edge]['weight'] for edge in G.edges]
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors)
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_color='orange')
        # Show the plot
        plt.axis('off')
        if save:
            plt.savefig(output_path+'/rule_pop_network.png', bbox_inches="tight")
        if show:
            plt.show()

    """def decode_rule_binary(self, binary_str, num_features):
        
        Decodes a binary encoded rule string (compact version with only 2-bit int values, no float ranges).
        
        Returns:
            [
                condition_indexes: list[int],
                condition_values: list[int],
                action: int,
                instance_outcome_count: dict[int, int]
            ]
        
        instance_outcome_count = self.explored_rules[binary_str]
        ptr = 0

        # --- Step 1: Decode bitmask (num_features bits) ---
        bitmask_str = binary_str[ptr:ptr + num_features]
        condition_indexes = [i for i, b in enumerate(bitmask_str) if b == '1']
        ptr += num_features

        # --- Step 2: Decode type indicators (num_features bits) ---
        type_indicators = binary_str[ptr:ptr + num_features]
        ptr += num_features

        # --- Step 3: Decode all condition values (2 bits each) ---
        condition_values = []
        for i in range(num_features):
            val_binary = binary_str[ptr:ptr + 2]
            val = int(val_binary, 2)
            ptr += 2
            condition_values.append(val)

        # Filter only used condition values (based on bitmask)
        used_condition_values = [condition_values[i] for i in condition_indexes]

        # --- Step 4: Decode action (2 bits) ---
        action_binary = binary_str[ptr:ptr + 2]
        action = int(action_binary, 2)
        ptr += 2

        return [condition_indexes, used_condition_values, action, instance_outcome_count]"""
    
    def decode_rule_binary(self, binary_str, num_features):
        """"
        Decodes a binary encoded rule string (without outcome count).
        Returns:
            [
                condition_indexes: list[int],
                condition_values: list[int or tuple(float, float)],
                action: int
            ]
        """
        instance_outcome_count = self.explored_rules[binary_str]
        ptr = 0  # bit pointer
        # --- Step 1: Decode bitmask ---
        bitmask_str = binary_str[ptr:ptr + num_features]
        condition_indexes = [i for i, b in enumerate(bitmask_str) if b == '1']
        ptr += num_features
        # --- Step 2: Decode type indicators ---
        type_indicators = binary_str[ptr:ptr + len(condition_indexes)]
        ptr += len(condition_indexes)
        # --- Step 3: Decode condition values ---
        condition_values = []
        for indicator in type_indicators:
            if indicator == '0':
                val_binary = binary_str[ptr:ptr + 32]
                condition_values.append(int(val_binary, 2))
                ptr += 32
            elif indicator == '1':
                min_binary = binary_str[ptr:ptr + 32]
                max_binary = binary_str[ptr + 32:ptr + 64]
                min_val = struct.unpack('>f', int(min_binary, 2).to_bytes(4, 'big'))[0]
                max_val = struct.unpack('>f', int(max_binary, 2).to_bytes(4, 'big'))[0]
                condition_values.append((min_val, max_val))
                ptr += 64
            #else:
                #raise ValueError(f”Invalid type indicator: {indicator}“)
        # --- Step 4: Decode action ---
        action_binary = binary_str[ptr:ptr + 32]
        action = int(action_binary, 2)
        ptr += 32
        return [condition_indexes, condition_values, action, instance_outcome_count]