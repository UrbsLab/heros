import copy
import pandas as pd
import ast
from .rule import RULE
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage#, dendrogram, leaves_list
import networkx as nx
from collections import defaultdict
from itertools import combinations

class RULE_POP:
    def __init__(self):
        """ Initializes rule population objects. """
        self.pop_set = []  # List rule objects making up the rule population
        self.match_set = []  # List of references to rules in population that make up a temporary match set (i.e. rules with 'IF' conditions matching current instance state)
        self.correct_set = []  # List of references to rules in population that make up correct set (i.e. rules with 'THEN' action matching current instance outcome)
        self.incorrect_set = [] # List of references to rules in population that make up incorrect set (i.e. rules with 'THEN' action not matching current instance outcome)
        self.micro_pop_count = 0 # Number of rules in the population defined by the sum of individual rule numerosities (aka 'micro' population count)
        self.ID_counter = 0

    def make_match_set(self, instance,heros,random):
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
        heros.timer.covering_time_start() #covering time tracking
        if do_covering:
            new_rule = RULE()
            new_rule.initialize_by_covering(set_numerosity_sum+1,instance_state,outcome_state,heros,random)
            if len(new_rule.condition_indexes) > 0: #prevents completely general rules from being added to the population
                self.add_covered_rule_to_pop(new_rule,heros)
                self.match_set.append(len(self.pop_set)-1)
        heros.timer.covering_time_stop() #covering time tracking

    def make_eval_match_set(self,instance_state,heros):
        """ Makes a match set {M} given an instance state. Used by predict function."""
        for i in range(len(self.pop_set)):
            rule = self.pop_set[i]
            if rule.match(instance_state,heros):
                self.match_set.append(i)

    def add_covered_rule_to_pop(self,new_rule,heros):
        """ Adds a new rule to the population via covering: either as a new rule entry in the population or increasing the numerosity of a rule that already exists. """
        heros.timer.covering_time_stop() #covering time tracking
        heros.timer.rule_eval_time_start() #rule evaluation time tracking
        if heros.outcome_type == 'class':
            front_updated = new_rule.complete_rule_evaluation_class(heros) #only called if brand new rule being added to population
        elif heros.outcome_type == 'quant':
            front_updated = new_rule.complete_rule_evaluation_quant(heros) #only called if brand new rule being added to population
        else:
            pass
        if heros.fitness_function == 'pareto' and front_updated:
            self.global_fitness_update(heros)
        heros.timer.rule_eval_time_stop() #rule evaluation time tracking
        heros.timer.covering_time_start() #covering time tracking
        #print("Add Covered Rule")
        #new_rule.show_rule() #Debug
        #Check if rule already in population and add to population
        self.add_offspring_rule_to_pop(new_rule,heros)
        #self.pop_set.append(new_rule)
        #self.micro_pop_count += 1

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
            if candidate_subsumer == None:
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
                    self.remove_from_incorrect_set(rule_index)
                    i -= 1
                i += 1

    def original_correct_set_subsumption(self,heros):
        """ NOT USED: Applies original correct set subsumption. The most general and accurate rule in the correct set is given the opportunity to subsume the others."""
        # Identify the most general and accurate rule in the correct set
        candidate_subsumer = None
        for rule_index in self.correct_set:
            rule = self.pop_set[rule_index]
            if candidate_subsumer == None or (rule.is_more_general(candidate_subsumer,heros) and rule.accuracy >= candidate_subsumer.accuracy):
                candidate_subsumer = rule
        # Check if the target 'subsumer' subsumes any other 
        if candidate_subsumer != None:
            i = 0
            while i < len(self.correct_set):
                rule_index = self.correct_set[i]
                if candidate_subsumer.is_more_general(self.pop_set[rule_index],heros):
                    candidate_subsumer.update_numerosity(self.pop_set[rule_index].numerosity)
                    self.remove_macro_rule(rule_index)
                    self.remove_from_match_set(rule_index)
                    self.remove_from_correct_set(rule_index)
                    self.remove_from_incorrect_set(rule_index)
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

    def remove_from_incorrect_set(self,rule_index):
        """ Delete reference to rule in population, contained in self.incorrect_set."""
        if rule_index in self.incorrect_set:
            self.incorrect_set.remove(rule_index)
        for j in range(len(self.incorrect_set)):
            ref = self.incorrect_set[j]
            if ref > rule_index:
                self.incorrect_set[j] -= 1

    def genetic_algorithm(self, instance, heros, random):
        instance_state = instance[0] #instance feature values
        outcome_state = instance[1] #instance outcome value
        # PARENT SELECTION *****************************************
        heros.timer.selection_time_start() #parent selection time tracking
        parent_list = self.tournament_selection(heros,random)
        heros.timer.selection_time_stop() #parent selection time tracking
        #self.show_rules(parent_list,"Parent") #Debug
        # INITIALIZE OFFSPRING *************************************
        heros.timer.mating_time_start() #mating time tracking
        offspring_list = []
        for parent_rule in parent_list:
            new_rule = RULE()
            new_rule.initialize_by_parent(parent_rule,heros)
            offspring_list.append(new_rule)
        # CROSSOVER OPERATOR **************************************
        if len(offspring_list) > 1: #crossover only applied between two parent rules
            if random.random() < heros.cross_prob:
                offspring_list[0].uniform_crossover(offspring_list[1],heros,random)
        # MUTATION OPERATOR ***************************************
        for offspring in offspring_list:
            offspring.mutation(instance_state,outcome_state,heros,random)
        heros.timer.mating_time_stop() #mating time tracking
        # EVALUATE OFFSPRING RULES ************************************
        front_updated = False
        heros.timer.rule_eval_time_start() #rule evaluation time tracking
        for offspring in offspring_list:
            if heros.outcome_type == 'class':
                front_updated = offspring.complete_rule_evaluation_class(heros) #only called if brand new rule being added to population
            elif heros.outcome_type == 'quant':
                front_updated = offspring.complete_rule_evaluation_quant(heros) #only called if brand new rule being added to population
            else:
                pass
            if heros.fitness_function == 'pareto' and front_updated:
                self.global_fitness_update(heros)
        heros.timer.rule_eval_time_stop() #rule evaluation time tracking
        #self.show_rules(offspring_list,"Offspring") #Debug
        # INSERT RULE(S) IN POPULATON (OPTIONAL GA SUBSUMPTION) ***************************
        self.insert_offspring(parent_list, offspring_list,heros)

    def tournament_selection(self,heros,random):
        """ Applies tournament selection to choose and return two parent rules. """
        parent_options = copy.deepcopy(self.correct_set)
        parent_list = []
        if len(parent_options) == 1: #only one rule in {C}
            parent_list = [self.pop_set[self.correct_set[0]]] #only one parent returned
        elif len(parent_options) == 2: #only two rules in {C}
            parent_list = [self.pop_set[self.correct_set[0]],self.pop_set[self.correct_set[1]]]
        else:
            while len(parent_list) < 2:
                tournament_size = int(len(parent_options)*heros.theta_sel)
                tournament_set = random.sample(parent_options,tournament_size)
                best_fitness = 0
                best_rule_index = self.correct_set[0]
                for i in tournament_set:
                    if self.pop_set[i].fitness >= best_fitness:
                        best_fitness = self.pop_set[i].fitness
                        best_rule_index = i
                parent_list.append(self.pop_set[best_rule_index])
                parent_options.remove(best_rule_index)
        return parent_list
    
    def insert_offspring(self,parent_list,offspring_list,heros):
        """ Activates GA subsumption (if used), and then inserts offpring rules into population as needed. """
        if heros.subsumption == 'ga' or heros.subsumption == 'both': #apply subsumption and insert rule(s) as needed
            heros.timer.subsumption_time_start()
            for offspring in offspring_list:
                self.ga_subsumption(offspring,parent_list,heros)
            heros.timer.subsumption_time_stop()
        else: #insert rule(s) as needed following rule equality check
            for offspring in offspring_list:
                self.add_offspring_rule_to_pop(offspring,heros) 

    def ga_subsumption(self,offspring,parent_list,heros):
        """ Applies GA subsumption. """
        offspring_subsumed = False
        for parent in parent_list:
            if not offspring_subsumed:
                if parent.subsumes(offspring,heros):
                    offspring_subsumed = True
                    self.micro_pop_count += 1
                    parent.update_numerosity(1)
        if not offspring_subsumed:
            self.add_offspring_rule_to_pop(offspring,heros)

    def add_offspring_rule_to_pop(self,offspring,heros):
        """ Adds a new rule to the population either as a new rule entry in the population or increasing the numerosity of a rule that already exists. """
        identical_rule = None
        heros.timer.rule_equality_time_start() #rule equality time tracking
        identical_rule = self.find_identical_rule(offspring)
        heros.timer.rule_equality_time_stop() #rule equality time tracking
        if identical_rule != None: #Identical rule found
            identical_rule.update_numerosity(1) #virtual copy of new rule added
        else: #new/original rule evaluated then added to population
            #print("Add Offspring Rule")
            #offspring.show_rule() #Debug
            offspring.assign_ID(self.ID_counter)
            self.pop_set.append(offspring)
            self.ID_counter += 1 #every time a new rule gets added to the pop (that isn't in the current pop) it is assigned a new unique ID
        self.micro_pop_count += 1

    def find_identical_rule(self,new_rule):
        """ Identifies if an identical rule already exists in the population. """
        for rule in self.pop_set:
            if new_rule.equals(rule): 
                return rule
        return None

    def make_correct_set(self,outcome_state,heros):
        """ Makes a correct set {C}"""
        for i in range(len(self.match_set)):
            rule_index = self.match_set[i]
            if heros.outcome_type == 'class':
                if self.pop_set[rule_index].action == outcome_state:
                    self.correct_set.append(rule_index)
                else:
                    self.incorrect_set.append(rule_index)
            elif heros.outcome_type == 'quant':
                if self.pop_set[rule_index].action[0] <= outcome_state <= self.pop_set[rule_index].action[1]:
                    self.correct_set.append(rule_index)
                else:
                    self.incorrect_set.append(rule_index)
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
            self.delete_rule(random)
        heros.timer.deletion_time_stop()
    
    def delete_rule(self,random):
        """ Probabilistically identifies a rule to delete with roulette wheel selection, and deletes it at the micro-rule level."""
        mean_fitness = self.get_pop_fitness_sum() / float(self.micro_pop_count)
        vote_sum = 0.0
        vote_list = []
        for rule in self.pop_set:
            vote = rule.get_deletion_vote(mean_fitness)
            vote_sum += vote
            vote_list.append(vote)
        i = 0
        for rule in self.pop_set:
            rule.deletion_prob = vote_list[i] / vote_sum 
            i += 1
        choicePoint = vote_sum  * random.random()  # Determine the choice point
        new_sum = 0.0
        for i in range(len(vote_list)): #RYAN- implement more efficiently
            rule = self.pop_set[i]
            new_sum = new_sum + vote_list[i]
            if new_sum > choicePoint:  # Select classifier for deletion
                # Delete classifier----------------------------------
                rule.update_numerosity(-1)
                self.micro_pop_count -= 1
                if rule.numerosity < 1:  # When all micro-classifiers for a given classifier have been depleted.
                    self.remove_macro_rule(i)
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
        self.incorrect_set = []

    def order_rule_conditions(self):
        """ Order the rule conditions by increasing feature index; keeping the ordering consistent between condition_indexes and condition_values."""
        for rule in self.pop_set:
            list1 = rule.condition_indexes
            list2 = rule.condition_values
            # Combine the two lists using zip
            combined = list(zip(list1, list2))
            # Sort the combined list based on the first list's values
            sorted_combined = sorted(combined, key=lambda x: x[0])
            # Unzip the sorted list back into two separate lists
            sorted_list1, sorted_list2 = zip(*sorted_combined)
            # Convert the tuples back to lists
            rule.condition_indexes = list(sorted_list1)
            rule.condition_values = list(sorted_list2)

    def load_rule_population(self, pop_df, heros):
        """ Load a HEROS rule population data frame, then instantiates and evaluates all rules.
            Each specified rule must have a condition and action at minimum. """
        rule_count = pop_df.shape[0] #rows in dataframe
        for rule_row in range(rule_count): #for each rule in the dataframe
            # Initialize the rule
            loaded_rule = RULE()
            # Set the rule condition
            loaded_rule.condition_indexes = ast.literal_eval(pop_df.loc[rule_row,'Condition Indexes'])
            loaded_rule.condition_values = ast.literal_eval(pop_df.loc[rule_row,'Condition Values'])
            # Set the rule action
            if heros.outcome_type =='class':
                loaded_rule.action = int(pop_df.loc[rule_row,'Action'])
            elif heros.outcome_type =='quant':
                loaded_rule.action = ast.literal_eval(pop_df.loc[rule_row,'Action'])
            else:
                pass
            # Set the rule numerosity
            if loaded_rule.numerosity == None:
                loaded_rule.numerosity = 1
            else:
                loaded_rule.numerosity = int(pop_df.loc[rule_row,'Numerosity'])
            # Set the rule average match set size
            if loaded_rule.ave_match_set_size == None:
                loaded_rule.ave_match_set_size = 1
            else:
                loaded_rule.ave_match_set_size = float(pop_df.loc[rule_row,'Average Match Set Size'])
            # Set the rule birth iteration
            loaded_rule.birth_iteration = 0
            # Evaluate loaded rule
            if heros.outcome_type == 'class':
                front_updated = loaded_rule.complete_rule_evaluation_class(heros) #only called if brand new rule being added to population
            elif heros.outcome_type == 'quant':
                front_updated = loaded_rule.complete_rule_evaluation_quant(heros) #only called if brand new rule being added to population
        # Update all rule fitness values (if pareto front rule fitness used)
        if heros.fitness_function == 'pareto':
            self.global_fitness_update(heros)
        # Add rule to the population
        self.pop_set.append(loaded_rule)
        self.ID_counter += 1 #every time a new rule gets added to the pop (that isn't in the current pop) it is assigned a new unique ID
        self.micro_pop_count += 1

    def export_rule_population(self):
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
                        'Specified Count',
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

    def plot_rule_pop_heatmap(self, feature_names, heros, weighting='useful_accuracy', specified_filter=None, display_micro=False, show=True, save=False, output_path=None, data_name=None):
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
                            if weighting == None or weighting == 'None':
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
                        if weighting == None or weighting == 'None':
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
            plt.savefig(output_path+'/'+data_name+'_clustered_rule_pop_heatmap.png', bbox_inches="tight")
        if show:
            plt.show()

    def plot_rule_pop_network(self, feature_names, weighting='useful_accuracy', display_micro=False, node_size=1000, edge_size=10, show=True, save=False, output_path=None, data_name=None):
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
                if weighting == None or weighting == 'None':
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
                if weighting == None or weighting == 'None':
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
            plt.savefig(output_path+'/'+data_name+'_rule_pop_network.png', bbox_inches="tight")
        if show:
            plt.show()