import copy
import pandas as pd
import random
import numpy as np
from collections import defaultdict
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage#, dendrogram, leaves_list
import networkx as nx

from collections import Counter #temporary debugging


class MODEL: 
    def __init__(self):
        """ Initializes objects that define an individual model (i.e. rule-set). """
        # REFERENCE OBJECTS ******************************************************
        # FIXED MODEL PARAMETERS ***************************************************
        self.rule_set = [] #list of rules in the set
        self.rule_IDs = [] #indexes of rules in original rule population
         # Other fixed rule parameters *************************************************************
        self.accuracy = None #rule-set accuracy (not to be confused with model accuracy) i.e. of the instances this ruleset matches, the proportion where this ruleset predicts the correct outcome
        self.coverage = None # proportion of instaces in training data covered by model
        self.birth_iteration = None #iteration number when this ruleset was first introduced (or re-introduced) to the population
        self.objectives = None
        # FLEXIBLE MODEL PARAMETERS ***************************************************
        self.fitness = None #model 'goodness' metric that drives many aspects of algorithm learning, discovery, and prediction
        self.deletion_prob = None #probability of model being selected for deletion
        self.model_on_front = None #identifies if the model was on the pareto front at the end of phase 2 training


    def initialize_all_rule_model(self,heros):
        """ Initializes a rule set containing all rules from the population. """
        rule_indexes = list(range(len(heros.rule_population.pop_set)))
        self.rule_set = []
        self.rule_IDs = []
        for i in rule_indexes:
            self.rule_set.append(heros.rule_population.pop_set[i])
            self.rule_IDs.append(heros.rule_population.pop_set[i].ID)
        self.birth_iteration = heros.model_iteration


    def initialize_randomly(self, rules_in_model,heros):
        """ Initializes a rule set by randomly selecting rules from the population. """
        index_list = list(range(len(heros.rule_population.pop_set)))
        rule_indexes = random.sample(index_list,rules_in_model)
        self.rule_set = []
        self.rule_IDs = []
        for i in rule_indexes:
            self.rule_set.append(heros.rule_population.pop_set[i])
            self.rule_IDs.append(heros.rule_population.pop_set[i].ID)
        self.birth_iteration = heros.model_iteration
    

    def select_matching_rules(self, data, rule_pop,heros):
        pool = copy.copy(rule_pop)
        train_data = copy.deepcopy(data)
        selected_rules = []
        for instance in train_data:
            chosen = []
            for rule in pool:
                # Check if the rule matches the instance
                if rule.match(instance, heros):
                    selected_rules.append(rule)
                    chosen.append(rule)
            pool = [rule for rule in pool if rule not in chosen]
        return selected_rules

    
    def initialize_target(self, rules_in_model, target, heros):
        """ Initializes rules sets based on a target rule accuracy, so that all rules in each initialized rule set have a similar accuracy value.  The intution behind this strategy is to make the model search flexible in dealing with both clean and noisy problems, exploring the space of rules that would be most suitible to either problem type. """
        pool = copy.copy(heros.rule_population.pop_set)
        self.rule_set = []
        self.rule_IDs = []
        #Begin by identifying any rules in the population with the same accuracy as the target.
        selected_rules = [rule for rule in pool if rule.accuracy == target] 
        if len(selected_rules) > rules_in_model:
            #Select a random number of the rules to equal the chosen rules_in_model
            selected_rules = random.sample(selected_rules,rules_in_model)
        elif len(selected_rules) < rules_in_model:
            #Remove selected rules from rule population pool
            for rule in selected_rules: 
                pool.remove(rule)
            #Probabilistically add rules with an accuracy as close to the target as possible. 
            while len(selected_rules) < rules_in_model:
                weights = []
                for rule in pool:
                    weights.append(float(1/(abs(rule.accuracy - target))))
                if sum(weights) == 0:
                    break
                else: 
                    rule = random.choices(pool, weights = weights)[0]
                    selected_rules.append(rule)
                    pool.remove(rule)
        else:
            pass
        for r in selected_rules:
            self.rule_set.append(r)
            self.rule_IDs.append(r.ID)
        self.birth_iteration = heros.model_iteration

    
    def check_for_duplicates(self,rule_IDs,code_region):
        """ Debugging """
        counts = Counter(rule_IDs)
        # Check if any value appears at least twice
        duplicates = [k for k, v in counts.items() if v >= 2]
        if duplicates:
            print(f"These values ("+str(code_region)+") appear at least twice in the list: {duplicates}")
            print(str(rule_IDs))
            print(len(rule_IDs))
            print(str(counts))
            print(str(5/0))


    def balanced_accuracy(self, y_true, y_pred):
        """
        Calculate balanced accuracy for binary or multiclass classification.
        Parameters:
        y_true (list or array): True class labels.
        y_pred (list or array): Predicted class labels.
        Returns:
        float: Balanced accuracy score.
        Method written by ChatGPT
        """
        # Convert to numpy arrays for easier indexing
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        # Get the unique classes
        classes = np.unique(y_true)
        # Initialize true positive (TP) and false negative (FN) counts for each class
        recall_per_class = defaultdict(float)
        # Calculate recall for each class
        for cls in classes:
            # True positives (TP): correctly predicted as class `cls`
            TP = np.sum((y_true == cls) & (y_pred == cls))
            # False negatives (FN): true class is `cls` but predicted as another class
            FN = np.sum((y_true == cls) & (y_pred != cls))
            # Recall for class `cls`
            recall_per_class[cls] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        # Calculate balanced accuracy as the mean of recall values
        balanced_acc = np.mean(list(recall_per_class.values()))
        return balanced_acc


    def predict(self, instance_state,heros):
        class_counts = {}
        for rule in self.rule_set:
            if rule.match(instance_state,heros):
                if rule.action in class_counts:
                    class_counts[rule.action] += 1 
                else:
                    class_counts[rule.action] = 1 
        if len(class_counts.keys()) > 0: 
            max_count = max(class_counts.values())
            tied_classes = [k for k, v in class_counts.items() if v == max_count]
            if len(tied_classes) > 1:
                pred = None #Will predict a wrong class to discourge ties
            else:
                pred = tied_classes[0]
            return True, pred 
        else: 
            return False, None # instance not covered by model
        

    def evaluate_model_class(self,heros):
        """ Evaluate set performance across the entire training dataset and update set parameters accordingly. """
        train_data = heros.env.train_data
        y_true = [] # true class values
        y_pred = [] # predicted class values
        uncovered_count = 0
        for instance_index in range(heros.env.num_instances):
            instance_state = train_data[0][instance_index]
            y_true.append(train_data[1][instance_index])
            covered, prediction = self.predict(instance_state,heros)
            if covered:
                if prediction != None:
                    y_pred.append(prediction)
                else: # Class tie occured
                    incorrect_classes = [x for x in heros.env.classes if x != train_data[1][instance_index]]
                    y_pred.append(random.choice(incorrect_classes)) #Predicts a wrong class if instance was not covered by model
            else:
                incorrect_classes = [x for x in heros.env.classes if x != train_data[1][instance_index]]
                y_pred.append(random.choice(incorrect_classes)) #Predicts a wrong class if instance was not covered by model
                uncovered_count += 1
        self.accuracy = self.balanced_accuracy(y_true, y_pred)
        self.coverage = (heros.env.num_instances - uncovered_count) / float(heros.env.num_instances)
        #self.objectives = (self.useful_accuracy,len(self.rule_set))
        self.objectives = (self.accuracy,len(self.rule_set))


    def copy_parent(self,parent,iteration):
        #Attributes cloned from parent
        for rule in parent.rule_set:
            self.rule_set.append(rule)
        for ruleID in parent.rule_IDs:
            self.rule_IDs.append(ruleID)
        self.birth_iteration = iteration


    def merge(self,other_parent):
        """ """
        set1 = set(self.rule_IDs)
        set2 = set(other_parent.rule_IDs)
        unique_to_list2 = set2 - set1
        temp_rule_IDs = self.rule_IDs + list(unique_to_list2) 
        temp_rule_set = [] 
        for ruleID in temp_rule_IDs:
            if ruleID in self.rule_IDs:
                index = self.rule_IDs.index(ruleID)
                temp_rule_set.append(self.rule_set[index])
            else:
                index = other_parent.rule_IDs.index(ruleID)
                temp_rule_set.append(other_parent.rule_set[index])  
        self.rule_IDs = temp_rule_IDs
        self.rule_set = temp_rule_set


    def uniform_crossover(self,other_offspring,random):
        """ Applies uniform crossover between this and another model, exchanging rules """           
        set1 = set(self.rule_IDs)
        set2 = set(other_offspring.rule_IDs)
        unique_to_list1 = set1 - set2
        unique_to_list2 = set2 - set1
        unique_rule_IDs = list(sorted(unique_to_list1.union(unique_to_list2)))
        swap_probability = 0.5
        for ruleID in unique_rule_IDs:
            if random.random() < swap_probability:
                if ruleID in self.rule_IDs:
                    index = self.rule_IDs.index(ruleID)
                    #Remove rule from self
                    self.rule_IDs.remove(ruleID)
                    rule_set = self.rule_set.pop(index)
                    #Add rule to other_offspring
                    other_offspring.rule_IDs.append(ruleID)
                    other_offspring.rule_set.append(rule_set)
                else: #ruleID is in other_offspring
                    index = other_offspring.rule_IDs.index(ruleID)
                    #Remove rule from self
                    other_offspring.rule_IDs.remove(ruleID)
                    rule_set = other_offspring.rule_set.pop(index)
                    #Add rule to other_offspring
                    self.rule_IDs.append(ruleID)
                    self.rule_set.append(rule_set)


    def mutation(self,random,heros):
        """ Applies muation to this model """
        if len(self.rule_IDs) == 0: #Initialize new model if empty after crossover
            target_rule_min = 10
            target_rule_max = int(len(heros.rule_population.pop_set)/2)
            if target_rule_max < target_rule_min:
                min_rules = 2
                max_rules = len(heros.rule_population.pop_set)
            else:
                min_rules = target_rule_min
                max_rules = int(len(heros.rule_population.pop_set)/2)
            rules_in_model = random.randint(min_rules,max_rules)
            index_list = list(range(len(heros.rule_population.pop_set)))
            rule_indexes = random.sample(index_list,rules_in_model)
            for i in rule_indexes:
                self.rule_set.append(heros.rule_population.pop_set[i])
                self.rule_IDs.append(heros.rule_population.pop_set[i].ID)
            self.birth_iteration = heros.model_iteration
        elif len(self.rule_IDs) == 1: # Addition and Swap Only (to avoid empty models)
            if random.random() < heros.mut_prob:
                other_rule_indexs = [] # rule position indexes in the rule population corresponding with other_rule_IDs
                #Identify all other rules in the population that may be candidates for addition to this model
                pop_index = 0
                for rule in heros.rule_population.pop_set:
                    if rule.ID not in self.rule_IDs:
                        #other_rule_IDs.append(rule.ID)
                        other_rule_indexs.append(pop_index)
                    pop_index += 1
                #Select the rule to add to this model
                random_rule_index = random.choice(other_rule_indexs)
                random_rule_ID = heros.rule_population.pop_set[random_rule_index].ID
                #Get the index to this rule within the rule population 
                if random.random() < 0.5: # Swap
                    self.rule_IDs.remove(self.rule_IDs[0])
                    del self.rule_set[0] #remove the one rule currently in the model
                # Addition
                self.rule_IDs.append(random_rule_ID)
                self.rule_set.append(heros.rule_population.pop_set[random_rule_index])
        else: # Addition, Deletion, or Swap 
            mutate_options = ['A','D','S'] #Add, delete, swap
            original_rule_IDs = []
            for id in self.rule_IDs:
                original_rule_IDs.append(id)
            for ruleID in original_rule_IDs:
                if random.random() < heros.mut_prob:
                    mutate_type = random.choice(mutate_options)
                    if mutate_type == 'D' or len(heros.rule_population.pop_set) <= len(self.rule_IDs):
                        rule_ID_index = self.rule_IDs.index(ruleID)
                        self.rule_IDs.remove(ruleID)
                        del self.rule_set[rule_ID_index]
                    else: #swap or add
                        other_rule_IDs = [] # rule ids from the rule population that are not in the current model
                        other_rule_indexs = [] # rule position indexes in the rule population corresponding with other_rule_IDs
                        #Identify all other rules in the population that may be candidates for addition to this model
                        pop_index = 0
                        for rule in heros.rule_population.pop_set:
                            if rule.ID not in self.rule_IDs and rule.ID not in original_rule_IDs:
                                other_rule_IDs.append(rule.ID)
                                other_rule_indexs.append(pop_index)
                            pop_index += 1
                        #Select the rule to add to this model
                        if len(other_rule_IDs) != 0:
                            random_rule_ID = random.choice(other_rule_IDs) #the id of a new rule to add to this model
                            #Get the index to this rule within the rule population 
                            random_rule_index = other_rule_IDs.index(random_rule_ID) 
                            random_rule_pop_index = other_rule_indexs[random_rule_index]
                            if mutate_type == 'S': # Swap
                                rule_ID_index = self.rule_IDs.index(ruleID)
                                self.rule_IDs.remove(ruleID)
                                del self.rule_set[rule_ID_index] #remove the one rule currently in the model
                                self.rule_IDs.append(random_rule_ID)
                                self.rule_set.append(heros.rule_population.pop_set[random_rule_pop_index])
                            elif mutate_type == 'A': # Addition
                                self.rule_IDs.append(random_rule_ID)
                                self.rule_set.append(heros.rule_population.pop_set[random_rule_pop_index])


    def mutation_acc_pressure(self,random,heros):
        """ Applies muation to this model """
        if len(self.rule_IDs) == 0: #Initialize new model if empty after crossover
            target_rule_min = 10
            target_rule_max = int(len(heros.rule_population.pop_set)/2)
            if target_rule_max < target_rule_min:
                min_rules = 2
                max_rules = len(heros.rule_population.pop_set)
            else:
                min_rules = target_rule_min
                max_rules = int(len(heros.rule_population.pop_set)/2)
            rules_in_model = random.randint(min_rules,max_rules)
            self.rule_set = random.choices(heros.rule_population.pop_set, weights=[obj.useful_accuracy for obj in heros.rule_population.pop_set],k=rules_in_model)
            for rule in self.rule_set:
                self.rule_IDs.append(rule.ID)
            self.birth_iteration = heros.model_iteration
        elif len(self.rule_IDs) == 1: # Addition and Swap Only (to avoid empty models)
            if random.random() < heros.mut_prob:
                other_rule_IDs = [] # rule ids from the rule population that are not in the current model
                other_rule_indexs = [] # rule position indexes in the rule population corresponding with other_rule_IDs
                #Identify all other rules in the population that may be candidates for addition to this model
                pop_index = 0
                for rule in heros.rule_population.pop_set:
                    if rule.ID not in self.rule_IDs:
                        other_rule_indexs.append(pop_index)
                    pop_index += 1
                #Select the rule to add to this model
                random_rule_index = random.choices(other_rule_indexs, weights=[heros.rule_population.pop_set[i].useful_accuracy for i in other_rule_indexs],k=1)[0] #the id of a new rule to add to this model
                random_rule_ID = heros.rule_population.pop_set[random_rule_index].ID
                if random.random() < 0.5: # Swap
                    self.rule_IDs.remove(self.rule_IDs[0])
                    del self.rule_set[0] #remove the one rule currently in the model
                # Addition
                self.rule_IDs.append(random_rule_ID)
                self.rule_set.append(heros.rule_population.pop_set[random_rule_index])
        else: # Addition, Deletion, or Swap 
            mutate_options = ['A','D','S'] #Add, delete, swap
            original_rule_IDs = []
            for id in self.rule_IDs:
                original_rule_IDs.append(id)
            for ruleID in original_rule_IDs:
                if random.random() < heros.mut_prob:
                    mutate_type = random.choice(mutate_options)
                    if mutate_type == 'D' or len(heros.rule_population.pop_set) <= len(self.rule_IDs):
                        rule_ID_index = self.rule_IDs.index(ruleID)
                        self.rule_IDs.remove(ruleID)
                        del self.rule_set[rule_ID_index]
                    else: #swap or add
                        other_rule_IDs = [] # rule ids from the rule population that are not in the current model
                        other_rule_indexs = [] # rule position indexes in the rule population corresponding with other_rule_IDs
                        other_rule_acc = []
                        #Identify all other rules in the population that may be candidates for addition to this model
                        pop_index = 0
                        for rule in heros.rule_population.pop_set:
                            if rule.ID not in self.rule_IDs and rule.ID not in original_rule_IDs:
                                other_rule_IDs.append(rule.ID)
                                other_rule_indexs.append(pop_index)
                                other_rule_acc.append(rule.useful_accuracy)
                            pop_index += 1
                        #Select the rule to add to this model
                        if len(other_rule_IDs) != 0:
                            random_rule_ID = random.choices(other_rule_IDs, weights=[i for i in other_rule_acc],k=1)[0]
                            #random_rule_ID = random.choice(other_rule_IDs) #the id of a new rule to add to this model
                            #Get the index to this rule within the rule population 
                            random_rule_index = other_rule_IDs.index(random_rule_ID) #error
                            random_rule_pop_index = other_rule_indexs[random_rule_index]
                            if mutate_type == 'S': # Swap
                                rule_ID_index = self.rule_IDs.index(ruleID)
                                self.rule_IDs.remove(ruleID)
                                del self.rule_set[rule_ID_index] #remove the one rule currently in the model
                                self.rule_IDs.append(random_rule_ID)
                                self.rule_set.append(heros.rule_population.pop_set[random_rule_pop_index])
                            elif mutate_type == 'A': # Addition
                                self.rule_IDs.append(random_rule_ID)
                                self.rule_set.append(heros.rule_population.pop_set[random_rule_pop_index])


    def show_model(self):
        """ Report basic information regarding this model. """
        print("Set-------------------------------------------")
        print("Rule IDs: " + str(self.rule_IDs))
        print("Fitness: " + str(self.fitness))
        print("Accuracy: " + str(self.accuracy))
        #print("Useful Accuracy: " +str(self.useful_accuracy))
        print("Coverage: " + str(self.coverage))
        print("Birth Iteration: " + str(self.birth_iteration))
        print("Rule Count: " + str(len(self.rule_IDs))) 
        print("Deletion Prob: " + str(self.deletion_prob))


    def plot_rule_pop_network(self, feature_names, heros, weighting='useful_accuracy', display_micro=False, node_size=1000, edge_size=10, show=True, save=False, output_path=None, data_name=None):
        """ Plots a network visualization of the rule population with feature specificity across rules as node size and feature co-specificity 
            across rules in the population as edge size.
        """
        # Initialize dictionaries to count the number of times each feature is specified in rules of the population and how often feature combinations are cospecified
        feat_spec_count = defaultdict(int)
        feat_cooccurrence_count = defaultdict(int)
        #Create dictionaries of specificity counts
        for rule in self.rule_set:
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
            plt.savefig(output_path+'/'+data_name+'_rule_pop_network.png', bbox_inches="tight")
        if show:
            plt.show()


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
            rule_spec_df = pd.DataFrame([[0.0] * heros.env.num_feat for _ in range(len(self.rule_set))])
            rule_weight_df = pd.DataFrame([[0.0] * heros.env.num_feat for _ in range(len(self.rule_set))])
        # Add feature names as the dataframe columns
        rule_weight_df.columns = feature_names
        rule_spec_df.columns = feature_names
        # Add feature specificities (and weights if selected) to this dataframe
        row = 0
        for rule in self.rule_set:
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
