import copy
import pandas as pd
import random
import numpy as np
from collections import defaultdict

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
        # FLEXIBLE MODEL PARAMETERS ***************************************************
        self.fitness = None #model 'goodness' metric that drives many aspects of algorithm learning, discovery, and prediction
        self.deletion_prob = None #probability of model being selected for deletion

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
        #self.check_for_duplicates(self.rule_IDs, "Initialize")

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
                    class_counts[rule.action] += 1 #rule.useful_accuracy
                else:
                    class_counts[rule.action] = 1 #rule.useful_accuracy
        if len(class_counts.keys()) > 0: 
            #pred = max(class_counts, key = lambda x: class_counts[x]) #Problematic, since ties lead to tie-based wins, not authentic ones
            max_count = max(class_counts.values())
            tied_classes = [k for k, v in class_counts.items() if v == max_count]
            if len(tied_classes) > 1:
                pred = None #Will predict a wrong class to discourge ties
            else:
                pred = tied_classes[0]
            return True, pred 
        else: 
            return False, None # instance not covered by model

    def complete_evaluation_class(self,heros):
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
        if heros.fitness_function == 'accuracy':
            self.fitness = pow(self.accuracy, heros.nu)
        elif heros.fitness_function == 'pareto':
            # Check if new cloud updates pareto front
            front_updated = heros.model_pareto.update_front(self.accuracy, len(self.rule_set), ['max', 'min'])
            #Calculate and update the rule fitness
            self.fitness = heros.model_pareto.get_pareto_fitness(self.accuracy, len(self.rule_set), False,heros)
            if self.fitness == None: 
                self.fitness = pow(self.accuracy, heros.nu)
            if front_updated:
                self.show_model() #debug
                return True
        else:
            print("Fitness metric not available")
        return False

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
        #self.check_for_duplicates(self.rule_IDs, "Merge")

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

        #self.check_for_duplicates(self.rule_IDs, "Crossover")

    def mutation(self,random,heros):
        """ Applies muation to this model """
        if len(self.rule_IDs) == 0: #Initialize new model if empty after crossover
            rules_in_model = random.randint(10,int(len(heros.rule_population.pop_set)/2))
            index_list = list(range(len(heros.rule_population.pop_set)))
            rule_indexes = random.sample(index_list,rules_in_model)
            for i in rule_indexes:
                self.rule_set.append(heros.rule_population.pop_set[i])
                self.rule_IDs.append(heros.rule_population.pop_set[i].ID)
            self.birth_iteration = heros.model_iteration
        elif len(self.rule_IDs) == 1: # Addition and Swap Only (to avoid empty models)
            for ruleID in self.rule_IDs:
                if random.random() < heros.mut_prob:
                    other_rule_IDs = [] # rule ids from the rule population that are not in the current model
                    other_rule_indexs = [] # rule position indexes in the rule population corresponding with other_rule_IDs
                    #Identify all other rules in the population that may be candidates for addition to this model
                    pop_index = 0
                    for rule in heros.rule_population.pop_set:
                        if rule.ID not in self.rule_IDs:
                            other_rule_IDs.append(rule.ID)
                            other_rule_indexs.append(pop_index)
                        pop_index += 1
                    #Select the rule to add to this model
                    random_rule_ID = random.choice(other_rule_IDs) #the id of a new rule to add to this model
                    #Get the index to this rule within the rule population 
                    random_rule_index = other_rule_IDs.index(random_rule_ID) 
                    random_rule_pop_index = other_rule_indexs[random_rule_index]
                    if random.random() < 0.5: # Swap
                        self.rule_IDs.remove(ruleID)
                        del self.rule_set[0] #remove the one rule currently in the model
                    # Addition
                    self.rule_IDs.append(random_rule_ID)
                    self.rule_set.append(heros.rule_population.pop_set[random_rule_pop_index])
        else: # Addition, Deletion, or Swap 
            mutate_options = ['A','D','S'] #Add, delete, swap
            original_rule_IDs = []
            for id in self.rule_IDs:
                original_rule_IDs.append(id)
            #print(original_rule_IDs)
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
        #self.check_for_duplicates(self.rule_IDs, "Mutation")

    def update_fitness(self,heros):
        """ Updates the rule fitness as a result of an update to the pareto front."""
        self.fitness = heros.model_pareto.get_pareto_fitness(self.accuracy, len(self.rule_set), False,heros)
        if self.fitness == None: #
            self.fitness = pow(self.accuracy, heros.nu)

    def update_deletion_prob(self, deletion_prob):
        """ Updates model deletion probability during deletion"""
        self.deletion_prob = deletion_prob

    def show_model(self):
        """ Report basic information regarding this model. """
        print("Set-------------------------------------------")
        print("Rule IDs: " + str(self.rule_IDs))
        print("Fitness: " + str(self.fitness))
        print("Accuracy: " + str(self.accuracy))
        print("Coverage: " + str(self.coverage))
        print("Birth Iteration: " + str(self.birth_iteration))
        print("Rule Count: " + str(len(self.rule_IDs))) 
        print("Deletion Prob: " + str(self.deletion_prob))

