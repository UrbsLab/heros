import copy
import pandas as pd
from .model import MODEL

class MODEL_POP: 
    def __init__(self):
        """ Initializes ruleset population objects. """
        self.pop_set = []
        self.offspring_pop = []
        self.top_rule_set = []
        self.match_set = []

    def make_eval_match_set(self,instance_state,heros):
        """ Makes a match set {M} given an instance state. Used by predict function."""
        for i in range(len(self.top_rule_set)):
            rule = self.top_rule_set[i]
            if rule.match(instance_state,heros):
                self.match_set.append(i)

    def clear_sets(self):
        """ """
        self.match_set = []

    def initialize_model_population(self,heros,random):
        """ Randomly initialize a model population. """
        front_updated_global = False
        while len(self.pop_set) < heros.model_pop_size:
            #Generate new model
            new_model = MODEL()
            rules_in_model = random.randint(10,int(len(heros.rule_population.pop_set)/2))
            new_model.initialize_randomly(rules_in_model,heros)
            #Check if model already in population, and add to population
            if not self.model_exists(new_model):
                # Evalute model and update model parameters
                front_updated = new_model.complete_evaluation_class(heros)
                if not front_updated_global:
                    front_updated_global = front_updated
                #Add model to population
                self.pop_set.append(new_model) #add to model population
        if heros.fitness_function == 'pareto' and front_updated_global:
            self.global_fitness_update(heros)

    def model_exists(self, new_model):
        """ Identifies if an identical set already exists in the population. Relies on the unique 'ID' of rules within the rule set. """
        for model in self.pop_set:
            if set(model.rule_IDs) == set(new_model.rule_IDs):
                return True
        for model in self.offspring_pop:
            if set(model.rule_IDs) == set(new_model.rule_IDs):
                return True
        return False

    def select_parent_pair(self,theta_sel,random):
        #Tournament Selection
        tSize = int(len(self.pop_set) * theta_sel) #Tournament Size
        parent_1 = self.tournament_selection(tSize,random)
        parent_2 = self.tournament_selection(tSize,random)
        while parent_1 == parent_2:
            parent_2 = self.tournament_selection(tSize,random)
        return [parent_1,parent_2]
    
    def tournament_selection(self,tSize,random):
        random.shuffle(self.pop_set)
        new_parent = max(self.pop_set[:tSize], key=lambda x: x.fitness)
        return new_parent
    
    def generate_offspring(self,iteration,parent_list,random,heros):
        random_gen_tries = 2 #Hard coded number of random model generation attempts
        random_gen_rule_min = 5 #Hard coded minimum number of rules in a randomly generated model
        front_updated_global = False
        # Clone Parents
        offspring_1 = MODEL()
        offspring_2 = MODEL()
        offspring_1.copy_parent(parent_list[0],iteration)
        offspring_2.copy_parent(parent_list[1],iteration)
        if random.random() < heros.merge_prob: #Generate a single novel bin that is the combination of the two parent bins (yielding 3 total bins created during this mating)
            offspring_3 = MODEL()
            offspring_3.copy_parent(parent_list[0],iteration)
            offspring_3.merge(parent_list[1])
            # If model already exists, generate a random one with a larger rule-set size range
            try_counter = 0
            while self.model_exists(offspring_3) and try_counter < random_gen_tries: 
                rules_in_model = random.randint(random_gen_rule_min,int(len(heros.rule_population.pop_set)))
                offspring_3.initialize_randomly(rules_in_model,heros)
                try_counter += 1
            # Evalute model and update model parameters
            if not self.model_exists(offspring_3):
                front_updated = offspring_3.complete_evaluation_class(heros)
                if not front_updated_global:
                    front_updated_global = front_updated
                #Add model to offspring population
                self.offspring_pop.append(offspring_3) #add to model population
        # Crossover
        if random.random() < heros.cross_prob:
            offspring_1.uniform_crossover(offspring_2,random)
        # Mutation - check for duplicate rules
        offspring_1.mutation(random,heros)
        offspring_2.mutation(random,heros)

        # Offspring 1 Checks -----------------
        #Check for Empty Model
        offspring_1_empty = False
        if len(offspring_1.rule_set) == 0:
            offspring_1_empty = True
        # If model is empty or already exists, generate a random one with a larger rule-set size range
        try_counter = 0
        while offspring_1_empty or (self.model_exists(offspring_1) and try_counter < random_gen_tries): 
            rules_in_model = random.randint(random_gen_rule_min,int(len(heros.rule_population.pop_set)))
            offspring_1.initialize_randomly(rules_in_model,heros)
            try_counter += 1
            offspring_1_empty = False
        # Evalute model and update model parameters
        if not self.model_exists(offspring_1):
            front_updated = offspring_1.complete_evaluation_class(heros)
            if not front_updated_global:
                front_updated_global = front_updated
            #Add model to offspring population
            self.offspring_pop.append(offspring_1) #add to model population

        # Offspring 2 Checks -----------------
        offspring_2_empty = False
        if len(offspring_2.rule_set) == 0:
            offspring_2_empty = True
        # If model is empty or already exists, generate a random one with a larger rule-set size range
        try_counter = 0
        while offspring_2_empty or (self.model_exists(offspring_2) and try_counter < random_gen_tries): 
            rules_in_model = random.randint(random_gen_rule_min,int(len(heros.rule_population.pop_set)))
            offspring_2.initialize_randomly(rules_in_model,heros)
            try_counter += 1
            offspring_2_empty = False
        # Evalute model and update model parameters
        if not self.model_exists(offspring_2):
            front_updated = offspring_2.complete_evaluation_class(heros)
            if not front_updated_global:
                front_updated_global = front_updated
            #Add model to offspring population
            self.offspring_pop.append(offspring_2) #add to model population
        return front_updated_global

    def add_offspring_into_pop(self):
        self.pop_set = self.pop_set + self.offspring_pop
        self.offspring_pop = []

    def global_fitness_update(self,heros):
        """ Relevant for pareto-front rule fitness. Updates the fitness of all rules in the population if the pareto front gets updated. """
        for model in self.pop_set:
            model.update_fitness(heros)

    def probabilistic_model_deletion(self,heros,random):
        """ """
        # Automatically delete bins with a fitness of 0
        delete_indexes = []
        i = 0
        for model in self.pop_set:
            if model.fitness == 0 and len(delete_indexes)<(len(self.pop_set)-heros.model_pop_size):
                delete_indexes.append(i)
            i += 1
        delete_indexes.sort(reverse=True) #sort in descending order so deletion does not affect subsequent indexes
        for index in delete_indexes:
            del self.pop_set[index]
        #until population size is reduced to it's maximum, calculate deletion probabilities and delete one model at a time with roulette wheel selection
        while len(self.pop_set) > heros.model_pop_size:
            #Calculate total fitness across all bins
            total_fitness = 0
            for model in self.pop_set:
                if model.fitness < 1:
                    total_fitness += 1/float(model.fitness)
            deletion_probabilities = []
            for model in self.pop_set:
                if model.fitness == 1:
                    deletion_probabilities.append(0)
                    model.update_deletion_prob(0)
                else:
                    deletion_probabilities.append(1/float(model.fitness))
                    model.update_deletion_prob(1/float(model.fitness))
            if sum(deletion_probabilities) == 0:
                break
            else:
                index = random.choices(range(len(self.pop_set)), weights=deletion_probabilities)[0]
                del self.pop_set[index]

    def export_model_population(self):
        """ Prepares and exports a dataframe capturing the rule population. """
        pop_list = []
        column_names = ['Rule IDs', 
                        'Number of Rules',
                        'Fitness', 
                        'Accuracy',
                        'Coverage', 
                        'Birth Iteration', 
                        'Deletion Probability']
        for model in self.pop_set: 
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
    

    def custom_sort_key(self, obj):
        return (-obj.accuracy,-obj.coverage,len(obj.rule_IDs))
    
    def sort_model_pop(self):
        self.pop_set = sorted(self.pop_set, key=self.custom_sort_key) 

    def get_top_model(self):
        """ Identifies the top model (assumes previous pop_set sorting), and extracts it's rules into self.top_rule_set """
        for rule in self.pop_set[0].rule_set:
            self.top_rule_set.append(rule)
    
    def export_top_model(self):
        """ Prepares and exports a dataframe capturing the top model, i.e. rule set."""
        set_list = []
        column_names = ['Condition Indexes',
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
        for rule in self.top_rule_set:
            rule_list = [rule.condition_indexes,
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
            set_list.append(rule_list)
        set_df = pd.DataFrame(set_list,columns=column_names)
        return set_df


    def export_indexed_model(self,index):
        """ """
        indexed_rule_set = []
        for rule in self.pop_set[index].rule_set:
           indexed_rule_set.append(rule)
        """ Prepares and exports a dataframe capturing the top model, i.e. rule set."""
        set_list = []
        column_names = ['Condition Indexes',
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
        for rule in indexed_rule_set:
            rule_list = [rule.condition_indexes,
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
            set_list.append(rule_list)
        set_df = pd.DataFrame(set_list,columns=column_names)
        return set_df