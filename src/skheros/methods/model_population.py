import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .model import MODEL


class MODEL_POP: 
    def __init__(self):
        """ Initializes ruleset population objects. """
        self.pop_set = []
        self.offspring_pop = []
        self.match_set = []
        self.target_rule_set = []
        self.pop_set_archive = {}
        self.pop_set_hold = None
        self.explored_models = set() #Efficient storage of all explored models for referencing during algorithm run
        self.archive_discovered_rules = True

    def skip_phase2(self,heros):
        """ Creates a model out of entire rule population. Used to effectively skip phase 2 optimization when rule population was very small (< 2) after cleaning/compaction """
        new_model = MODEL()
        new_model.initialize_all_rule_model(heros)
        # Evalute model and update model parameters
        new_model.evaluate_model_class(heros)
        self.pop_set.append(new_model) #add to model population
        self.pop_set[0].model_on_front = True


    def archive_model_pop(self,iteration):
        self.pop_set_archive[int(iteration)] = copy.deepcopy(self.pop_set)


    def change_model_pop(self,iteration):
        self.pop_set_hold = copy.deepcopy(self.pop_set)
        self.pop_set = self.pop_set_archive[int(iteration)]


    def restore_model_pop(self):
        self.pop_set = self.pop_set_hold
        self.pop_set_hold = None


    def make_eval_match_set(self,instance_state,heros):
        """ Makes a match set {M} given an instance state. Used by predict function."""
        for i in range(len(self.target_rule_set)):
            rule = self.target_rule_set[i]
            if rule.match(instance_state,heros):
                self.match_set.append(i)


    def clear_sets(self):
        """ """
        self.match_set = []


    def dominates(self,p,q):
        """Check if p dominates q. A model dominates another if it has a more optimal value for at least one objective."""
        objective_directions = ['max', 'min'] #maximize balanced accuracy and minimize rule-set size
        better_in_all_objectives = True
        better_in_at_least_one_objective = False
        for val1, val2, obj in zip(p.objectives, q.objectives, objective_directions):
            if obj == 'max':
                if val1 < val2:
                    better_in_all_objectives = False
                if val1 > val2:
                    better_in_at_least_one_objective = True
            elif obj == 'min':
                if val1 > val2:
                    better_in_all_objectives = False
                if val1 < val2:
                    better_in_at_least_one_objective = True
            else:
                raise ValueError("Objectives must be 'max' or 'min'")
        return better_in_all_objectives and better_in_at_least_one_objective


    def fast_non_dominated_sort(self,heros):
        """NSGA-II-like fast non dominated sorting of solutions (i.e. models) into a series of non-dominated fronts."""
        fronts = [[]]
        domination_counts = {sol: 0 for sol in self.pop_set}
        dominated_solutions = {sol: [] for sol in self.pop_set}
        #Cout dominated solutions and counts for each solution
        for p in self.pop_set:
            for q in self.pop_set:
                if self.dominates(p, q):
                    dominated_solutions[p].append(q)
                elif self.dominates(q, p):
                    domination_counts[p] += 1
            if domination_counts[p] == 0:
                fronts[0].append(p)
        # Determine set on non-dominated fronts based on dominated solutions and counts
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        return fronts[:-1]


    def calculate_crowding_distance(self,front):
        """Assigns crowding distance to solutions within a Pareto front."""
        num_solutions = len(front)
        if num_solutions == 0:
            return
        distances = {sol: 0 for sol in front}
        num_objectives = len(front[0].objectives)
        for m in range(num_objectives):
            front.sort(key=lambda sol: sol.objectives[m])
            distances[front[0]] = distances[front[-1]] = float('inf')
            min_obj = front[0].objectives[m]
            max_obj = front[-1].objectives[m]
            if max_obj == min_obj:
                continue  # Avoid division by zero
            for i in range(1, num_solutions - 1):
                distances[front[i]] += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / (max_obj - min_obj)
        
        return distances


    def binary_tournament_selection(self,crowding_distances,random):
        """Selects a solution using binary tournament selection."""
        #Edge case catch (for very small rule populations, which leads to very small model populations)
        if len(self.pop_set) == 1:
            return self.pop_set[0]

        a, b = random.sample(self.pop_set, 2)
        if self.dominates(a, b):
            return a
        elif self.dominates(b, a):
            return b
        elif a in crowding_distances and b in crowding_distances:
            if crowding_distances[a] > crowding_distances[b]:
                return a
            elif crowding_distances[a] < crowding_distances[b]:
                return b
            else:
                return random.choice([a,b])
        else:
            return random.choice([a,b])


    def add_new_explored_model(self,new_list,list_collection):
        """Add a new list to the collection, ignoring order."""
        list_collection.add(frozenset(new_list)) # Convert to frozenset before adding


    def clear_explored_models(self):
        self.explored_models = None


    def list_exists(self, new_list, list_collection):
        """Check if a list (unordered) exists in the collection."""
        return frozenset(new_list) in list_collection  # O(1) lookup on average


    def initialize_model_population(self,heros,random,model_pop_init):
        """ Randomly initialize a model population. """
        failed_attempts_max = 100
        fail_count = 0
        #Determine min and max rule counts for initialized models 
        target_rule_min = 10
        target_rule_max = int(len(heros.rule_population.pop_set)/2)
        if target_rule_max < target_rule_min:
            min_rules = 1
            max_rules = len(heros.rule_population.pop_set)
        else:
            min_rules = target_rule_min
            max_rules = int(len(heros.rule_population.pop_set)/2)
        #Initialize methods --------------
        if model_pop_init == "random":
            while len(self.pop_set) < heros.model_pop_size and fail_count < failed_attempts_max:
                new_model = MODEL()
                rules_in_model = random.randint(min_rules,max_rules)
                new_model.initialize_randomly(rules_in_model,heros)
                #Check if model already in population, and add to population
                if self.archive_discovered_rules:
                    if not self.list_exists(new_model.rule_IDs, self.explored_models):
                        # Evalute model and update model parameters
                        new_model.evaluate_model_class(heros)
                        #Add model to population
                        self.pop_set.append(new_model) #add to model population
                        self.add_new_explored_model(new_model.rule_IDs, self.explored_models)
                    else:
                        fail_count += 1
                else: #No model archiving
                    if not self.model_exists(new_model):
                        # Evalute model and update model parameters
                        new_model.evaluate_model_class(heros)
                        #Add model to population
                        self.pop_set.append(new_model) #add to model population
                        self.add_new_explored_model(new_model.rule_IDs, self.explored_models)
                    else:
                        fail_count += 1

        elif model_pop_init == 'target_acc':
            # Paramters for 'target_acc' init
            if heros.nu > 1:
                target_list = [1.0] * int(heros.model_pop_size/5.0)
            else:
                min_accuracy = 0.55
                target_list = np.linspace(min_accuracy,1.0,int(heros.model_pop_size/5.0)).tolist() #aim for 5 bins to be initialized for each target accuracy
                target_list.reverse() #start by creating a model with maximally accurate rules
            target_list_counter = 0
            while len(self.pop_set) < heros.model_pop_size and fail_count < failed_attempts_max:
                new_model = MODEL()
                rules_in_model = random.randint(min_rules,max_rules)
                new_model.initialize_target(rules_in_model, target_list[target_list_counter], heros)
                if target_list_counter > int(heros.model_pop_size/5.0) - 2:
                    target_list_counter = 0
                else:
                    target_list_counter += 1
                #Check if model already in population, and add to population
                if self.archive_discovered_rules:
                    if not self.list_exists(new_model.rule_IDs, self.explored_models):
                        # Evalute model and update model parameters
                        new_model.evaluate_model_class(heros)
                        #Add model to population
                        self.pop_set.append(new_model) #add to model population
                        self.add_new_explored_model(new_model.rule_IDs, self.explored_models)
                    else:
                        fail_count += 1
                else: #No model archiving
                    if not self.model_exists(new_model):
                        # Evalute model and update model parameters
                        new_model.evaluate_model_class(heros)
                        #Add model to population
                        self.pop_set.append(new_model) #add to model population
                        self.add_new_explored_model(new_model.rule_IDs, self.explored_models)
                    else:
                        fail_count += 1
        else:
            print("Specified model initialization method not available.")


    def model_exists(self, new_model):
        """ Identifies if an identical set already exists in the population. Relies on the unique 'ID' of rules within the rule set. """
        for model in self.pop_set:
            if set(model.rule_IDs) == set(new_model.rule_IDs):
                return True
        for model in self.offspring_pop:
            if set(model.rule_IDs) == set(new_model.rule_IDs):
                return True
        return False


    def generate_offspring(self,iteration,parent_list,random,heros):
        random_gen_tries = 2 #Hard coded number of random model generation attempts
        random_gen_rule_min = 5 #Hard coded minimum number of rules in a randomly generated model
        if len(heros.rule_population.pop_set) < random_gen_rule_min: #Catches potential error if rule population size ends up to be very small after rule compaction.
            random_gen_rule_min = 1
        new_model_count = 0
        # Clone Parents
        offspring_1 = MODEL()
        offspring_2 = MODEL()
        offspring_1.copy_parent(parent_list[0],iteration)
        offspring_2.copy_parent(parent_list[1],iteration)
        if random.random() < heros.merge_prob: #Generate a single novel model that is the combination of the two parent models (yielding 3 total models created during this mating)
            offspring_3 = MODEL()
            offspring_3.copy_parent(parent_list[0],iteration)
            offspring_3.merge(parent_list[1])
            # If model already exists, generate a random one with a larger rule-set size range
            try_counter = 0
            if self.archive_discovered_rules:
                while self.list_exists(offspring_3.rule_IDs, self.explored_models) and try_counter < random_gen_tries: 
                    rules_in_model = random.randint(random_gen_rule_min,int(len(heros.rule_population.pop_set)))
                    if heros.model_pop_init == "random":
                        offspring_3.initialize_randomly(rules_in_model,heros)
                    elif heros.model_pop_init == "target_acc":
                        if heros.nu > 1: #Pressure to be highly accurate - revert to using random init
                            offspring_3.initialize_target(rules_in_model,1.0, heros)
                        else:
                            target = random.uniform(0.55,1.0)
                            offspring_3.initialize_target(rules_in_model,target, heros)
                    else: 
                        print("Specified model initialization method not available.")
                    try_counter += 1
            else: # no archiving
                while self.model_exists(offspring_3) and try_counter < random_gen_tries: 
                    rules_in_model = random.randint(random_gen_rule_min,int(len(heros.rule_population.pop_set)))
                    if heros.model_pop_init == "random":
                        offspring_3.initialize_randomly(rules_in_model,heros)
                    elif heros.model_pop_init == "target_acc":
                        if heros.nu > 1: #Pressure to be highly accurate - revert to using random init
                            offspring_3.initialize_target(rules_in_model,1.0, heros)
                        else:
                            target = random.uniform(0.55,1.0)
                            offspring_3.initialize_target(rules_in_model,target, heros)
                    else: 
                        print("Specified model initialization method not available.")
                    try_counter += 1
            # Evalute model and update model parameters
            if self.archive_discovered_rules:
                if not self.list_exists(offspring_3.rule_IDs, self.explored_models):
                    offspring_3.evaluate_model_class(heros)
                    #Add model to offspring population
                    self.offspring_pop.append(offspring_3) #add to model population
                    self.add_new_explored_model(offspring_3.rule_IDs, self.explored_models)
                    new_model_count += 1
            else: #no archiving
                if not self.model_exists(offspring_3):
                    offspring_3.evaluate_model_class(heros)
                    #Add model to offspring population
                    self.offspring_pop.append(offspring_3) #add to model population
                    self.add_new_explored_model(offspring_3.rule_IDs, self.explored_models)
                    new_model_count += 1
        # Crossover
        if random.random() < heros.cross_prob:
            offspring_1.uniform_crossover(offspring_2,random)
        # Mutation - check for duplicate rules
        if heros.nu > 1:
            offspring_1.mutation_acc_pressure(random,heros)
            offspring_2.mutation_acc_pressure(random,heros)
        else:
            offspring_1.mutation(random,heros)
            offspring_2.mutation(random,heros)
        # Offspring 1 Checks -----------------
        #Check for Empty Model
        offspring_1_empty = False
        if len(offspring_1.rule_set) == 0:
            offspring_1_empty = True
        # If model is empty or already exists, generate a random one with a larger rule-set size range
        try_counter = 0
        if self.archive_discovered_rules:
            while offspring_1_empty or (self.list_exists(offspring_1.rule_IDs, self.explored_models) and try_counter < random_gen_tries): 
                rules_in_model = random.randint(random_gen_rule_min,int(len(heros.rule_population.pop_set)))
                if heros.model_pop_init == "random":
                    offspring_1.initialize_randomly(rules_in_model,heros)
                elif heros.model_pop_init == "target_acc":
                    if heros.nu > 1: #Pressure to be highly accurate - revert to using random init
                        offspring_1.initialize_target(rules_in_model,1.0, heros)
                    else:
                        target = random.uniform(0.55,1.0)
                        offspring_1.initialize_target(rules_in_model,target, heros)
                else: 
                    print("Specified model initialization method not available.")
                try_counter += 1
                offspring_1_empty = False
        else: #no archiving
            while offspring_1_empty or (self.model_exists(offspring_1) and try_counter < random_gen_tries):
                rules_in_model = random.randint(random_gen_rule_min,int(len(heros.rule_population.pop_set)))
                if heros.model_pop_init == "random":
                    offspring_1.initialize_randomly(rules_in_model,heros)
                elif heros.model_pop_init == "target_acc":
                    if heros.nu > 1: #Pressure to be highly accurate - revert to using random init
                        offspring_1.initialize_target(rules_in_model,1.0, heros)
                    else:
                        target = random.uniform(0.55,1.0)
                        offspring_1.initialize_target(rules_in_model,target, heros)
                else: 
                    print("Specified model initialization method not available.")
                try_counter += 1
                offspring_1_empty = False
        # Evalute model and update model parameters
        if self.archive_discovered_rules:
            if not self.list_exists(offspring_1.rule_IDs, self.explored_models):
                offspring_1.evaluate_model_class(heros)
                #Add model to offspring population
                self.offspring_pop.append(offspring_1) #add to model population
                self.add_new_explored_model(offspring_1.rule_IDs, self.explored_models)
                new_model_count += 1
        else: # no archiving
            if not self.model_exists(offspring_1):
                offspring_1.evaluate_model_class(heros)
                #Add model to offspring population
                self.offspring_pop.append(offspring_1) #add to model population
                self.add_new_explored_model(offspring_1.rule_IDs, self.explored_models)
                new_model_count += 1

        # Offspring 2 Checks -----------------
        offspring_2_empty = False
        if len(offspring_2.rule_set) == 0:
            offspring_2_empty = True
        # If model is empty or already exists, generate a random one with a larger rule-set size range
        try_counter = 0
        if self.archive_discovered_rules:
            while offspring_2_empty or (self.list_exists(offspring_2.rule_IDs, self.explored_models) and try_counter < random_gen_tries): 
                rules_in_model = random.randint(random_gen_rule_min,int(len(heros.rule_population.pop_set)))
                if heros.model_pop_init == "random":
                    offspring_2.initialize_randomly(rules_in_model,heros)
                elif heros.model_pop_init == "target_acc":
                    if heros.nu > 1: #Pressure to be highly accurate - revert to using random init
                        offspring_2.initialize_target(rules_in_model,1.0, heros)
                    else:
                        target = random.uniform(0.55,1.0)
                        offspring_2.initialize_target(rules_in_model,target, heros)
                else: 
                    print("Specified model initialization method not available.")
                try_counter += 1
                offspring_2_empty = False
        else: #no archiving
            while offspring_2_empty or (self.model_exists(offspring_2) and try_counter < random_gen_tries): 
                rules_in_model = random.randint(random_gen_rule_min,int(len(heros.rule_population.pop_set)))
                if heros.model_pop_init == "random":
                    offspring_2.initialize_randomly(rules_in_model,heros)
                elif heros.model_pop_init == "target_acc":
                    if heros.nu > 1: #Pressure to be highly accurate - revert to using random init
                        offspring_2.initialize_target(rules_in_model,1.0, heros)
                    else:
                        target = random.uniform(0.55,1.0)
                        offspring_2.initialize_target(rules_in_model,target, heros)
                else: 
                    print("Specified model initialization method not available.")
                try_counter += 1
                offspring_2_empty = False
        # Evalute model and update model parameters
        if self.archive_discovered_rules:
            #if not self.model_exists(offspring_2):
            if not self.list_exists(offspring_2.rule_IDs, self.explored_models):
                offspring_2.evaluate_model_class(heros)
                #Add model to offspring population
                self.offspring_pop.append(offspring_2) #add to model population
                self.add_new_explored_model(offspring_2.rule_IDs, self.explored_models)
                new_model_count += 1
            return new_model_count > 0
        else: #no archiving
            if not self.model_exists(offspring_2):
                offspring_2.evaluate_model_class(heros)
                #Add model to offspring population
                self.offspring_pop.append(offspring_2) #add to model population
                self.add_new_explored_model(offspring_2.rule_IDs, self.explored_models)
                new_model_count += 1
            return new_model_count > 0

    def add_offspring_into_pop(self):
        self.pop_set = self.pop_set + self.offspring_pop
        self.offspring_pop = []


    def model_deletion(self,heros,fronts,crowding_distances):
        """ NSGAII-like model deletion"""
        #Add solutions from full best fronts that there is space for
        if len(self.pop_set) > heros.model_pop_size:
            new_pop_set = []
            i = 0
            while len(new_pop_set) + len(fronts[i]) <= heros.model_pop_size:# and i < len(fronts): 
                new_pop_set.extend(fronts[i])
                i += 1
            # Sort the last front by crowding distance and select remaining
            if len(new_pop_set) < heros.model_pop_size:# and i < len(fronts):
                last_front = sorted(fronts[i], key=lambda sol: crowding_distances[sol], reverse=True)
                new_pop_set.extend(last_front[:heros.model_pop_size - len(new_pop_set)])
            #Replace model population with best models that remain
            self.pop_set = new_pop_set


    def export_model_population(self):
        """ Prepares and exports a dataframe capturing the rule population. """
        pop_list = []
        column_names = ['Rule IDs', 
                        'Number of Rules',
                        'Fitness', 
                        'Accuracy',
                        'Coverage', 
                        'Birth Iteration', 
                        'Deletion Probability', 
                        'Model on Front']
        for model in self.pop_set: 
            model_list = [str(model.rule_IDs), 
                          len(model.rule_set), 
                          model.fitness, 
                          model.accuracy,
                        model.coverage, 
                        model.birth_iteration, 
                        model.deletion_prob,
                        model.model_on_front]
            pop_list.append(model_list)
        pop_df = pd.DataFrame(pop_list, columns = column_names)
        return pop_df
    

    def custom_sort_key(self, obj):
        return (-obj.accuracy,-obj.coverage,len(obj.rule_IDs))
    

    def sort_model_pop(self):
        self.pop_set = sorted(self.pop_set, key=self.custom_sort_key) 


    def get_max(self):
        #uses min since custom sort key does inverse ranking for sorting from max to min desired metrics
        #accuracy is most important followed by coverage and lastly length of rule-set.
        return min(self.pop_set, key = self.custom_sort_key) 


    def get_target_model(self, target_model):
        self.target_rule_set = []
        for rule in self.pop_set[target_model].rule_set:
            self.target_rule_set.append(rule)
    

    def identify_models_on_front(self):
        """ Identifies which models are on the model pareto front at the end of model training and updates the 'model_on_front' parameter accordingly. """
        pareto_front = self.fast_non_dominated_sort(self.pop_set)[0]
        for model in self.pop_set:
            if model in pareto_front:
                model.model_on_front = 1 #on front
            else:
                model.model_on_front = 0 #not on front


    def get_all_model_fronts(self):
        return self.fast_non_dominated_sort(self.pop_set)
    

    def export_top_training_model(self):
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
        for rule in self.top_training_rule_set:
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
        """ Prepares and exports a dataframe capturing the indexed model, i.e. rule set."""
        indexed_rule_set = []
        for rule in self.pop_set[index].rule_set:
           indexed_rule_set.append(rule)
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
    
    def plot_model_pareto_fronts(self,fronts,show, save, output_path):
        """Visualizes all Pareto fronts with different colors. In the NSGAII-style."""
        colors = plt.cm.viridis(np.linspace(0, 1, len(fronts)))  # Color gradient
        plt.figure(figsize=(10, 6))
        fronts.reverse()
        for i, front in enumerate(fronts):
            if len(front) == 0:
                continue
            # Compute crowding distance for current front
            crowding_distances = self.calculate_crowding_distance(front) 
            x_vals = [sol.objectives[1] for sol in front]
            y_vals = [sol.objectives[0] for sol in front]
            # Normalize crowding distances, ignoring inf values
            finite_crowding_distances = [crowding_distances[sol] for sol in front if crowding_distances[sol] != float('inf')]
            min_crowd = min(finite_crowding_distances, default=0)
            max_crowd = max(finite_crowding_distances, default=1)
            sizes = [150 if crowding_distances[sol] == float('inf') else 20 + 80 * (crowding_distances[sol] - min_crowd) / (max_crowd - min_crowd + 1e-6) for sol in front]
            # Sort by x-values to ensure proper line connection
            sorted_pairs = sorted(zip(x_vals, y_vals))
            sorted_x, sorted_y = zip(*sorted_pairs)
            # Plot lines connecting solutions on the same front
            plt.plot(sorted_x, sorted_y, color=colors[i], linestyle='-', alpha=0.7)
            # Plot solutions with sizes reflecting crowding distance
            custom_labels = []
            number = len(fronts)
            for _ in range(len(fronts)-1):
                custom_labels.append("Sub-Front "+str(number))
                number -= 1
            custom_labels.append("Non-Dominated Front")
            plt.scatter(x_vals, y_vals, s=sizes, color=colors[i], label=custom_labels[i], edgecolors='black', linewidth=0.8)
        plt.xlabel("Rule-Set Size")
        plt.ylabel("Coverage Penalized Balanced Accuracy")
        # Get the legend handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()
        # Reverse the order
        plt.legend(handles[::-1], labels[::-1],loc='lower right', fontsize='small')
        if save:
            plt.savefig(output_path+'/pareto_fronts_models.png', bbox_inches="tight")
        if show:
            plt.show()


