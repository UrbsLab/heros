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
        # Key rule population parameters --------------------------------------------------------------------------
        self.pop_set = []  # List rule objects making up the rule population
        self.match_set = []  # List of references to rules in population that make up a temporary match set (i.e. rules with 'IF' conditions matching current instance state)
        self.correct_set = []  # List of references to rules in population that make up correct set (i.e. rules with 'THEN' action matching current instance outcome)
        self.micro_pop_count = 0 # Number of rules in the population defined by the sum of individual rule numerosities (aka 'micro' population count)
        self.ID_counter = 0 # A unique id given to each new rule discovered (that isn't in the current rule population).
        # Population performance tracking -----------------------------------------------------------------------------
        self.pop_set_archive = {}
        self.pop_set_hold = None
        # Niche Update Lists (experimental) --------------------------------------------------------------------------
        self.new_rule_set = [] # Idexes of rules that were newly introduced in the current iteration
        self.rules_for_niche_update = [] # All unique rules across all niche updates that participated in a {C}
        # Rule Archiving (experimental) -------------------------------------------------------------------------------
        self.explored_rules = []
        self.archive_discovered_rules = False #True value is experimental


    def add_new_explored_rules(self,rule):
        """Stores the unique and essential information to reconstitute an explored rule without re-evaluation."""
        rule_entry = [rule.condition_indexes, rule.condition_values, rule.action, rule.instance_outcome_count, rule.ID, rule.birth_iteration]
        self.explored_rules.append(rule_entry)


    def clear_explored_rules(self):
        self.explored_rules = None


    def rule_exists(self, target_rule):
        """Checks the explored rules list to see if a given 'new' rule has been previously discovered and evaluated, returning that rule's reference in explored rules."""
        #print('test')
        #print(target_rule.condition_indexes)
        #print(target_rule.condition_values)

        for rule_summary in self.explored_rules:
            if self.equals(target_rule,rule_summary):
                #print(rule_summary[0])
                #print(rule_summary[1])
                return rule_summary
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


    def dominates(self,p,q):
        """Check if p dominates q. A rule dominates another if it has a more optimal value for at least one objective."""
        objective_directions = ['max', 'max'] #maximize useful accuracy and useful coverage
        better_in_all_objectives = True
        better_in_at_least_one_objective = False
        for val1, val2, obj in zip(self.pop_set[p].objectives, self.pop_set[q].objectives, objective_directions):
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
    

    def fast_non_dominated_sort_M(self):
        """NSGA-II-like fast non dominated sorting of rules into a series of non-dominated fronts."""
        fronts = [[]]
        domination_counts = {sol: 0 for sol in self.match_set}
        dominated_solutions = {sol: [] for sol in self.match_set}
        #Cout dominated solutions and counts for each solution
        for p in self.match_set:
            for q in self.match_set:
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


    def fast_non_dominated_sort_C(self, correct_set):
        """NSGA-II-like fast non dominated sorting of rules into a series of non-dominated fronts."""
        fronts = [[]]
        domination_counts = {sol: 0 for sol in correct_set}
        dominated_solutions = {sol: [] for sol in correct_set}
        #Cout dominated solutions and counts for each solution
        for p in correct_set:
            for q in correct_set:
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
        num_objectives = len(self.pop_set[front[0]].objectives)
        for m in range(num_objectives):
            front.sort(key=lambda sol: self.pop_set[sol].objectives[m])
            distances[front[0]] = distances[front[-1]] = float('inf')
            min_obj = self.pop_set[front[0]].objectives[m]
            max_obj = self.pop_set[front[-1]].objectives[m]
            if max_obj == min_obj:
                continue  # Avoid division by zero
            for i in range(1, num_solutions - 1):
                distances[front[i]] += (self.pop_set[front[i + 1]].objectives[m] - self.pop_set[front[i - 1]].objectives[m]) / (max_obj - min_obj)
        
        return distances
    

    def binary_tournament_selection(self,crowding_distances,random):
        """Selects a solution using binary tournament selection."""
        #Edge case catch (for very small rule populations, which leads to very small model populations)
        if len(self.match_set) == 1:
            return self.match_set[0]
        a, b = random.sample(self.match_set, 2)
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
        

    def update_niche_metrics_C(self, instance_index, fronts, crowding_distances):
        """ """
        #print('updated niche metrics --------')
        #print('fronts: '+str(fronts))
        #print('crowd: '+str(crowding_distances))
        rank = 1
        for front in fronts: # For each front
            #print('front '+str(front))
            for rule_index in front: # For each rule within the front
                #print('rule_index '+str(rule_index))
                self.pop_set[rule_index].update_niche_metrics(instance_index, rank,crowding_distances[rule_index])
            rank += 1


    def initialize_niche_rules(self):
        """ """
        # Add all rules in current {C}
        #print('initialize niche rules')
        #print(self.correct_set)
        self.rules_for_niche_update = copy.deepcopy(self.correct_set)


    def update_impacted_niches(self,heros):
        """ For each newly introduced rule this method re-evaluates dominance and crowding for each rule in every other niche (i.e. instance) {M}/{C} where this new rule would be included.  These other niches are defined by the instances that this rule would be both {M}/{C}. """
        for rule_index in self.new_rule_set: #Each NEW rule introduced this generation
            #print('current new rule index - NICHE - being updated: '+str(rule_index))
            correctly_predicted_instances = list(self.pop_set[rule_index].correct_instance_indexes.keys())
            for instance_index in correctly_predicted_instances: # Each training instance that this rule matched and made the correct prediction
                #print('current instance index -NICHE - being updated: '+str(instance_index))
                temp_match_set = []
                temp_correct_set = []
                instance_state = heros.env.train_data[0][instance_index]
                outcome_state = heros.env.train_data[1][instance_index]
                # Form match set
                for i in range(len(self.pop_set)):
                    rule = self.pop_set[i]
                    if rule.match(instance_state,heros):
                        temp_match_set.append(i) #adds index to rule in pop_set
                        # Form correct set
                        if rule.rule_is_correct(outcome_state,heros):
                            temp_correct_set.append(i)
                            # Add any rules that were in a correct set to the unique rule list for a final update later
                            self.rules_for_niche_update.append(i)
                    else:
                        pass
                #Apply NSGAII-like fast non dominated sorting of models into ranked fronts of models
                fronts = self.fast_non_dominated_sort_C(temp_correct_set)
                #Calculate crowding distances
                crowding_distances = {sol: d for front in fronts for sol, d in self.calculate_crowding_distance(front).items()}
                # Update {C} rule rank and crowding distances
                self.update_niche_metrics_C(instance_index, fronts, crowding_distances)


    def global_niche_update(self):
        """ """
        #print('rules for niche update')
        #print(self.rules_for_niche_update)
        for rule_index in self.rules_for_niche_update:
            self.pop_set[rule_index].global_niche_metric_update()


    def deletion(self,heros,np):
        """ """
        #print(self.new_rule_set)
        heros.timer.deletion_time_start() 
        if self.micro_pop_count > heros.pop_size:
            # Calculate number of rules to delete
            delete_count = self.micro_pop_count - heros.pop_size
            #print(delete_count)
            #print(self.micro_pop_count)

            deletion_list = []
            headers = ['index','rank','numerosity','crowding','deletion weight']
            rule_index = 0
            rule_list = []
            for rule in self.pop_set:
                rule_list = [rule_index, rule.top_niche_rank, rule.numerosity, rule.max_niche_crowding_distance, 0]
                rule_index += 1
                deletion_list.append(rule_list)
            df = pd.DataFrame(deletion_list,columns = headers)
            #Determine deletion ranking
            if df['rank'].max() == 1: #Only non-dominated rules exist
                # Delete rules with larger numerosity first, then use crowding as a tie breaker
                sorted_df = df.sort_values(by=['numerosity','crowding'], ascending=[True, False])
            elif df[df['rank'] > 1]['numerosity'].max() > 1: #if there is a dominated rule with numerosity > 1
                # Delete from rule with worst rank then highest numerosity then lowest crowding
                df = df[(df['rank'] > 1) & (df['numerosity'] > 1)] #Focus on 
                sorted_df = df.sort_values(by=['rank','numerosity','crowding'], ascending=[True, True, False])
            # Only now do we potentially delete from rank 1 rules numerosity
            else:
                rule_index = 0
                for rule in self.pop_set:
                    deletion_weight = 0
                    if rule.top_niche_rank == 1 and rule.numerosity == 1: 
                        deletion_weight = 0 #always protected in situations where {P} includes any dominated rules
                    else:
                        deletion_weight = (rule.numerosity*heros.diversity) + rule.top_niche_rank
                        rule.deletion_prob = deletion_weight
                    df.at[rule_index, 'deletion weight'] = deletion_weight
                    rule_index += 1
                sorted_df = df.sort_values(by=['deletion weight','crowding'], ascending=[True, False])
            # Pick lowest deletion score -ranked rules for deletion
            selected_rule_indexes = sorted_df['index'].tail(delete_count).tolist()
            sorted_list = sorted(selected_rule_indexes, reverse=True)
            # Delete selected rules (either numerosity reduction or rule removal)
            for rule_index in sorted_list:
                self.pop_set[rule_index].update_numerosity(-1)
                self.micro_pop_count -= 1
                if self.pop_set[rule_index].numerosity < 1: # When all micro-classifiers for a given classifier have been depleted.
                    self.pop_set.pop(rule_index)
        heros.timer.deletion_time_stop()


    def deletion_first_experimental(self,heros,np):
        """ """
        heros.timer.deletion_time_start()
        if self.micro_pop_count > heros.pop_size:
            # Calculate number of rules to delete
            delete_count = self.micro_pop_count - heros.pop_size
            rule_indexes = []
            deletion_votes = []
            deletion_probs = []
            rule_index = 0
            for rule in self.pop_set:
                rule_indexes.append(rule_index)
                deletion_votes.append(rule.get_deletion_vote(np))
                rule_index += 1
            vote_sum = sum(deletion_votes)
            i = 0
            for rule in self.pop_set:
                rule.deletion_prob = deletion_votes[i] / vote_sum
                deletion_probs.append(rule.deletion_prob)
                i +=1
            # Select rules to be deleted
            selected_rule_indexes = list(np.random.choice(rule_indexes, size=delete_count, replace=False, p=deletion_probs))
            sorted_list = sorted(selected_rule_indexes, reverse=True)
            # Delete selected rules (either numerosity reduction or rule removal)
            for rule_index in sorted_list:
                self.pop_set[rule_index].update_numerosity(-1)
                self.micro_pop_count -= 1
                if self.pop_set[rule_index].numerosity < 1: # When all micro-classifiers for a given classifier have been depleted.
                    self.pop_set.pop(rule_index)
        heros.timer.deletion_time_stop()


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
            new_rule = RULE(heros,np)
            new_rule.initialize_by_covering(set_numerosity_sum+1,instance_state,outcome_state,heros,random,np)
            #self.debug_confirm_offspring_match(new_rule, instance,heros,'covering',None)
            if len(new_rule.condition_indexes) > 0: #prevents completely general rules from being added to the population
                #Check for duplicate rule in {P} - important since covering runs if {C} is empty, which can generate an existing rule in the match set
                if self.archive_discovered_rules:
                    rule_summary = self.rule_exists(new_rule)
                    if rule_summary == None:
                        self.evaluate_covered_rule(new_rule,outcome_state,heros,random)
                    else:
                        new_rule.reestablish_rule(rule_summary,heros)
                else:
                    self.evaluate_covered_rule(new_rule,outcome_state,heros,random)
                if self.no_identical_rule_exists(new_rule,heros,'match_set'):
                    #new_rule.show_rule_short('covered rule')
                    self.add_rule_to_pop(new_rule)
                    #self.match_set.append(len(self.pop_set)-1)
        heros.timer.covering_time_stop() #covering time tracking


    def make_eval_match_set(self,instance_state,heros):
        """ Makes a match set {M} given an instance state. Used by predict function."""
        for i in range(len(self.pop_set)):
            rule = self.pop_set[i]
            if rule.match(instance_state,heros):
                self.match_set.append(i)


    def global_fitness_update(self,heros):
        """ Relevant for pareto-front rule fitness. Updates the fitness of all rules in the population if the pareto front gets updated. """
        # Update rule front
        for rule in self.pop_set:
            rule.update_rule_pareto_front(heros)
        # Update rule fitness
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

    def genetic_algorithm(self, instance, crowding_distances, heros, random,np):
        instance_state = instance[0] #instance feature values
        outcome_state = instance[1] #instance outcome value
        # PARENT SELECTION *****************************************
        heros.timer.selection_time_start() #parent selection time tracking
        parent1 = self.binary_tournament_selection(crowding_distances,random)
        parent2 = self.binary_tournament_selection(crowding_distances,random)

        #self.pop_set[parent1].show_rule_short('parent rule')
        #self.pop_set[parent2].show_rule_short('parent rule')
        parent_list = [self.pop_set[parent1],self.pop_set[parent2]]
        #parent_list = self.tournament_selection(heros,random)
        heros.timer.selection_time_stop() #parent selection time tracking
        # INITIALIZE OFFSPRING *************************************
        heros.timer.mating_time_start() #mating time tracking
        offspring_list = []
        for parent_rule in parent_list:
            new_rule = RULE(heros,np)
            new_rule.initialize_by_parent(parent_rule,heros)
            offspring_list.append(new_rule)
        # CROSSOVER OPERATOR **************************************
        if len(offspring_list) > 1: #crossover only applied between two parent rules
            if random.random() < heros.cross_prob:
                #print('crossover run')
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
                #print('duplicate offspring found - and removed')
                offspring_list.pop()
                if len(offspring_list) > 1:
                    print("ERROR: More than 2 expected offspring in GA")
        #print('OFFSPRING')
        #for offspring in offspring_list: #debugging
        #    offspring.show_rule_short('offspring rule')
        # CHECK FOR DUPLICATE RULES IN {P} and EVALUATE Non-Duplicate Ruels
        front_updated = False
        final_offspring_list = []
        for offspring in offspring_list:
            if self.archive_discovered_rules:
                rule_summary = self.rule_exists(offspring)
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
            else: # DEBUG
                pass
                #print('identical rule found in population')
        """
        # Update all rule fitness values if one or both offspring rules updated the pareto front
        heros.timer.rule_eval_time_start() #rule evaluation time tracking
        if heros.fitness_function == 'pareto' and front_updated: #new 3/29/25
            self.global_fitness_update(heros) #Re-evaluates all rule fitness values in rule population
            #In update fitness of two offspring rules that are not yet in the population
            for offspring in final_offspring_list:
                offspring.update_rule_fitness(heros)
        heros.timer.rule_eval_time_stop() #rule evaluation time tracking
        """
        # INSERT RULE(S) IN POPULATON (OPTIONAL GA SUBSUMPTION) ***************************
        self.process_offspring(parent_list,final_offspring_list,outcome_state,heros)


    def process_offspring(self,parent_list,offspring_list,outcome_state,heros):
        """ Activates GA subsumption (if used), and then inserts offpring rules into population as needed. """
        if heros.subsumption == 'ga' or heros.subsumption == 'both': #apply subsumption and insert rule(s) as needed
            heros.timer.subsumption_time_start()
            for offspring in offspring_list:
                self.ga_subsumption(offspring,parent_list,outcome_state,heros)
            heros.timer.subsumption_time_stop()
        else: #insert rule(s) as needed following rule equality check
            for offspring in offspring_list:
                self.add_rule_to_pop(offspring)
                if offspring.rule_is_correct(outcome_state,heros):
                    self.correct_set.append(len(self.pop_set)-1)


    def ga_subsumption(self,offspring,parent_list,outcome_state,heros):
        """ Applies GA subsumption. """
        offspring_subsumed = False
        for parent in parent_list:
            if not offspring_subsumed:
                if parent.subsumes(offspring,heros):
                    offspring_subsumed = True
                    self.micro_pop_count += 1
                    parent.update_numerosity(1)
        if not offspring_subsumed:
            self.add_rule_to_pop(offspring)
            if offspring.rule_is_correct(outcome_state,heros):
                self.correct_set.append(len(self.pop_set)-1)

    """
    def add_covered_rule_to_pop(self,new_rule,outcome_state,heros,random): #old now
        #Adds a new rule to the population via covering: either as a new rule entry in the population or increasing the numerosity of a rule that already exists. 
        heros.timer.covering_time_stop() #covering time tracking
        heros.timer.rule_eval_time_start() #rule evaluation time tracking
        if heros.outcome_type == 'class':
            front_updated = new_rule.complete_rule_evaluation_class(heros,random,outcome_state) #only called if brand new rule being added to population
        elif heros.outcome_type == 'quant':
            front_updated = new_rule.complete_rule_evaluation_quant(heros) #only called if brand new rule being added to population
        else:
            pass
        if heros.fitness_function == 'pareto' and front_updated: #new 3/29/25
            self.global_fitness_update(heros)
        heros.timer.rule_eval_time_stop() #rule evaluation time tracking
        heros.timer.covering_time_start() #covering time tracking
    """

    def evaluate_covered_rule(self,new_rule,outcome_state,heros,random):
        heros.timer.covering_time_stop() #covering time tracking
        heros.timer.rule_eval_time_start() #rule evaluation time tracking
        if heros.outcome_type == 'class':
            front_updated = new_rule.complete_rule_evaluation_class(heros,random,outcome_state) #only called if brand new rule being added to population
        elif heros.outcome_type == 'quant':
            front_updated = new_rule.complete_rule_evaluation_quant(heros) #only called if brand new rule being added to population
        else:
            pass
        """
        if heros.fitness_function == 'pareto' and front_updated: 
            self.global_fitness_update(heros)
        """
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


    def add_rule_to_pop(self,new_rule):
        """ Add new and novel rule to population, updating key relevant parameters. """
        
        new_rule.assign_ID(self.ID_counter)
        self.pop_set.append(new_rule)
        self.match_set.append(len(self.pop_set)-1)
        self.new_rule_set.append(len(self.pop_set)-1)
        self.ID_counter += 1 #every time a new rule gets added to the pop (that isn't in the current pop) it is assigned a new unique ID
        self.micro_pop_count += 1
        if self.archive_discovered_rules:
            self.add_new_explored_rules(new_rule)


    def make_correct_set(self,outcome_state,heros):
        """ Makes a correct set {C}"""
        for i in range(len(self.match_set)):
            rule_index = self.match_set[i]
            if self.pop_set[rule_index].rule_is_correct(outcome_state,heros):
                self.correct_set.append(rule_index)


    def update_rule_parameters(self,heros):
        """ Updates all relevant rule parameters for rules in the current match set. """
        match_set_numerosity_sum = 0
        for rule_index in self.match_set:
            match_set_numerosity_sum += self.pop_set[rule_index].numerosity
        for rule_index in self.match_set:
            self.pop_set[rule_index].update_ave_match_set_size(match_set_numerosity_sum,heros)


    def deletion_original(self,heros,random):
        """ Applies probabalistic deletion to the rule population to maintain maximum population size."""
        heros.timer.deletion_time_start()
        while self.micro_pop_count > heros.pop_size:
            self.delete_rule(random)
        heros.timer.deletion_time_stop()
    

    def delete_rule(self,random):
        """ Probabilistically identifies a rule to delete with roulette wheel selection, and deletes it at the micro-rule level."""
        vote_sum = 0.0
        vote_list = []
        for rule in self.pop_set:
            vote = rule.get_deletion_vote()
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
                rule.update_numerosity(-1)
                self.micro_pop_count -= 1
                if rule.numerosity < 1:  # When all micro-classifiers for a given classifier have been depleted.
                    self.remove_macro_rule(i)
                return
    

    def remove_macro_rule(self,rule_index):
        """ Removes the given (macro-) rule from the population. """
        self.pop_set.pop(rule_index)
    

    def clear_sets(self):
        """ Clears out references in the match and correct sets for the next learning iteration. """
        self.match_set = []
        self.correct_set = []
        self.new_rule_set = []


    def order_all_rule_conditions(self):
        """ Order the rule conditions by increasing feature index; keeping the ordering consistent between condition_indexes and condition_values."""
        for rule in self.pop_set:
            rule.order_rule_conditions()


    def load_rule_population(self, pop_df, heros, random,np):
        """ Load a HEROS rule population data frame, then instantiates and evaluates all rules.
            Each specified rule must have a condition and action at minimum. """
        rule_count = pop_df.shape[0] #rows in dataframe
        for rule_row in range(rule_count): #for each rule in the dataframe
            # Initialize the rule
            loaded_rule = RULE(heros,np)
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
            if loaded_rule.numerosity is None:
                loaded_rule.numerosity = 1
            else:
                loaded_rule.numerosity = int(pop_df.loc[rule_row,'Numerosity'])
            # Set the rule average match set size
            if loaded_rule.ave_match_set_size is None:
                loaded_rule.ave_match_set_size = 1
            else:
                loaded_rule.ave_match_set_size = float(pop_df.loc[rule_row,'Average Match Set Size'])
            # Set the rule birth iteration
            loaded_rule.birth_iteration = 0
            # Evaluate loaded rule
            if heros.outcome_type == 'class':
                front_updated = loaded_rule.complete_rule_evaluation_class(heros,random,None) #only called if brand new rule being added to population
            elif heros.outcome_type == 'quant':
                front_updated = loaded_rule.complete_rule_evaluation_quant(heros) #only called if brand new rule being added to population
        # Update all rule fitness values (if pareto front rule fitness used)
        """
        if heros.fitness_function == 'pareto': #new 3/29/25
            self.global_fitness_update(heros)
        """
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
                        'Top Niche Rank',
                        'Top Niche Crowding',
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
                         rule.top_niche_rank,
                         rule.max_niche_crowding_distance,
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