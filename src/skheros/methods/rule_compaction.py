import copy

class COMPACT:
    def __init__(self,heros):
        self.original_pop_size = len(heros.rule_population.pop_set)
        if heros.verbose:
            print("--------------------------------------------------------------------")
            print("Original Population Size: "+str(self.original_pop_size))
        self.original_pop_copy = copy.deepcopy(heros.rule_population.pop_set)


    def clear_pop_copy(self):
        self.original_pop_copy = None


    def basic_rule_cleaning(self,heros):
        """ Removes obviously bad rules from the population based on having a useful accuracy <= 0. """
        index = 0
        sufficient_rule_pop_remain = True
        for i in range(len(heros.rule_population.pop_set)):
            rule = heros.rule_population.pop_set[index]
            if rule.useful_accuracy <= 0:
                #Remove bad rule from population
                heros.rule_population.pop_set.pop(index)
                heros.rule_population.micro_pop_count -= rule.numerosity
            else:
                index += 1
        if heros.verbose:
            print("Post-Cleaning Population Size: "+str(len(heros.rule_population.pop_set)))
        if len(heros.rule_population.pop_set) < 2: #Phase 1 failed to find quality rules, so there is no need to do phase 2.
            sufficient_rule_pop_remain = False
            heros.rule_population.pop_set = self.original_pop_copy # restore original rule popululation (which will be used by default as the final model)
            print("Phase 2 skipped due to insufficient number of quality rules discovered following rule cleaning (i.e. < 2)")
        return sufficient_rule_pop_remain


    def custom_sort_key(self, obj):
        return (len(obj.condition_indexes),-obj.useful_accuracy)


    def subsumption_compation(self,heros):
        """ A simple subsumption-based rule compaction strategy. Sorts rule population by increasing specificity and decreasing useful accuracy.
            Checks if any rule in the population can subsume any other """
        # Sort rules by increasing specificity then by decreasing useful_accuracy
        heros.rule_population.pop_set = sorted(heros.rule_population.pop_set, key=self.custom_sort_key)
        pop_set = heros.rule_population.pop_set
        subsumption_history = {}
        # Every rule is given the opportunity to subsume every other rule. This includes rules of the same specificity
        # (same feature set) whose quantitative ranges fall within its own, so the comparison cannot be limited to later specificity levels.
        current_index = 0
        while current_index < len(pop_set): #for each remaining rule as a candidate subsumer
            other_index = 0
            while other_index < len(pop_set): #for each remaining rule as a candidate subsumed
                if other_index != current_index and pop_set[current_index].useful_accuracy >= pop_set[other_index].useful_accuracy: #candidate subsumer has >= useful_accuracy of candidate subsumed
                    if self.subsumes(pop_set[current_index],pop_set[other_index],heros): #candidate subsumer meets all requirements to subsume
                        #Update subsumption history
                        other_specificity = len(pop_set[other_index].condition_indexes)
                        if other_specificity in subsumption_history:
                            subsumption_history[other_specificity] += 1
                        else:
                            subsumption_history[other_specificity] = 1
                        #Update subsumer numerosity
                        pop_set[current_index].numerosity += pop_set[other_index].numerosity
                        #Remove subsumed rule from population
                        pop_set.pop(other_index)
                        if other_index < current_index: #removal shifts the subsumer down one position
                            current_index -= 1
                        continue #the next rule now occupies other_index
                other_index += 1
            current_index += 1
        if heros.verbose:
            subsumption_history = dict(sorted(subsumption_history.items()))
            for key in subsumption_history:
                print(str(subsumption_history[key])+ ' rules subsumed with a specificity of '+str(key))
            print("Post-Subsumption Compaction Population Size: "+str(len(heros.rule_population.pop_set)))
        sufficient_rule_pop_remain = True        
        if len(heros.rule_population.pop_set) < 2: #Phase 1 failed to find quality rules, so there is no need to do phase 2.
            sufficient_rule_pop_remain = False
            heros.rule_population.pop_set = self.original_pop_copy # restore original rule popululation (which will be used by default as the final model)
            print("Phase 2 skipped due to insufficient number of quality rules discovered following rule compaction (i.e. < 2)")
        return sufficient_rule_pop_remain


    def subsumes(self,current_level_rule,next_level_rule,heros):
        """ Determines if 'self' rule meets conditions for subsuming the 'other_rule'. A rule is a subsumer if:
        (1) It has the same action as the other rule.
        (2) It is more general than the other rule, covering all of the instance space of the other rule.
        """
        if heros.outcome_type == 'class': #class outcome
            if not current_level_rule.action == next_level_rule.action:
                return False
        elif heros.outcome_type == 'quant': #quantiative outcome
            if current_level_rule.action[0] > next_level_rule.action[0] or current_level_rule.action[1] < next_level_rule.action[1]:
                return False
        if not self.is_more_general_compaction(current_level_rule,next_level_rule,heros):
            return False
        else:
            return True
        

    def is_more_general_compaction(self,current_level_rule,next_level_rule,heros):
        """ Checks the conditions determining if current_level_rule is more general than next_level_rule as defined by being a candidate subsumer.
        (1) next_level_rule specifies at least the same subset of features as current_level_rule
        (2) if a given feature is categorical the values are the same
        (3) if a given feature is quantitative the current_level_rule low boundary is <= than for next_level_rule
            and the current_level_rule high boundary is >= than for next_level_rule, i.e. the range is equal or larger (more general) at both ends."""
        for feat in current_level_rule.condition_indexes: #for each feature specified in current_level_rule
            if feat not in next_level_rule.condition_indexes: #does self cover all the instance space of next_level_rule
                return False #not more general
            self_rule_position = current_level_rule.condition_indexes.index(feat)
            other_rule_position = next_level_rule.condition_indexes.index(feat)
            if heros.env.feat_types[feat]: #categorical feature
                if current_level_rule.condition_values[self_rule_position] != next_level_rule.condition_values[other_rule_position]:
                    return False #not more general
            else: #quantiative feature
                # Current assumption - subsumer has a wider quanatiative feature range inclusive of the other rule's range
                if current_level_rule.condition_values[self_rule_position][0] > next_level_rule.condition_values[other_rule_position][0]: #low end
                    return False #not more general
                if current_level_rule.condition_values[self_rule_position][1] < next_level_rule.condition_values[other_rule_position][1]: #high end
                    return False #not more general
        return True