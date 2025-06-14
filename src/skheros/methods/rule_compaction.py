import copy

class COMPACT:
    def __init__(self,heros):

        self.original_pop_size = len(heros.rule_population.pop_set)
        if heros.verbose:
            print("--------------------------------------------------------------------")
            print("Original Population Size: "+str(self.original_pop_size))

    def basic_rule_cleaning(self,heros):
        """ Removes obviously bad rules from the population based on having a useful accuracy <= 0. """
        index = 0
        for i in range(len(heros.rule_population.pop_set)):
            rule = heros.rule_population.pop_set[index]
            if rule.useful_accuracy <= 0:
                #Remove bad rule from population
                heros.rule_population.pop_set.pop(index)
            else:
                index += 1
        if heros.verbose:
            print("Post-Cleaning Population Size: "+str(len(heros.rule_population.pop_set)))

    def custom_sort_key(self, obj):
        return (len(obj.condition_indexes),-obj.useful_accuracy)

    def subsumption_compation(self,heros):
        """ A simple subsumption-based rule compaction strategy. Sorts rule population by increasing specificity and decreasing useful accuracy.
            Checks if any rule in the population can subsume any other """
        # Sort rules by increasing specificity then by decreasing useful_accuracy
        heros.rule_population.pop_set = sorted(heros.rule_population.pop_set, key=self.custom_sort_key)
        subsumption_history = {}
        keep_going = True 
        master_index = 0 #keeps track of the index of the first rule in the current specificity level being examined
        while keep_going: #applied once per rule specificity level (examines all rules of this specificity as potential subsumers)
            final_level_specificity = len(heros.rule_population.pop_set[len(heros.rule_population.pop_set)-1].condition_indexes)
            try:
                current_level_specificity = len(heros.rule_population.pop_set[master_index].condition_indexes)
                if current_level_specificity >= final_level_specificity:
                    break
            except: #current_level_specificity == final_level_specificity:
                break
            current_level_index_start = master_index
            # Find start index of next rule specificity level
            current_level_index_end = master_index #initialized
            next_level_specificity = len(heros.rule_population.pop_set[current_level_index_end+1].condition_indexes) #initialized
            same_level = True
            while same_level:
                next_level_specificity = len(heros.rule_population.pop_set[current_level_index_end+1].condition_indexes)
                if next_level_specificity != current_level_specificity: #next index is the start of the next level
                    same_level = False
                else: 
                    current_level_index_end += 1
            # Check if current level rules can subsume any later level rules
            for current_level_index in range(current_level_index_start,current_level_index_end+1): #for each rule of current specificity level
                other_level_index = copy.deepcopy(current_level_index_end+1)
                # Check if any other rules remain beyond the current level
                other_rules_remain = True
                if other_level_index > len(heros.rule_population.pop_set) - 1: 
                    other_rules_remain = False #ends while loop below
                    keep_going = False #no other opportunities for subsumption - ends main while loop above
                while other_rules_remain:
                    if heros.rule_population.pop_set[current_level_index].useful_accuracy >= heros.rule_population.pop_set[other_level_index].useful_accuracy: #candidate subsumer has >= useful_accuracy of candidate subsumed
                        if self.subsumes(heros.rule_population.pop_set[current_level_index],heros.rule_population.pop_set[other_level_index],heros): #candidate subsumer meets all requirements to subsume
                            #Update subsumption history
                            current_other_level = len(heros.rule_population.pop_set[other_level_index].condition_indexes)
                            if current_other_level in subsumption_history:
                                subsumption_history[current_other_level] += 1
                            else:
                                subsumption_history[current_other_level] = 1
                            #Update subsumer numerosity
                            heros.rule_population.pop_set[current_level_index].numerosity += heros.rule_population.pop_set[other_level_index].numerosity
                            #Remove subsumed rule from population
                            heros.rule_population.pop_set.pop(other_level_index)
                        else: #other rule cannot be subsumed
                            other_level_index +=1
                    else: #other rule cannot be subsumed
                        other_level_index +=1
                    #Check if we reached the last rule in the population
                    if other_level_index > len(heros.rule_population.pop_set) - 1: #we reached the last rule in the population for this cycle chek
                        other_rules_remain = False
            master_index = current_level_index_end + 1 #set to the start of the next specificity level
        if heros.verbose:
            print("Post-Subsumption Compaction Population Size: "+str(len(heros.rule_population.pop_set)))
            subsumption_history = dict(sorted(subsumption_history.items()))
            for key in subsumption_history:
                print(str(subsumption_history[key])+ ' rules subsumed with a specificity of '+str(key))

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
        (2) if a given feature is quanatiative the current_level_rule low boundary is <= than for next_level_rule
            and the current_level_rule high boundary is >= than for next_level_rule, i.e. the range is equal or larger (more general) at both ends."""
        for feat in current_level_rule.condition_indexes: #for each feature specified in current_level_rule 
            if feat not in next_level_rule.condition_indexes: #does self cover all the instance space of next_level_rule
                return False #not more general
            if not heros.env.feat_types[feat]: #quantiative feature
                self_rule_position = current_level_rule.condition_indexes.index(feat)
                other_rule_position = next_level_rule.condition_indexes.index(feat)
                # Current assumption - subsumer has a wider quanatiative feature range inclusive of the other rule's range
                if current_level_rule.condition_values[self_rule_position][0] > next_level_rule.condition_values[other_rule_position][0]: #low end
                    return False #not more general
                if current_level_rule.condition_values[self_rule_position][1] < next_level_rule.condition_values[other_rule_position][1]: #high end
                    return False #not more general
        return True
