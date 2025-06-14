import time

class TIME_TRACK:
    def __init__(self):
        """ Initializes all Timer values for the algorithm """
        # Global time objects
        self.global_start_time = time.time() #stores timer start time
        self.time_global = 0.0 #stores global run time
        self.time_append = 0.0 #used when loading an pre-trained rule population to update the global time
        self.time_init = 0.0 #stores global algorithm initialization time
        self.time_covering = 0.0 #stores global covering time
        self.time_rule_equality = 0.0 #stores global time adding rules to population (and checking for identical rules)
        self.time_matching = 0.0 #stores global matching time
        self.time_rule_eval = 0.0 #stores global time to evaluate rule parameters
        self.time_feature_track = 0.0 #stores global time to update feature tracking scores
        self.time_subsumption = 0.0 #stores global subsumption time
        self.time_selection = 0.0 #stores global parent selection time
        self.time_mating = 0.0 #stores global parent rule mating time (i.e. genetic operators applied to create offspring rules)
        self.time_deletion = 0.0 #stores global rule deletion time
        self.time_compaction = 0.0 # stores global time to conduct rule population compaction if applied
        self.time_prediction = 0.0 #stores global time used to apply prediction to estimate training accuracy during learning

        #Temporary start time objects
        self.time_start_init = 0.0
        self.time_start_covering = 0.0
        self.time_start_rule_equality = 0.0
        self.time_start_matching = 0.0
        self.time_start_rule_eval = 0.0
        self.time_start_feature_track = 0.0
        self.time_start_subsumption = 0.0
        self.time_start_selection = 0.0
        self.time_start_mating = 0.0
        self.time_start_deletion = 0.0
        self.time_start_compaction = 0.0
        self.time_start_prediction = 0.0

    # Algorithm Initialization ***********************************************************
    def init_time_start(self):
        self.time_start_init = time.time()

    def init_time_stop(self):
        diff = time.time() - self.time_start_init
        self.time_init  += diff

    # Covering ************************************************************
    def covering_time_start(self):
        self.time_start_covering  = time.time()

    def covering_time_stop(self):
        diff = time.time() - self.time_start_covering 
        self.time_covering += diff

    # Rule Equality ************************************************************
    def rule_equality_time_start(self):
        self.time_start_rule_equality  = time.time()

    def rule_equality_time_stop(self):
        diff = time.time() - self.time_start_rule_equality 
        self.time_rule_equality += diff

    # Matching ************************************************************
    def matching_time_start(self):
        self.time_start_matching  = time.time()

    def matching_time_stop(self):
        diff = time.time() - self.time_start_matching 
        self.time_matching += diff

    # Rule Evaluation ************************************************************
    def rule_eval_time_start(self):
        self.time_start_rule_eval  = time.time()

    def rule_eval_time_stop(self):
        diff = time.time() - self.time_start_rule_eval 
        self.time_rule_eval += diff

    # Feature Tracking ************************************************************
    def feature_track_time_start(self):
        self.time_start_feature_track  = time.time()

    def feature_track_time_stop(self):
        diff = time.time() - self.time_start_feature_track 
        self.time_feature_track += diff

    # Subsumption ************************************************************
    def subsumption_time_start(self):
        self.time_start_subsumption  = time.time()

    def subsumption_time_stop(self):
        diff = time.time() - self.time_start_subsumption 
        self.time_subsumption += diff

    # Selection ************************************************************
    def selection_time_start(self):
        self.time_start_selection  = time.time()

    def selection_time_stop(self):
        diff = time.time() - self.time_start_selection 
        self.time_selection += diff

    # Mating ************************************************************
    def mating_time_start(self):
        self.time_start_mating  = time.time()

    def mating_time_stop(self):
        diff = time.time() - self.time_start_mating 
        self.time_mating += diff

    # Deletion ************************************************************
    def deletion_time_start(self):
        self.time_start_deletion  = time.time()

    def deletion_time_stop(self):
        diff = time.time() - self.time_start_deletion 
        self.time_deletion += diff

    # Rule Population Compaction ************************************************************
    def compaction_time_start(self):
        self.time_start_compaction  = time.time()

    def compaction_time_stop(self):
        diff = time.time() - self.time_start_compaction 
        self.time_compaction += diff

    # Prediction ************************************************************
    def prediction_time_start(self):
        self.time_start_prediction  = time.time()

    def prediction_time_stop(self):
        diff = time.time() - self.time_start_prediction 
        self.time_prediction += diff

    # Update global time ***********************************************************
    def update_global_time(self):
        self.time_global = (time.time() - self.global_start_time) + self.time_append

 
