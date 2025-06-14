import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

class MODEL_PARETO:
    def __init__(self):  
        #Definte the parts of the Pareto Front
        self.model_front = []  #list of objective pair sets (useful_accuracy,useful_coverage) for each non-dominated model (ordered by increasing accuracy)
        self.model_front_scaled = []
        self.metric_limits = [None]*2 #assumes two metrics being optimized
        self.front_diagonal_lengths = [] # length of the diagonal lines from the orgin to each point on the pareto front (used to calculate model fitness)

    def update_front(self,candidate_metric_1,candidate_metric_2,objectives):
        """  Handles process of checking and updating the model pareto front. Only set up for two objectives. """
        original_front = copy.deepcopy(self.model_front)
        candidate_model = (candidate_metric_1,candidate_metric_2)
        if not candidate_model in self.model_front: # Check that candidate model objectives are not equal to any existing objective pairs on the front
            non_dominated_models = []
            candidate_dominated = False
            for front_model in self.model_front:
                if self.dominates(front_model,candidate_model,objectives): #does front model dominate candidate model (if so, this check is ended)
                    candidate_dominated = True #prevents candidate from being added 
                    break # original front is preserved
                elif not self.dominates(candidate_model,front_model,objectives): #does the candidate model dominate the front model (if so it will get added to the front)
                    non_dominated_models.append(front_model)
            if candidate_dominated: #at least one front model dominates the candidate model
                non_dominated_models = self.model_front
            else: #no front models dominate the candidate model
                non_dominated_models.append(candidate_model)
            # Update the model front to include only non dominated models
            self.model_front = sorted(non_dominated_models, key=lambda x: x[0])
            # Update the maximum values found in models on the pareto front and update model_front_scaled as needed
            self.metric_limits[0] = max(self.model_front, key=lambda x: x[0])[0]
            self.metric_limits[1] = max(self.model_front, key=lambda x: x[1])[1]
            if self.metric_limits[1] != 0.0:
                self.model_front_scaled = [(x[0],x[1] /float(self.metric_limits[1])) for x in self.model_front]
            else:
                self.model_front_scaled = self.model_front
        if original_front == self.model_front:
            return False
        else:
            print("Front Update") #debug
            print(original_front) #debug
            print(self.model_front) #debug
            return True

    def dominates(self,p,q,objectives):
        """Check if p dominates q. A model dominates another if it has a more optimal value for at least one objective."""
        better_in_all_objectives = True
        better_in_at_least_one_objective = False
        for val1, val2, obj in zip(p, q, objectives):
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

    def get_pareto_fitness(self,candidate_metric_1,candidate_metric_2, landscape,heros):
        """ Calculate model fitness releative to the model pareto front. Only set up for two objectives. """
        # First handle simple special cases
        if len(self.model_front_scaled) == 1 and self.model_front_scaled[0][0] == 0.0 and self.model_front_scaled[0][1] == 0.0:
            return None #Handles unlikely special case where the only point on the front has zero accuracy and zero rules.
        if landscape: # Special cases when calculating fitness landscape for visualization (points that will never be models)
            if candidate_metric_1 > 1.0 and candidate_metric_2 <= self.model_front_scaled[-1][1]: 
                return 1.0 #Accuray is greater than 1 on front plot and rule count is <= the maximum found (top left of front)
            if candidate_metric_2 / float(self.metric_limits[1]) < self.model_front_scaled[0][1] and candidate_metric_1 >= self.model_front_scaled[0][0]:
                return 1.0 #Rule Count is less than the minimum found and accuracy is >= the mimimum found (left top of front)
            if candidate_metric_1 > self.model_front_scaled[0][0] and candidate_metric_2 / float(self.metric_limits[1]) < self.model_front_scaled[-1][1] and self.point_beyond_front(candidate_metric_1,candidate_metric_2 / float(self.metric_limits[1])):
                return 1.0 # Point lies just beyond the front but not into the previously captured extremes.
        if candidate_metric_1 == 0.0 and candidate_metric_2 == self.metric_limits[1]:
            return 0.0 #Worst fitness combination (i.e. 0 accuracy and highest rule count)
        elif (candidate_metric_1,candidate_metric_2) in self.model_front: # models on the front return an ideal fitness
            return 1.0
        elif candidate_metric_1 == self.metric_limits[0]: #model has the maximum accuracy (possibly update)
            return 1.0
        #elif candidate_metric_2 <= self.model_front_scaled[0][1]:
        #    return candidate_metric_1 / self.model_front_scaled[0][0]
        
        #if landscape: # Special cases when calculating fitness landscape for visualization
        #    if candidate_metric_1 > 1.0 or candidate_metric_2 / float(self.metric_limits[1]) < self.model_front_scaled[0][1]:
        #        return 1.0 #Special case where accuray is greater than 1 or rule count is less than 1
        #    if candidate_metric_1 > self.model_front_scaled[0][0] and candidate_metric_2 / float(self.metric_limits[1]) < self.model_front_scaled[-1][1] and self.point_beyond_front(candidate_metric_1,candidate_metric_2 / float(self.metric_limits[1])):
        #        return 1.0 
        #if candidate_metric_1 == 0.0 and candidate_metric_2 == self.metric_limits[1]:
        #    return 0.0
        #elif (candidate_metric_1,candidate_metric_2) in self.model_front: # models on the front return an ideal fitness
        #    return 1.0
        #elif candidate_metric_1 == self.metric_limits[0]:
        #    return 1.0
        #elif candidate_metric_2 == self.metric_limits[1]: #model has the maximum value of one objective
        #    return 1.0
        else:
            scaled_candidate_metric_2 = candidate_metric_2 / float(self.metric_limits[1])
            model_objectives = (candidate_metric_1,scaled_candidate_metric_2)
            

            # Find the closest distance between the model and the pareto front (tempfront ordered by increasing accuracy, i.e. metric 1)
            temp_front = []
            #temp_front = [(self.model_front_scaled[0][0],0.0)] #min bin size boundary (horizontal not vertical)
            for front_point in self.model_front_scaled:
                temp_front.append(front_point)
            temp_front.append((self.model_front_scaled[-1][0],scaled_candidate_metric_2)) #max accuracy boundary

            min_distance = float('inf')
            for i in range(len(temp_front) - 1):
                segment_start = temp_front[i]
                segment_end = temp_front[i + 1]
                distance = self.point_to_segment_distance(model_objectives, segment_start, segment_end)
                min_distance = min(min_distance, distance)
            pareto_fitness = 1 - min_distance
            if heros.nu > 1: # Apply pressure to maximize model accuracy
                """
                # Apply fitness penalty to models that fall under the diagonal to the max specificity front point (model slope is less than slope to that front point)
                model_slope = self.slope((0.0,0.0),model_objectives)
                front_point_slope = self.slope((0.0,0.0),self.model_front_scaled[0])
                if model_slope < front_point_slope:
                    pareto_fitness = pareto_fitness / float(heros.nu)
                """
                pareto_fitness = pareto_fitness * pow(candidate_metric_1 , heros.nu)
            return pareto_fitness
        
        
    def point_to_segment_distance(self, point, segment_start, segment_end):
        """ """
        # Vector from segment start to segment end (normalize vector so starting point is 0,0)
        segment_vector = np.array(segment_end) - np.array(segment_start)
        # Vector from segment start to the point (normalize vector so starting point is 0,0)
        point_vector = np.array(point) - np.array(segment_start)
        # Project the point_vector onto the segment_vector to find the closest point on the segment
        segment_length_squared = np.dot(segment_vector, segment_vector) #Calculate segment length squared to be used to do the projection
        if segment_length_squared == 0: #Safty check that the segment start and stop are not the same (if so just return distance to that single point)
            # The segment start and end points are the same
            return self.euclidean_distance(point, segment_start)
        # If segment has length: project the point vector onto the segment vector (identifies the perpendicular intersect)
        projection = np.dot(point_vector, segment_vector) / segment_length_squared
        projection_clamped = max(0, min(1, projection)) #checks if the interstect is within the segment (because the projection was forced between 0, and 1)
        # Find the closest point on the segment (either the perpendicular intersect or distance to segment end)
        closest_point_on_segment = np.array(segment_start) + projection_clamped * segment_vector
        # Return the distance from the point to this closest point on the segment
        return self.euclidean_distance(point, closest_point_on_segment)
    
    def euclidean_distance(self,point1,point2):
        """ Calculates the euclidean distance between two n-dimensional points"""
        if len(point1) != len(point2):
            raise ValueError("Both points must have the same number of dimensions")
        distance = math.sqrt(sum((y - x) ** 2 for y, x in zip(point1, point2)))
        return distance

    def slope(self,point1,point2):
        """ Calculates the slopes between two 2-dimensional points """
        if point1[1] == point2[1]: # line is vertical (both points have 0 coverage)
            slope = np.inf
        else:
            slope = (point2[0] - point1[0]) / (point2[1] - point1[1])
        return slope
    
    def point_beyond_front(self,candidate_metric_1,scaled_candidate_metric_2):
        """ Used for creating pareto front landscape visualization background fitness landscape. """
        # Define line segment from the origin (0,1) to the model's objective (y,x)
        model_start = (0,1)
        model_end = (candidate_metric_1,scaled_candidate_metric_2)
        # Identify segments making up front to check
        intersects = False
        i = 0
        while not intersects and i < len(self.model_front_scaled) - 1:
            segment_start = self.model_front_scaled[i]
            segment_end = self.model_front_scaled[i + 1]
            intersects = self.do_intersect(model_start,model_end,segment_start,segment_end)
            if intersects:
                return True
            i += 1
        return False
    
    def do_intersect(self,p1, q1, p2, q2):
        """ Main function to check whether the line segment p1q1 and p2q2 intersect. """
        # Find the four orientations needed for the general and special cases
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)
        # General case
        if o1 != o2 and o3 != o4:
            return True
        # Special cases
        # p1, q1 and p2 are collinear and p2 lies on segment p1q1
        if o1 == 0 and self.on_segment(p1, p2, q1):
            return True
        # p1, q1 and q2 are collinear and q2 lies on segment p1q1
        if o2 == 0 and self.on_segment(p1, q2, q1):
            return True
        # p2, q2 and p1 are collinear and p1 lies on segment p2q2
        if o3 == 0 and self.on_segment(p2, p1, q2):
            return True
        # p2, q2 and q1 are collinear and q1 lies on segment p2q2
        if o4 == 0 and self.on_segment(p2, q1, q2):
            return True
        return False

    def on_segment(self, p, q, r):
        """
        Given three collinear points p, q, r, the function checks if point q lies on the segment pr.
        """
        if q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]):
            return True
        return False

    def orientation(self, p, q, r):
        """
        To find the orientation of the ordered triplet (p, q, r).
        The function returns:
        0 -> p, q, and r are collinear
        1 -> Clockwise
        2 -> Counterclockwise
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        elif val > 0:
            return 1
        else:
            return 2
        

    def plot_pareto_landscape(self, resolution, model_population, plot_models, heros, show=True, save=False, output_path=None, data_name=None):
        # Generate fitness landscape ******************************
        x = np.linspace(0.00,1.00*self.metric_limits[1],resolution) #model size
        y = np.linspace(0.00,1.00,resolution) #accuracy
        #X,Y = np.meshgrid(x,y)
        Z = [[None for _ in range(resolution)] for _ in range(resolution)]
        for i in range(len(x)):
            for j in range(len(y)):
                Z[j][i] = self.get_pareto_fitness(y[j],x[i],True,heros) #accuracy,coverage  (rows,columns)
        # Prepare to plot model front *****************************
        metric_1_front_list = [None]*len(self.model_front_scaled)
        metric_2_front_list = [None]*len(self.model_front_scaled)
        i = 0
        for model in self.model_front_scaled:
            metric_1_front_list[i] = model[0]
            metric_2_front_list[i] = model[1]
            i+=1
        # Plot Setup *********************************************
        plt.figure(figsize=(10,6)) #(10, 8))
        plt.imshow(Z, extent=[0.00, 1.00, 0.00, 1.00], interpolation='nearest', origin='lower', cmap='magma', aspect='auto') #cmap='viridis' 'magma', alpha=0.6
        # Plot model front ***************************************
        plt.plot(np.array(metric_2_front_list), np.array(metric_1_front_list), 'o-', ms=10, lw=2, color='black')
        # Plot pareto front boundaries to plot edge
        plt.plot([metric_2_front_list[-1],1],[metric_1_front_list[-1],metric_1_front_list[-1]],'--',lw=1, color='black') # horizontal line
        plt.plot([metric_2_front_list[0],metric_2_front_list[0]],[metric_1_front_list[0],0],'--',lw=1, color='black') # vertical line
        # Add colorbar for the gradient
        cbar = plt.colorbar(shrink=0.8)
        cbar.set_label('Fitness Value')
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['0', '0.25', '0.5', '0.75', '1.0'])
        # Add labels and title
        plt.xlabel('Rule-Set Size', fontsize=14)
        plt.ylabel('Coverage Penalized Balanced Accuracy', fontsize=14)
        # Set the axis limits between 0 and 1
        plt.xlim(0.00, 1.00)
        plt.ylim(0.00, 1.00)
        #custom_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        #custom_labels = [int(x * self.metric_limits[1]) for x in [0, 0.2, 0.4, 0.6, 0.8, 1.0]]
        #plt.xticks(custom_ticks, custom_labels)
        def create_x_tick_transform(multiplier):
            def x_tick_transform(x, pos):
                return f"{x * multiplier:.1f}"  # Multiply by value and convert to an integer
            return x_tick_transform
        plt.gca().xaxis.set_major_formatter(FuncFormatter(create_x_tick_transform(self.metric_limits[1])))
        # Prepare to plot model population ***********************
        if plot_models:
            master_metric_1_list = []
            master_metric_2_list = []
            for i in range(len(model_population.pop_set)):
                model = model_population.pop_set[i]
                master_metric_1_list.append(model.accuracy) 
                master_metric_2_list.append(len(model.rule_set)/float(self.metric_limits[1]))
            plt.plot(np.array(master_metric_2_list), np.array(master_metric_1_list), 'o', ms=3, lw=1, color='grey')
        plt.legend(loc='upper left', bbox_to_anchor=(1.25, 1), fontsize='small')
        plt.subplots_adjust(right=0.75)
        if save:
            plt.savefig(output_path+'/'+data_name+'_pareto_fitness_landscape_models.png', bbox_inches="tight")
        if show:
            plt.show()


    def plot_pareto_landscape_extended(self, resolution, model_population, plot_models, heros, show=True, save=False, output_path=None, data_name=None):
        if plot_models:
            master_metric_1_list = []
            master_metric_2_list_noscale = []
            for i in range(len(model_population.pop_set)):
                model = model_population.pop_set[i]
                master_metric_1_list.append(model.accuracy) 
                #master_metric_2_list.append(len(model.rule_set)/float(self.metric_limits[1]))
                master_metric_2_list_noscale.append(len(model.rule_set))
        max_rule_set_size = max(master_metric_2_list_noscale)

        # Generate fitness landscape ******************************
        x = np.linspace(0.00,1.00*max_rule_set_size,resolution) #model size
        y = np.linspace(0.00,1.00,resolution) #accuracy
        #X,Y = np.meshgrid(x,y)
        Z = [[None for _ in range(resolution)] for _ in range(resolution)]
        for i in range(len(x)):
            for j in range(len(y)):
                Z[j][i] = self.get_pareto_fitness(y[j],x[i],True,heros) #accuracy,coverage  (rows,columns)
        # Prepare to plot model front *****************************
        metric_1_front_list = [None]*len(self.model_front)
        metric_2_front_list_temp = [None]*len(self.model_front)
        i = 0
        for model in self.model_front:
            metric_1_front_list[i] = float(model[0])
            metric_2_front_list_temp[i] = float(model[1])
            i+=1
        print(metric_2_front_list_temp)
        metric_2_front_list = [value / float(max_rule_set_size) for value in metric_2_front_list_temp]
        print(metric_2_front_list)
        # Plot Setup *********************************************
        plt.figure(figsize=(10,6)) #(10, 8))
        plt.imshow(Z, extent=[0.00, 1.00, 0.00, 1.00], interpolation='nearest', origin='lower', cmap='magma', aspect='auto') #cmap='viridis' 'magma', alpha=0.6
        # Plot model front ***************************************
        plt.plot(np.array(metric_2_front_list), np.array(metric_1_front_list), 'o-', ms=10, lw=2, color='black')
        # Plot pareto front boundaries to plot edge
        plt.plot([metric_2_front_list[-1],1],[metric_1_front_list[-1],metric_1_front_list[-1]],'--',lw=1, color='black') # horizontal line
        plt.plot([metric_2_front_list[0],metric_2_front_list[0]],[metric_1_front_list[0],0],'--',lw=1, color='black') # vertical line
        # Add colorbar for the gradient
        cbar = plt.colorbar(shrink=0.8)
        cbar.set_label('Fitness Value')
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['0', '0.25', '0.5', '0.75', '1.0'])
        # Add labels and title
        plt.xlabel('Rule-Set Size', fontsize=14)
        plt.ylabel('Coverage Penalized Balanced Accuracy', fontsize=14)
        # Set the axis limits between 0 and 1
        plt.xlim(0.00, 1.00)
        plt.ylim(0.00, 1.00)
        #custom_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        #custom_labels = [int(x * self.metric_limits[1]) for x in [0, 0.2, 0.4, 0.6, 0.8, 1.0]]
        #plt.xticks(custom_ticks, custom_labels)
        def create_x_tick_transform(multiplier):
            def x_tick_transform(x, pos):
                return f"{x * multiplier:.1f}"  # Multiply by value and convert to an integer
            return x_tick_transform
        plt.gca().xaxis.set_major_formatter(FuncFormatter(create_x_tick_transform(max_rule_set_size)))
        # Prepare to plot model population ***********************
        if plot_models:
            master_metric_2_list = []
            for value in master_metric_2_list_noscale:
                master_metric_2_list.append(value / float(max_rule_set_size))
            plt.plot(np.array(master_metric_2_list), np.array(master_metric_1_list), 'o', ms=3, lw=1, color='grey')
        plt.legend(loc='upper left', bbox_to_anchor=(1.25, 1), fontsize='small')
        plt.subplots_adjust(right=0.75)
        if save:
            plt.savefig(output_path+'/'+data_name+'_pareto_fitness_landscape_models.png', bbox_inches="tight")
        if show:
            plt.show()
