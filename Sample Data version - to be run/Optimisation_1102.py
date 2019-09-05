# packages required for optimisation process
import pandas as pd
import cvxpy as cp
import swat
import numpy as np
from cplex import *
import random
import itertools as ite
import time
import datetime

#turn off the chained assignment warning
pd.set_option('mode.chained_assignment', None)

def process_data(distance_full, avg_full_frame):
    '''
    this function prepare data to suitable form for processing later
    '''
    distance_full.index = distance_full.columns
    avg_full_frame = avg_full_frame.drop(columns = 'Unnamed: 0')
    avg_full_frame.index = list(avg_full_frame.newindex)
    return(distance_full, avg_full_frame)

def pre_proc_cluster(dist_frame, demand_full_list, station_full_list,  cluster, t_h):
    '''this function prepares the input data for modelling'''
    #create index for later using in assignment
    index_val = len(station_full_list[cluster])

    demand = demand_full_list[cluster]
    station_index = station_full_list[cluster]
    station_proc_list = []
    demand_list = []
    # removing stations with a demand below the threshold (Removal of zero demand stations)
    for i in range(index_val):
        if abs(demand[i]) > t_h:
            station_proc_list += [str(int(station_index[i]))]
            demand_list += [demand[i]]
    distance_frame_updated = dist_frame.loc[station_proc_list, station_proc_list]
    # update distance list with the removal of stations
    dist_frame = distance_frame_updated
    index_val = len(demand_list)
    return(index_val, demand_list, dist_frame, station_proc_list)

def demand_opt(demand, index_val, cluster, cluster_ID):
    '''minimax optimisation for demand'''
    demand_check = sum(demand)
    # Assigning variables to store adjusted demand values and the Q value (the optimal demand adjustment value)
    adj_var = cp.Variable((index_val), integer = True)
    Q = cp.Variable((1), integer = True)

    # Constraints setting all variables to be less than or equal to Q and greater than or equal to 0
    const1 = []
    const2 = []
    for i in range(index_val):
        const1 += [adj_var[i] <= Q]
        const2 += [adj_var[i] >= 0]

    # Constraint forcing the total demand adjustments in a cluster to be equal to the net demand value for that cluster
    const3 = [(cp.sum(adj_var)) == abs(demand_check)]

    # Constraint forcing the adjustment to not exceed the current demand of the corresponding station
    const4 = []
    for i in range(index_val):
        const4 += [adj_var[i] <= abs(demand[i])]

    # constraint forcing the Q to be greater than or equal to 0
    const11 = [Q >= 0]

    # constraint ensuring that adjustments are only made to the demand in the stations that have the same sign as the net demand of its cluster
    const5 = []
    if demand_check < 0:
        for i in range(index_val):
            if demand[i] > 0:
                const5 += [adj_var[i] == 0]
    elif demand_check > 0:
        for i in range(index_val):
            if demand[i] < 0:
                const5 += [adj_var[i] == 0]

    # declaring objective function
    objective_func = cp.Minimize(Q)
    # combine all constraints
    const = (const1 + const2 + const11 + const3 +const4 + const5)
    # assign problem with objective function and constraints to be solved using CPLEX solver
    prob1 = cp.Problem(objective_func, const)
    prob1.solve(solver = cp.CPLEX, verbose = False)
    # get the value of adjusted amount in each station
    adj_list = list(adj_var.value)
    # calculate new demand of each stations
    if demand_check > 0:
        new_demand = [np.subtract(demand, adj_list)]
    elif demand_check < 0:
        new_demand = [np.add(demand, adj_list)]
    else:
        new_demand = [np.array(demand)]

    print("Status for optimize demand of cluster: ", cluster_ID[cluster], "is ", prob1.status, " with the maximum adjustment in each station is: ", prob1.value)

    return(new_demand, adj_list, prob1)


def adj_demand_aft_opt(new_demand, station_proc_list, dist_frame):
    '''
    this function removes stations with zero demand after adjustment and update demand list, stations list, distance list
    '''
    final_demand = [new_demand[(new_demand != 0)]]
    final_station_proc_list = [station_proc_list[i] for i in range(len(new_demand)) if abs(new_demand[i]) > 0]
    distance_frame_updated = dist_frame.loc[final_station_proc_list, final_station_proc_list]
    dist_frame = distance_frame_updated

    return(final_demand, final_station_proc_list, dist_frame)

def choose_start_point(demand, station_proc_list):
    '''
    this function creates list of starting point for each cluster
    '''
    #create list of starting point with surplus
    start_point_list = []
    start_point_index_list = []
    for i in range(len(station_proc_list)):
        if (demand[i] > 0):
            start_point_list += [station_proc_list[i]]
            start_point_index_list += [i]
    return(start_point_list, start_point_index_list)

def main_opt(distance_frame1, index_val, demand, Cap, start_point):
    '''
    function for main optimization
    '''
    # assign BigM for later use
    BigM = 100000
    # convert original distance data to numpy array
    distance_np_array = distance_frame1.to_numpy()

    # create binary variables for the selection of arcs (1 if selected, 0 otherwise)
    select_bivar = cp.Variable((index_val, index_val), name = "selected or not var", boolean = True)

    # create variables for the load in the vehicles after visiting a station
    load_aft_visit = cp.Variable((index_val), integer = True)

    #create lower and upper bound for the load after visiting each station
    lb_load = np.zeros((index_val))
    for i in range(index_val):
        lb_load[i] = max(0, demand[i])

    ub_load = np.zeros((index_val))
    for i in range(index_val):
        ub_load[i] = min(Cap, Cap + demand[i])

    # create an array by multiplying distances and selection variables for use in objective function
    dist_select_array = cp.multiply(distance_np_array, select_bivar)

    # create the sum of selection variables for inflow and outflow of each station
    expr1_xij = cp.sum(select_bivar, axis = 1, keepdims = False)  # outflow
    expr2_xji = cp.sum(select_bivar, axis = 0, keepdims = False)  # inflow
    expr3_x0j = cp.sum(select_bivar[start_point,:])
    expr4_xj0 = cp.sum(select_bivar[:,start_point])

    sum_dist_obj = cp.sum(dist_select_array)

    # assign objective function to be the sum of driving distances through the nodes in cluster with final arc removed
    # to make the route of vehicels a path, not a tour
    objective = cp.Minimize(sum_dist_obj - cp.sum(dist_select_array[:,start_point]))

    # constraint 1 and 2 ensure that each station is only visited 1 time
    constraint1 = []
    for i in range(index_val):
        constraint1 += [expr1_xij[i] == 1]

    constraint2 = []
    for i in range(index_val):
        constraint2 += [expr2_xji[i] == 1]

    # Constraint 3 and 31 create artifical arc from endpoint to start point (the value of distance of this arc is removed in objective function)
    constraint3 = [expr3_x0j == 1]
    constraint31 = [expr4_xj0 == 1]

     # subtour elimination constraint for pairs and individual nodes
    constraint4 = []
    constraint41 = []             # need to adjust for all set of index, not only one
    for i in range(index_val):
        for j in range(index_val):
            if (j != i):
                constraint4 += [select_bivar[i,j] + select_bivar[j,i] <= 1]
            if (j == i):
                constraint41 += [select_bivar[i,j] == 0]

    # constraint on the number of bikes loaded and un-loaded at each station to ensure they exactly match the demand/surplus at that station
    constraint5 = []
    constraint6 = []
    for i in range(index_val):
        for j in range(index_val):
            if j != start_point:
                constraint5 += [load_aft_visit[j] - load_aft_visit[i] >= demand[j] - BigM*(1-select_bivar[i,j])]

    for i in range(index_val):
        if i != start_point:
            for j in range(index_val):
                 constraint6 += [load_aft_visit[j] - load_aft_visit[i] <= demand[j] + BigM*(1-select_bivar[i,j])]

    # constraints on lower and upper bound for the load in each vehicle after visiting each station
    constraint7 = []
    constraint8 = []
    for i in range(index_val):
        constraint7 += [load_aft_visit[i] >= lb_load[i]]
        constraint8 += [load_aft_visit[i] <= ub_load[i]]

    # constraints on the load in each vehicle after visiting the final station being equal to 0
    constraint10 = []
    constraint11 = []
    for i in range(index_val):
        if i != start_point:
            constraint10 += [load_aft_visit[i] <= 0 + BigM*(1-select_bivar[i, start_point])]
            constraint11 += [load_aft_visit[i] >= 0 - BigM*(1-select_bivar[i, start_point])]

    #combined all constraint
    constraint = (constraint1 + constraint2 + constraint3 + constraint31 + constraint4 + constraint41
              + constraint5 + constraint6
              + constraint7 + constraint8
              + constraint10 + constraint11)

    # assign problem with objective function and constraints to be solved using CPLEX solver
    prob = cp.Problem(objective,constraint)
    prob.solve(solver = cp.CPLEX, verbose = False, warm_start = False)

    # create a list pf combinations from 3 to (n-1) stations for subtour checking
    ver_combine = []
    for i in range(3, (index_val-1)):
        ver_combine += list(ite.combinations(list(range(index_val)), i))

    # detect and add subtour eliminations constraints
    constraint12 = []
    for j in range(len(ver_combine)):
        sum_arc_value = sum(select_bivar[list(ver_combine[j])][:,list(ver_combine[j])].value)
        sum_arc_check = cp.sum(select_bivar[list(ver_combine[j])][:, list(ver_combine[j])])
        if (sum(sum_arc_value) >= len(ver_combine[j])):
            constraint12 += [sum_arc_check <= (len(ver_combine[j]) - 1)]

    # iterating through the subtour detection process until there are no subtours in the temporary solution
    new_constraint = constraint
    if len(constraint12) == 0:
        new_prob = prob
    else:
        while len(constraint12) != 0:
            new_prob =[]
            new_constraint += constraint12
            new_prob = cp.Problem(prob.objective, new_constraint)
            new_prob.solve(solver = cp.CPLEX, verbose = False, warm_start = False)
            constraint12 = []
            for j in range(len(ver_combine)):
                sum_arc_value = sum(select_bivar[list(ver_combine[j])][:,list(ver_combine[j])].value)
                sum_arc_check = cp.sum(select_bivar[list(ver_combine[j])][:, list(ver_combine[j])])
                if (sum(sum_arc_value) >= len(ver_combine[j])):
                    constraint12 += [sum_arc_check <= (len(ver_combine[j]) - 1)]

    return(new_prob, select_bivar, load_aft_visit)

#input data from csv file
distance_full = pd.read_csv('DISTANCE_PROCESSED_CLUSTER.csv')
avg_full = pd.read_csv('cluster_output_data_1102.csv')

# store start time for calculating run time
start = time.time()

print("Processing data for input...")
# call function for processing data and store the outputs
process_data_output = process_data(distance_full, avg_full)
distance_full_frame_proc = process_data_output[0]
avg_full_frame = process_data_output[1]

# create a list of indexes and convert to integers
index_for_sta = list(avg_full_frame.number)
for i in range(len(index_for_sta)):
    index_for_sta[i] = int(index_for_sta[i])
avg_full_frame.index = index_for_sta

# create a list of cluster IDs
cluster_ID = avg_full_frame.cluster.unique()
# create subset data for each cluster
demand_full_list = []
station_full_list = []
for i in range(len(cluster_ID)):
    demand_full_list += [list(avg_full_frame.demand[avg_full_frame['cluster'] == cluster_ID[i]])]
    station_full_list += [list(avg_full_frame.number[avg_full_frame['cluster'] == cluster_ID[i]])]
weight_full_list = []
for i in range(len(cluster_ID)):
    weight_full_list += [list(avg_full_frame.Weight[avg_full_frame['cluster'] == cluster_ID[i]])]

print("Finish data processing")
print("Running Optimisation models...")

# assign threshold and vehicle capacity value (set by user)
threshold = 0
Cap = 18

# create empty list to store output values
pre_proc_cluster_output = []
index_val_main_list = []
demand_main_list = []
distance_frame1_list = []
station_proc_full_list = []

demand_opt_output_list = []
demand_new_list = []
adj_list_main_list = []
prob1_list = []

adj_demand_aft_opt_op = []
station_proc_adj_list =[]
index_val_adj_list = []
distance_frame_adj_list =[]
demand_adj_list = []

start_point_list = []
start_point_index_list = []
choose_start_point_output = []

# create prob_best_value with every element equal to 1000000.0 for reference when finding best value in each cluster
prob_best_value = [1000000.0] * len(cluster_ID)
best_start_point = np.zeros((len(cluster_ID)))
best_index = np.zeros((len(cluster_ID)))

select_bivar_best_list = []
load_aft_visit_best_list =[]
# iterate through each cluster
for cluster in range(len(cluster_ID)):
    # call the cluster processing function and store the outputs
    pre_proc_cluster_output += [pre_proc_cluster(distance_full_frame_proc, demand_full_list,
                                                 station_full_list,  cluster, threshold)]

    index_val_main_list += [pre_proc_cluster_output[cluster][0]]
    demand_main_list += [pre_proc_cluster_output[cluster][1]]
    distance_frame1_list += [pre_proc_cluster_output[cluster][2]]
    station_proc_full_list += [pre_proc_cluster_output[cluster][3]]

    # call the demand optimisation function and store the outputs
    demand_opt_output_list += [demand_opt(demand_main_list[cluster], index_val_main_list[cluster],
                                          cluster,cluster_ID)]
    demand_new_list += demand_opt_output_list[cluster][0]
    adj_list_main_list += [demand_opt_output_list[cluster][1]]
    prob1_list += [demand_opt_output_list[cluster][2]]

    # call the function that removes zero demand stations after adjustment, and store the outputs
    adj_demand_aft_opt_op += [adj_demand_aft_opt(demand_new_list[cluster], station_proc_full_list[cluster],
                                                 distance_frame1_list[cluster])]
    demand_adj_list += adj_demand_aft_opt_op[cluster][0]
    station_proc_adj_list += [adj_demand_aft_opt_op[cluster][1]]
    distance_frame_adj_list += [adj_demand_aft_opt_op[cluster][2]]
    index_val_adj_list += [len(demand_adj_list[cluster])]

    # call the function to create starting point list for each cluster
    choose_start_point_output += [choose_start_point(demand_adj_list[cluster], station_proc_adj_list[cluster])]
    start_point_list += [choose_start_point_output[cluster][0]]
    start_point_index_list += [choose_start_point_output[cluster][1]]

    print("running the solver and finding best value...")
    # check that the cluster requires redistribution
    if len(station_proc_adj_list[cluster]) != 0:
        main_prob = []
        select_bivar_list= []
        load_aft_visit_list = []
        temp_prob =[]

        # iterate through all starting points
        for i in range(len(start_point_list[cluster])):
            # call main optimisation function and store the outputs
            temp_prob += [main_opt(distance_frame_adj_list[cluster], index_val_adj_list[cluster], demand_adj_list[cluster], Cap, start_point_index_list[cluster][i])]
            main_prob += [temp_prob[i][0]]
            select_bivar_list += [temp_prob[i][1]]
            load_aft_visit_list += [temp_prob[i][2]]
            # find the best solution for each cluster
            if (temp_prob[i][0].value < prob_best_value[cluster]):
                prob_best_value[cluster] = temp_prob[i][0].value
                best_start_point[cluster] = start_point_list[cluster][i]
                best_index[cluster] = i
        # store variable values for the best solution in each cluster
        load_aft_visit_best_list += [load_aft_visit_list[int(best_index[cluster])].value]
        select_bivar_best_list += [select_bivar_list[int(best_index[cluster])].value]
        print("finish solving main optimisation model for cluster: ", cluster_ID[cluster], "with optimal value is: ", prob_best_value[cluster])
    else:
        load_aft_visit_best_list += [np.zeros(index_val_adj_list[cluster])]
        select_bivar_best_list += [np.zeros((index_val_adj_list[cluster], index_val_adj_list[cluster]))]
        print("no redistribution needed")
print("Optimisation process complete")
# calculate run time
end = time.time()
print("Running time is: ", end - start)

# create dataframe for demand before/after adjustment for exporting
print("Processing data for output...")
station_full_list_conct = []
demand_main_list_conct = []

# merge all the elements in the detailed list of stations and demands in each cluster to a single list
for i in range(len(station_proc_full_list)):
    station_full_list_conct == station_full_list_conct.extend(station_proc_full_list[i])
    demand_main_list_conct == demand_main_list_conct.extend(demand_main_list[i])
# convert stations list to integers
for i in range(len(station_full_list_conct)):
    station_full_list_conct[i] = int(station_full_list_conct[i])
# combine 2 above lists into a single dataframe
changed_array = pd.DataFrame(list(zip(station_full_list_conct, demand_main_list_conct)), index = station_full_list_conct, columns = ['number','Old_demand'])

# create lists of processed stations, demands and loads after visit (results of demand optimisation and processing)
demand_adj_list_conct =[]
station_proc_adj_list_conct = []
load_aft_visit_conct = []
for i in range(len(station_proc_adj_list)):
    demand_adj_list_conct == demand_adj_list_conct.extend(demand_adj_list[i])
    station_proc_adj_list_conct == station_proc_adj_list_conct.extend(station_proc_adj_list[i])
    load_aft_visit_conct == load_aft_visit_conct.extend(load_aft_visit_best_list[i])
# convert stations list to integers
for i in range(len(station_proc_adj_list_conct)):
    station_proc_adj_list_conct[i] = int(station_proc_adj_list_conct[i])

# combined above lists into a single dataframe
adj_array = pd.DataFrame(list(zip(station_proc_adj_list_conct, demand_adj_list_conct,load_aft_visit_conct)), index = station_proc_adj_list_conct, columns =['number','adj_demand','load_aft_visit'])

# convert number column to integer
avg_full_frame.number = avg_full_frame.number.astype(int)

# output_data_merge is the results table that needs to be exported
output_data_merge = avg_full_frame.merge(changed_array, how = 'left',  on = 'number')
output_data_merge = output_data_merge.merge(adj_array, how = 'left',  on = 'number')
output_data_merge.Old_demand = output_data_merge.demand

output_data_merge['Adj_amount'] = output_data_merge.adj_demand - output_data_merge.Old_demand

# convert station_proc_full_list_int from list of list to list, convert to int for sorting
station_proc_full_list_int = []
for i in range(len(station_proc_adj_list)):
    for j in range(len(station_proc_adj_list[i])):
        station_proc_full_list_int += [int(station_proc_adj_list[i][j])]
station_proc_full_list_int = sorted(station_proc_full_list_int)
# convert back station_proc_full_list_int to str for indexing
for i in range(len(station_proc_full_list_int)):
    station_proc_full_list_int[i] = str(station_proc_full_list_int[i])

# create a list of dataframes for the selection variable values in each cluster
select_bivar_frame_list  = []
for i in range(len(select_bivar_best_list)):
    select_bivar_frame_list += [pd.DataFrame(select_bivar_best_list[i])]
    select_bivar_frame_list[i].index = station_proc_adj_list[i]
    select_bivar_frame_list[i].columns = station_proc_adj_list[i]

# select_bivar_matrix_output is the output arcs matrix (the routes)
select_bivar_matrix_output = pd.DataFrame(0.0, index = station_proc_full_list_int,
                                            columns = station_proc_full_list_int)

# fill in values for select_bivar_matrix_output from original frame
for i in range(len(select_bivar_frame_list)):
    for j in select_bivar_frame_list[i].index:
        for k in select_bivar_frame_list[i].columns:
            select_bivar_matrix_output.loc[j, k] = select_bivar_frame_list[i].loc[j,k]

# round values to avoid errors in calculation of decimals from system then convert values to integers
select_bivar_matrix_output = np.around(select_bivar_matrix_output)
select_bivar_matrix_output = select_bivar_matrix_output.astype(int)

#create to_station list: next stations that need to be visited
to_station_list = np.zeros(len(select_bivar_matrix_output))
for i in range(len(select_bivar_matrix_output)):
    if list(select_bivar_matrix_output.iloc[i,:]).count(1) != 0:
        need_index = list(select_bivar_matrix_output.iloc[i,:]).index(1)
        to_station_list[i] = select_bivar_matrix_output.columns[need_index]
    else:
        to_station_list[i] = 0

# convert to_station list to integers
for i in range(len(to_station_list)):
    to_station_list[i] = int(to_station_list[i])

#create new columns for number and to_station
select_bivar_matrix_output['number'] = select_bivar_matrix_output.index
select_bivar_matrix_output['to_station'] =  to_station_list

# create dataframe of from_station and to_station with associated Latitudes and Longitudes
from_to_df = pd.DataFrame(list(zip(list(select_bivar_matrix_output.number),
                          list(select_bivar_matrix_output.to_station))),
                          index = select_bivar_matrix_output.index, columns = ['from_station','to_station'])

from_to_df['number'] = from_to_df['from_station'].astype(int)

from_to_df['to_lat'] = 0
from_to_df['to_log'] = 0
from_to_df['to_lat']=  from_to_df['to_lat'].astype(float)
from_to_df['to_log']=  from_to_df['to_log'].astype(float)
from_to_df['startpoint'] = 0
from_to_df['Tdistance'] = 0
for i in range(len(from_to_df.to_station)):
    if (from_to_df.to_station[i] != 0):
        from_to_df['to_lat'][i] = output_data_merge.Latitude[output_data_merge.number == from_to_df.to_station[i]]
        from_to_df['to_log'][i] = output_data_merge.Longitude[output_data_merge.number == from_to_df.to_station[i]]

#create columns with value 1 for station that is starting point of each cluster
best_start_point_int = []
for i in range(len(best_start_point)):
    best_start_point_int += [int(best_start_point[i])]

#create columns with value 1 for station that is starting point of each cluster
for i in range(len(from_to_df.to_station)):
    for j in range(len(best_start_point_int)):
        if best_start_point_int[j] == int(from_to_df.from_station[i]):
            from_to_df['startpoint'][i] = 1
        if best_start_point_int[j] == int(from_to_df.to_station[i]):
            from_to_df.to_station[i] = 0

# gather driving distance
for i in range(len(from_to_df.to_station)):
    if from_to_df.to_station[i] != 0:
        from_to_df['Tdistance'][i] = distance_full_frame_proc.loc[str(from_to_df.from_station[i]), str(int(from_to_df.to_station[i]))]

# merge newly created dataframe and original one,
# output data to csv file
output_data_merge = output_data_merge.merge(from_to_df, how = 'left',  on = 'number')
output_data_merge.to_csv(r'output_data_opted_1102.csv')
print("Data outputed in output_data_opted_1102.csv")
