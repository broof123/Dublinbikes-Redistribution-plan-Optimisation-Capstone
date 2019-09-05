# packages required for clustering process
import pandas as pd
import numpy as np
import swat
import random
import itertools as ite
import time
import datetime

#turn-off warning for chained assignment
pd.set_option('mode.chained_assignment', None)

#import data from csv file
distance_full = pd.read_csv('DISTANCE_PROCESSED_CLUSTER.csv')
avg_full = pd.read_csv('clusterinput2003.csv')

def data_prep(distance_full, avg_full, threshold_cluster):
    '''
    this function generates a random center point for each cluster and assigns stations to those clusters
    '''
    # adjust extreme demand values of each station (if too high) to less than or equal to the capacity of the vehicle (18)
    temp_dem_list = list(avg_full.demand)
    for i in range(len(temp_dem_list)):
        if temp_dem_list[i] > 18:
            temp_dem_list[i] = 18.0
        elif temp_dem_list[i] < -18:
            temp_dem_list[i] = -18.0
    avg_full.demand = temp_dem_list

    # convert type of index and columns name in distance dataframe
    distindex = list(distance_full.columns)
    for i in range(len(distindex)):
        distindex[i] = int(distindex[i])
    distance_full.index = distindex
    distance_full.columns = distindex

    # calculate number of cluster
    avg_full.index = list(avg_full.newindex)
    number_of_cluster = round(0.1*len(avg_full.index)) - 2
    stationlist = list(avg_full.newindex)
    # create upper bound and lower bound for number of stations in each cluster
    ub_cluster = round(len(avg_full.index)/number_of_cluster) + threshold_cluster
    lb_cluster = round(len(avg_full.index)/number_of_cluster) - threshold_cluster

    return(distance_full, avg_full, stationlist, number_of_cluster, ub_cluster, lb_cluster)

def clustering_stations(number_of_cluster, stationlist, distance_full, avg_full_frame, ub_cluster, lb_cluster):
    '''
    this function generates a random center point for each cluster and assigns stations to those clusters
    '''
    #generate random center points
    init_center_list = random.sample(stationlist, k = number_of_cluster)
    no_station_in_cluster = np.ones(len(init_center_list))
    # assign cluster center to cluster ID
    for i in init_center_list:
        avg_full_frame.cluster[avg_full_frame.number == i] = i

    for i in stationlist:
        if (i not in init_center_list):
            # create mindist as a reference value to find the minimum distance
            mindist = 1000000.00
            # assign stations to each cluster to reach the lower bound for the number of stations in each cluster
            if min(no_station_in_cluster) < lb_cluster:
                for j in init_center_list:
                    if ((distance_full.loc[j, i] < mindist) and (no_station_in_cluster[init_center_list.index(j)] < lb_cluster)):
                        mindist = distance_full.loc[j, i]
                        avg_full_frame.cluster[i] = j
                no_station_in_cluster[init_center_list.index(avg_full_frame.cluster[i])] = no_station_in_cluster[init_center_list.index(avg_full_frame.cluster[i])] + 1
            # assign remaining stations to clusters that do not exceed the upper bound
            else:
                for j in init_center_list:
                    if ((distance_full.loc[j, i] < mindist) and (no_station_in_cluster[init_center_list.index(j)] < ub_cluster)):
                        mindist = distance_full.loc[j, i]
                        avg_full_frame.cluster[i] = j
                no_station_in_cluster[init_center_list.index(avg_full_frame.cluster[i])] = no_station_in_cluster[init_center_list.index(avg_full_frame.cluster[i])] + 1

    # create lists of stations and associated demands in each cluster
    demand_cluster_list_detail = []
    station_cluster_list = []
    cluster_demand_list = []
    for i in init_center_list:
        demand_cluster_list_detail += [list((avg_full_frame.demand[avg_full_frame['cluster'] == i]))]
        station_cluster_list += [list(avg_full_frame.number[avg_full_frame['cluster'] == i])]
    #create list of net demand in each cluster
    for i in range(len(init_center_list)):
        cluster_demand_list += [sum(demand_cluster_list_detail[i])]

    return(init_center_list, no_station_in_cluster, demand_cluster_list_detail,
           station_cluster_list, cluster_demand_list, avg_full_frame)

def clustering_proc(cluster_demand_list, distance_full, avg_full_frame,
                    station_cluster_list, demand_cluster_list_detail,
                    number_station_in_cluster, ub_cluster, lb_cluster, Cap):
    '''
    this function re-organizes the number of stations in each cluster
    by removing the stations that exceed the demand threshold or
    upper bound for the number of stations in a given cluster
    '''
    # remove stations from clusters which exceed demand threshold or the number of stations in a given cluster
    for i in range(len(cluster_demand_list)):
        #check if the net demand in the cluster exceed the threshold or not
        if (abs(cluster_demand_list[i]) > Cap):
            #create copy of station list, demand list for checking and tracking stations removed
            station_cluster_list_check = station_cluster_list[i].copy()
            demand_cluster_list_check = demand_cluster_list_detail[i].copy()
            cluster_demand_list_check = cluster_demand_list[i]

            for k in range(len(station_cluster_list_check)):
                if station_cluster_list_check[k] != init_center_list[i]:
                    # remove the neccesary stations from each cluster (set the cluster ID of that station to 0)
                    if (abs(cluster_demand_list_check) > (abs(cluster_demand_list_check - demand_cluster_list_check[k]))):
                        avg_full_frame.cluster[station_cluster_list_check[k]] = 0
                        # recalculate net demand per cluster
                        cluster_demand_list_check = cluster_demand_list_check - demand_cluster_list_check[k]
                        # update checklists
                        station_cluster_list_check[k] = 0
                        number_station_in_cluster[i] = number_station_in_cluster[i] - 1
                    # re-organize cluster with removed stations
                    station_cluster_list[i] = [i for i in station_cluster_list_check if i != 0]
                    demand_cluster_list_detail[i] = [demand_cluster_list_check[i] for i in range(len(station_cluster_list_check)) if station_cluster_list_check[i] != 0]
                    cluster_demand_list[i] = cluster_demand_list_check
        # check if the number of stations in a cluster exceed the upper bound
        elif number_station_in_cluster[i] > ub_cluster:
            #create copy of station list, demand list for checking and tracking stations removed
            station_cluster_list_check = station_cluster_list[i].copy()
            demand_cluster_list_check = demand_cluster_list_detail[i].copy()
            cluster_demand_list_check = cluster_demand_list[i]

            for k in range(len(station_cluster_list_check)):
                if station_cluster_list_check[k] != init_center_list[i]:
                    # remove the neccesary stations from each cluster (set the cluster ID of that station to 0)
                    if (abs(cluster_demand_list_check - demand_cluster_list_check[k])) <= Cap:
                        avg_full_frame.cluster[station_cluster_list_check[k]] = 0
                        # recalculate net demand per cluster
                        cluster_demand_list_check = cluster_demand_list_check - demand_cluster_list_check[k]
                        # update checklists
                        station_cluster_list_check[k] = 0
                        number_station_in_cluster[i] = number_station_in_cluster[i] - 1
                    # re-organize cluster with removed stations
                    station_cluster_list[i] = [i for i in station_cluster_list_check if i != 0]
                    demand_cluster_list_detail[i] = [demand_cluster_list_check[i] for i in range(len(station_cluster_list_check)) if station_cluster_list_check[i] != 0]
                    cluster_demand_list[i] = cluster_demand_list_check

    #create a list of the unlabelled stations with associated demands
    unclustered_list = list(avg_full_frame.number[avg_full_frame.cluster == 0])
    demand_unclustered_list = list(avg_full_frame.demand[avg_full_frame.number.isin(unclustered_list)])
    #create a list of the shorted clusters (cluster with the number of stations below lower bound) with the number of stations and net demand
    shorted_cluster = [init_center_list[i] for i in range(len(init_center_list)) if number_station_in_cluster[i] < lb_cluster]
    number_station_in_shorted_cluster = [number_station_in_cluster[i] for i in range(len(number_station_in_cluster)) if number_station_in_cluster[i] < lb_cluster]
    demand_shorted_cluster = [cluster_demand_list[i] for i in range(len(cluster_demand_list)) if number_station_in_cluster[i] < lb_cluster]

    # add stations to clusters that don't have enough stations and demand not exceeding the threshold
    # prioritize cluster with the minimum distance from center to that station
    if len(unclustered_list) != 0:
        for k in range(len(unclustered_list)):
            # add stations to shorted clusters first
            if len(shorted_cluster) != 0:
                # check if exist cluster doesn't have enough station or not
                if min(number_station_in_shorted_cluster) < lb_cluster:
                    #calculate the distance from current unlabelled station to each center point
                    distance_to_each_shorted_centers = list(distance_full.loc[shorted_cluster, unclustered_list[k]])
                    #find nearest shorted cluster for that station
                    Ref_min_value = 1000000.0   # ref_min_value is a reference value when finding the minimum distance
                    #add unlabelled station to the cluster that has minimum distance from center to that station
                    for i in range(len(shorted_cluster)):
                        if ((abs(demand_shorted_cluster[i] + demand_unclustered_list[k]) <= Cap)
                            and (distance_to_each_shorted_centers[i] < Ref_min_value)):

                            avg_full_frame.cluster[avg_full_frame.number.isin([unclustered_list[k]])] = shorted_cluster[i]
                            Ref_min_value = distance_to_each_shorted_centers[i]
                        # update list of stations and list of net demands in shorted clusters
                        number_station_in_shorted_cluster = []
                        demand_shorted_cluster = []
                        for i in shorted_cluster:
                            number_station_in_shorted_cluster += [list(avg_full_frame.cluster).count(i)]
                            demand_shorted_cluster += [sum(avg_full_frame.demand[avg_full_frame.cluster == i])]

            # when no shorted cluster exists, consider all clusters
            # find nearest cluster for that station
            distance_to_each_centers = list(distance_full.loc[init_center_list, unclustered_list[k]])
            Ref_min_value = 1000000.0

            for i in range(len(init_center_list)):

                if ((abs(cluster_demand_list[i] + demand_unclustered_list[k]) <= Cap)
                    and (distance_to_each_centers[i] < Ref_min_value) and (number_station_in_cluster[i]  < ub_cluster)):

                    avg_full_frame.cluster[avg_full_frame.number.isin([unclustered_list[k]])] = init_center_list[i]
                    Ref_min_value = distance_to_each_centers[i]
                # update list of stations and net demands for each cluster
                number_station_in_cluster = []
                cluster_demand_list = []
                for i in init_center_list:
                    number_station_in_cluster += [list(avg_full_frame.cluster).count(i)]
                    cluster_demand_list += [sum(avg_full_frame.demand[avg_full_frame.cluster == i])]

    return(avg_full_frame, station_cluster_list, demand_cluster_list_detail, cluster_demand_list, number_station_in_cluster )


def get_total_inner_dist(station_cluster_list, distance_full):
    '''
    this function calculate the total distance between stations in a cluster
    '''
    total_dist = np.zeros(len(station_cluster_list))
    for i in range(len(station_cluster_list)):
        for j in range(len(station_cluster_list[i])):
            for k in range(len(station_cluster_list[i])):
                total_dist[i] += distance_full.loc[station_cluster_list[i][j],station_cluster_list[i][k]]
    maxdist = max(total_dist)
    return maxdist

# store start time for calculating run time
start = time.time()
print("Preparing data...")

# preparing data by calling data_prep function, then storing outputs
data_prep_output = data_prep(distance_full, avg_full, 2)
distance_full = data_prep_output[0]
avg_full_frame = data_prep_output[1]
stationlist = data_prep_output[2]
number_of_cluster = data_prep_output[3]
ub_cluster = data_prep_output[4]
lb_cluster = data_prep_output[5]
print("Data processing complete")
print("Running constrained clustering algorithm with 20 iterations")

# number of clustering iterations to run (set by user)
iteration = 20
# create empty list to stores values
cluster_results_list = []
max_inner_sum_dist = []
demand_each_cluster = []

# if number of cluster <= 1, all stations are assigned to a single cluster
if number_of_cluster <= 1:
    avg_full_frame.cluster = 1
    cluster_results_list = list(avg_full_frame.cluster)
else:
    # if the number of cluster > 1, call the clustering_stations function
    for ite in range(iteration):
        # reset cluster of every station to 0 in the beginning of each iteration
        avg_full_frame.cluster = 0
        # call function to cluster all stations and store the outputs
        clustering_stations_output = clustering_stations(number_of_cluster, stationlist, distance_full, avg_full_frame, ub_cluster, lb_cluster)
        init_center_list = clustering_stations_output[0]
        no_station_in_cluster = clustering_stations_output[1]
        demand_cluster_list_detail = clustering_stations_output[2]
        station_cluster_list = clustering_stations_output[3]
        cluster_demand_list = clustering_stations_output[4]
        avg_full_frame = clustering_stations_output[5]

        # assign Threshold for demand limit (set by user)
        Threshold_demand = 10
        # Check if the net demand/surplus in every cluster is less than or equal to the demand threshold
        # if not, call the cluster processing function and store the outputs
        if not((abs(min(cluster_demand_list)) <= Threshold_demand) and (max(cluster_demand_list) <= Threshold_demand)):
            clustering_proc_output = clustering_proc(cluster_demand_list, distance_full, avg_full_frame,
                            station_cluster_list, demand_cluster_list_detail,
                            no_station_in_cluster, ub_cluster, lb_cluster, 10)
            avg_full_frame = clustering_proc_output[0]
            station_cluster_list = clustering_proc_output[1]
            demand_cluster_list_detail = clustering_proc_output[2]
            cluster_demand_list = clustering_proc_output[3]
            no_station_in_cluster = clustering_proc_output[4]
        #calculate distance inner cluster, save results of each iteration to a list
        max_inner_sum_dist +=  [get_total_inner_dist(station_cluster_list, distance_full)]
        cluster_results_list += [list(avg_full_frame.cluster)]
        demand_each_cluster += [cluster_demand_list]
    # Find the best cluster result from all iterations by comparing the total inner cluster distances
    Ref_min_total_dist = 100000000
    for i in range(iteration):
        if ((max_inner_sum_dist[i] < Ref_min_total_dist) and (cluster_results_list[i].count(0) == 0)):
            best_index = i
            Ref_min_total_dist = max_inner_sum_dist[i]
    # save the best result
    avg_full_frame.cluster = cluster_results_list[best_index]
    best_demand_cluster = demand_each_cluster[best_index]
# calculate run time
end = time.time()
#output the results to csv file for further optimisation
avg_full_frame.to_csv(r'cluster_output_data_2003.csv')

print("Clustering finished, running time is: ", end - start)
print("Data are outputed in cluster_output_data_2003.csv")
print("Please run the optimisation file for optimisation")
