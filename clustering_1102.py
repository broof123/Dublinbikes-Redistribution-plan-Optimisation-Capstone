import pandas as pd
import numpy as np
import swat
import random
import itertools as ite
import time
import datetime

pd.set_option('mode.chained_assignment', None)

distance_full = pd.read_csv('DISTANCE_PROCESSED_CLUSTER.csv')
avg_full = pd.read_csv('clusterinput1102.csv')

def data_prep(distance_full, avg_full, threshold_cluster):
    #adjust extreme demand of single stations to less than or equal to the cap of vehicle
    temp_dem_list = list(avg_full.demand)
    for i in range(len(temp_dem_list)):
        if temp_dem_list[i] > 18:
            temp_dem_list[i] = 18.0
        elif temp_dem_list[i] < -18:
            temp_dem_list[i] = -18.0
    avg_full.demand = temp_dem_list

    distindex = list(distance_full.columns)
    for i in range(len(distindex)):
        distindex[i] = int(distindex[i])
    distance_full.index = distindex
    distance_full.columns = distindex

    avg_full.index = list(avg_full.newindex)
    number_of_cluster = round(0.1*len(avg_full.index)) - 2
    stationlist = list(avg_full.newindex)

    ub_cluster = round(len(avg_full.index)/number_of_cluster) + threshold_cluster
    lb_cluster = round(len(avg_full.index)/number_of_cluster) - threshold_cluster

    return(distance_full, avg_full, stationlist, number_of_cluster, ub_cluster, lb_cluster)

def clustering_stations(number_of_cluster, stationlist, distance_full, avg_full_frame, ub_cluster, lb_cluster):
    init_center_list = random.sample(stationlist, k = number_of_cluster)

    no_station_in_cluster = np.ones(len(init_center_list))

    for i in init_center_list:
        avg_full_frame.cluster[avg_full_frame.number == i] = i

    for i in stationlist:
        if (i not in init_center_list):
            mindist = 1000000.00
            # assign stations to each cluster to reach lower bound
            #while min(no_station_in_cluster) < lb_cluster:
            if min(no_station_in_cluster) < lb_cluster:
                for j in init_center_list:
                    if ((distance_full.loc[j, i] < mindist) and (no_station_in_cluster[init_center_list.index(j)] < lb_cluster)):
                        mindist = distance_full.loc[j, i]
                        avg_full_frame.cluster[i] = j
                #temp_index = avg_full_frame.cluster[stationlist.index(i)]
                no_station_in_cluster[init_center_list.index(avg_full_frame.cluster[i])] = no_station_in_cluster[init_center_list.index(avg_full_frame.cluster[i])] + 1


            # assign remain stations to cluster that not exceed upper bound
            else:
                for j in init_center_list:
                    if ((distance_full.loc[j, i] < mindist) and (no_station_in_cluster[init_center_list.index(j)] < ub_cluster)):
                        mindist = distance_full.loc[j, i]
                        avg_full_frame.cluster[i] = j
                #temp_index = avg_full_frame.cluster[stationlist.index(i)]
                no_station_in_cluster[init_center_list.index(avg_full_frame.cluster[i])] = no_station_in_cluster[init_center_list.index(avg_full_frame.cluster[i])] + 1

    demand_cluster_list_detail = []
    station_cluster_list = []
    cluster_demand_list = []
    for i in init_center_list:
        demand_cluster_list_detail += [list((avg_full_frame.demand[avg_full_frame['cluster'] == i]))]
        station_cluster_list += [list(avg_full_frame.number[avg_full_frame['cluster'] == i])]


    for i in range(len(init_center_list)):
        cluster_demand_list += [sum(demand_cluster_list_detail[i])]


    return(init_center_list, no_station_in_cluster, demand_cluster_list_detail,
           station_cluster_list, cluster_demand_list, avg_full_frame)

def clustering_proc(cluster_demand_list, distance_full, avg_full_frame,
                    station_cluster_list, demand_cluster_list_detail,
                    number_station_in_cluster, ub_cluster, lb_cluster, Cap):
    '''
    this function re-organize number of stations in each cluster by remove stations from cluster exceed demand threshold or
    over upper bound of number of stations
    '''
    # remove stations from cluster exceed demand cap or exceed number of station in each cluster
    for i in range(len(cluster_demand_list)):
        if (abs(cluster_demand_list[i]) > Cap):
            #dist_to_centers_temp = []
            #for k in range(len(station_cluster_list[i])):
            #    dist_to_centers_temp += [(sum(distance_full.loc[init_center_list, station_cluster_list[i][k]])
            #                                  - distance_full.loc[init_center_list[i], station_cluster_list[i][k]])]
            #build function for calculate total distance to centers
            station_cluster_list_check = station_cluster_list[i].copy()                #create copy of station list for ref
            demand_cluster_list_check = demand_cluster_list_detail[i].copy()              # create copy of demand list for ref
            cluster_demand_list_check = cluster_demand_list[i]

            for k in range(len(station_cluster_list_check)):
                if station_cluster_list_check[k] != init_center_list[i]:
                    # find station with min distance to centers
                    if (abs(cluster_demand_list_check) > (abs(cluster_demand_list_check - demand_cluster_list_check[k]))):
                        avg_full_frame.cluster[station_cluster_list_check[k]] = 0            # set cluster of min item to 0, remove from current cluste
                        # recalculate demand
                        cluster_demand_list_check = cluster_demand_list_check - demand_cluster_list_check[k]
                        # update check list
                        station_cluster_list_check[k] = 0

                        #dist_to_centers_temp.pop(k)
                        number_station_in_cluster[i] = number_station_in_cluster[i] - 1

                    station_cluster_list[i] = [i for i in station_cluster_list_check if i != 0]
                    demand_cluster_list_detail[i] = [demand_cluster_list_check[i] for i in range(len(station_cluster_list_check)) if station_cluster_list_check[i] != 0]
                    cluster_demand_list[i] = cluster_demand_list_check

        elif number_station_in_cluster[i] > ub_cluster:
            #dist_to_centers_temp = []
            #for k in range(len(station_cluster_list[i])):
            #    dist_to_centers_temp += [(sum(distance_full.loc[init_center_list, station_cluster_list[i][k]])
            #                                  - distance_full.loc[init_center_list[i], station_cluster_list[i][k]])]
            station_cluster_list_check = station_cluster_list[i].copy()         #create copy of station list for ref
            demand_cluster_list_check = demand_cluster_list_detail[i].copy()    # create copy of demand list for ref
            cluster_demand_list_check = cluster_demand_list[i]

            for k in range(len(station_cluster_list_check)):
                if station_cluster_list_check[k] != init_center_list[i]:
                    if (abs(cluster_demand_list_check - demand_cluster_list_check[k])) <= Cap:
                        avg_full_frame.cluster[station_cluster_list_check[k]] = 0  # set cluster of min item to 0, remove from current cluste
                        # recalculate demand
                        cluster_demand_list_check = cluster_demand_list_check - demand_cluster_list_check[k]
                        # update check list
                        station_cluster_list_check[k] = 0

                        #dist_to_centers_temp.pop(k)
                        number_station_in_cluster[i] = number_station_in_cluster[i] - 1

                    station_cluster_list[i] = [i for i in station_cluster_list_check if i != 0]
                    demand_cluster_list_detail[i] = [demand_cluster_list_check[i] for i in range(len(station_cluster_list_check)) if station_cluster_list_check[i] != 0]
                    cluster_demand_list[i] = cluster_demand_list_check

    unclustered_list = list(avg_full_frame.number[avg_full_frame.cluster == 0])
    demand_unclustered_list = list(avg_full_frame.demand[avg_full_frame.number.isin(unclustered_list)])
    shorted_cluster = [init_center_list[i] for i in range(len(init_center_list)) if number_station_in_cluster[i] < lb_cluster]
    number_station_in_shorted_cluster = [number_station_in_cluster[i] for i in range(len(number_station_in_cluster)) if number_station_in_cluster[i] < lb_cluster]
    demand_shorted_cluster = [cluster_demand_list[i] for i in range(len(cluster_demand_list)) if number_station_in_cluster[i] < lb_cluster]

    # adding back stations to cluster that doesn't have enough stations and demand not exceed the cap
    if len(unclustered_list) != 0:
        for k in range(len(unclustered_list)):
            if len(shorted_cluster) != 0:
                if min(number_station_in_shorted_cluster) < lb_cluster:
                    distance_to_each_shorted_centers = list(distance_full.loc[shorted_cluster, unclustered_list[k]])
                    #find nearest shorted clusters for that station
                    #minindex_distance_to_shorted_center = np.argmin(distance_to_each_shorted_centers)
                    Ref_min_value = 1000000.0

                    for i in range(len(shorted_cluster)):
                        if ((abs(demand_shorted_cluster[i] + demand_unclustered_list[k]) <= Cap)
                            and (distance_to_each_shorted_centers[i] < Ref_min_value)):

                            avg_full_frame.cluster[avg_full_frame.number.isin([unclustered_list[k]])] = shorted_cluster[i]
                            Ref_min_value = distance_to_each_shorted_centers[i]

                        number_station_in_shorted_cluster = []
                        demand_shorted_cluster = []
                        for i in shorted_cluster:
                            number_station_in_shorted_cluster += [list(avg_full_frame.cluster).count(i)]
                            demand_shorted_cluster += [sum(avg_full_frame.demand[avg_full_frame.cluster == i])]
            distance_to_each_centers = list(distance_full.loc[init_center_list, unclustered_list[k]])
            #find nearest shorted clusters for that station
            #minindex_distance_to_centers = np.argmin(distance_to_each_centers)
            Ref_min_value = 1000000.0

            for i in range(len(init_center_list)):

                if ((abs(cluster_demand_list[i] + demand_unclustered_list[k]) <= Cap)
                    and (distance_to_each_centers[i] < Ref_min_value) and (number_station_in_cluster[i]  < ub_cluster)):

                    avg_full_frame.cluster[avg_full_frame.number.isin([unclustered_list[k]])] = init_center_list[i]
                    Ref_min_value = distance_to_each_centers[i]

                #number_station_in_cluster[i] = number_station_in_cluster[i] + 1
                number_station_in_cluster = []
                cluster_demand_list = []
                for i in init_center_list:
                    number_station_in_cluster += [list(avg_full_frame.cluster).count(i)]
                    cluster_demand_list += [sum(avg_full_frame.demand[avg_full_frame.cluster == i])]

    return(avg_full_frame, station_cluster_list, demand_cluster_list_detail, cluster_demand_list, number_station_in_cluster )


def get_total_inner_dist(station_cluster_list, distance_full):
    total_dist = np.zeros(len(station_cluster_list))
    for i in range(len(station_cluster_list)):
        for j in range(len(station_cluster_list[i])):
            for k in range(len(station_cluster_list[i])):
                total_dist[i] += distance_full.loc[station_cluster_list[i][j],station_cluster_list[i][k]]
    maxdist = max(total_dist)
    return maxdist

start = time.time()
print("Preparing data...")

data_prep_output = data_prep(distance_full, avg_full, 2)
distance_full = data_prep_output[0]
avg_full_frame = data_prep_output[1]
stationlist = data_prep_output[2]
number_of_cluster = data_prep_output[3]
ub_cluster = data_prep_output[4]
lb_cluster = data_prep_output[5]
print("Data processing complete")
print("Running constrained clustering algorithm with 20 iterations")
iteration = 20
cluster_results_list = []
max_inner_sum_dist = []
demand_each_cluster = []
if number_of_cluster <= 1:
    avg_full_frame.cluster = 1
    cluster_results_list = list(avg_full_frame.cluster)
else:
    for ite in range(iteration):
        avg_full_frame.cluster = 0
        clustering_stations_output = clustering_stations(number_of_cluster, stationlist, distance_full, avg_full_frame, ub_cluster, lb_cluster)
        init_center_list = clustering_stations_output[0]
        no_station_in_cluster = clustering_stations_output[1]
        demand_cluster_list_detail = clustering_stations_output[2]
        station_cluster_list = clustering_stations_output[3]
        cluster_demand_list = clustering_stations_output[4]
        avg_full_frame = clustering_stations_output[5]

        Threshold_demand = 10               #can be change
        if not((abs(min(cluster_demand_list)) <= Threshold_demand) and (max(cluster_demand_list) <= Threshold_demand)):
            clustering_proc_output = clustering_proc(cluster_demand_list, distance_full, avg_full_frame,
                            station_cluster_list, demand_cluster_list_detail,
                            no_station_in_cluster, ub_cluster, lb_cluster, 10)
            avg_full_frame = clustering_proc_output[0]
            station_cluster_list = clustering_proc_output[1]
            demand_cluster_list_detail = clustering_proc_output[2]
            cluster_demand_list = clustering_proc_output[3]
            no_station_in_cluster = clustering_proc_output[4]
        max_inner_sum_dist +=  [get_total_inner_dist(station_cluster_list, distance_full)]
        cluster_results_list += [list(avg_full_frame.cluster)]
        demand_each_cluster += [cluster_demand_list]
    Ref_min_total_dist = 100000000
    for i in range(iteration):
        if ((max_inner_sum_dist[i] < Ref_min_total_dist) and (cluster_results_list[i].count(0) == 0)):
            best_index = i
            Ref_min_total_dist = max_inner_sum_dist[i]

    avg_full_frame.cluster = cluster_results_list[best_index]
    best_demand_cluster = demand_each_cluster[best_index]
end = time.time()
avg_full_frame.to_csv(r'cluster_output_data_1102.csv')

print("Clustering finished, running time is: ", end - start)
print("Data are outputed in cluster_output_data_1102.csv")
print("Please run the optimisation file for optimisation")
