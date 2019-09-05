import osmnx as ox
import networkx as nx
import os
import pandas as pd
import numpy as np

def driving_distance(area_graph, startpoint, endpoint):
    """
    This function calculates the driving distance along an osmnx street network between two coordinate-points.
    """

    #Find nodes closest to the specified Coordinates
    node_start = ox.utils.get_nearest_node(area_graph, startpoint)
    node_stop = ox.utils.get_nearest_node(area_graph, endpoint)
    try:
        #Calculate the shortest network distance between the nodes via the edges length
        distance = round(nx.shortest_path_length(area_graph, node_start, node_stop, weight="length"),0)
    except nx.NetworkXNoPath:
        distance = "NA"
    return distance

print("Downloading Dublin maps from OpenStreetMap API...")
#%matplotlib inline
#Load
G = ox.graph_from_place('Dublin, Ireland', network_type='drive')

ox.save_graph_osm(G, filename='mynetwork.osm')
E = ox.graph_from_file(filename='data\mynetwork.osm', bidirectional = False)


#Settings for Streetnetwork-Download
map_file = 'data\mynetwork.osm'
force_creat = False

#This Checks if the Streetnetwork File exists (or creation is overwritten)
if (not os.path.isfile(map_file))or force_creat:
    area_graph = ox.graph_from_place('Dublin, Ireland', network_type='drive')
    ox.save_graph_osm(area_graph, filename=map_file)
else:
    area_graph = ox.graph_from_file(filename = map_file, bidirectional = False)

print("Map download complete")

# input lat long from position file
position = pd.read_csv("position.csv")
print("position data of stations loaded.")
print("Calculating driving distance between stations by Dijkstra Algorithm for SPP...")


cordinate = []
station_number = list(position.Number)
station_number_sort = sorted(station_number)
for i in range(position.shape[0]):
    cordinate.append([round(position.iloc[i,3],6), round(position.iloc[i,4],6)])
cordinate_dict = dict(zip(station_number, cordinate))

'create dataframe to store distance value'
zero_data = np.zeros(shape = (position.shape[0], position.shape[0]))
distance_table = pd.DataFrame(zero_data, index = station_number_sort, columns = station_number_sort)

for i in station_number:
    for j in station_number:
        startpoint = cordinate_dict[i]
        endpoint = cordinate_dict[j]
        distance_table.loc[i,j] = driving_distance(area_graph,startpoint, endpoint)

distance_table.to_csv(r'raw_distance.csv')
print("Process complete")
print("Data outputed in raw_distance.csv")
