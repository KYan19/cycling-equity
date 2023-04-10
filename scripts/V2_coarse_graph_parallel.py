import osmnx
import networkx as nx
import pandas as pd
import pickle
import geopandas
import itertools
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from shapely.geometry import Point, LineString

def coarsify_nodes(dem_data):
    '''use ACS data to find centroids of city tracts, which serve as nodes in the coarse-grained network'''
    # Calculate centroids of tract regions as new coarse nodes and save in GeoDataFrame
    coarse_nodes = {'osmid':dem_data['tract'],'x':dem_data.centroid.x, 'y':dem_data.centroid.y,
                 'geometry':dem_data['geometry'],'tract':dem_data['tract']}
    coarse_nodes = geopandas.GeoDataFrame(coarse_nodes)
    coarse_nodes = coarse_nodes.set_index('osmid')
    return coarse_nodes

def label_layer_edges(layer_edges,coarse_nodes):
    '''within dataframe, label each edge with the census tracts that it intersects'''
    # label each edge when it intersects a census tract (geometry stored in coarse_nodes)
    labelled_edges = layer_edges.sjoin(coarse_nodes,how='inner',predicate='intersects')
    # edges that cross between two tracts will appear in two rows
    # aggregate those rows so each edge is labelled once, with a joined string of all tracts it intersects
    agg_functions = {'highway':'first','length':'first','geometry':'first',
         'tract':''.join}
    labelled_edges = labelled_edges.groupby(['u','v','key']).aggregate(agg_functions)
    return labelled_edges

def avg_shortest_path_pool(x):
    '''helper function to split work across nodes'''
    start = x[0]
    end = x[1]
    G = x[2]
    # if path exists between two points, add shortest path length to running total
    try:
        dist = nx.shortest_path_length(G,start,end,weight='length')
        return dist
    # skip if there is no path between the two points
    except nx.NetworkXNoPath:
        return np.NAN
    
def avg_shortest_path(G,nodes,u,v):
    '''find the mean of all shortest paths between possible nodes in two tracts
       INPUT: networkx graph for a single layer, nodes from that layer, origin and destination census tracts
       OUTPUT: length of shortest path between origin and destination in this layer, or NaN if no such path exists'''

    # compute all shortest paths between the two tracts
    length_shortest_paths = pool.map(avg_shortest_path_pool, 
            itertools.product(nodes[nodes['tract']==u].index, nodes[nodes['tract']==v].index,[G]))
    
    avg_shortest_path = np.nanmean(list(length_shortest_paths))
    return avg_shortest_path

def coarsify_edges(coarse_nodes,streets,nodes_wtract):
    '''INPUT: GeoDataFrame of coarse nodes, list of GeoDataFrames for labelled edges from different layers,
              list of of GeoDataFrames for labelled nodes from different layers
       OUTPUT: GeoDataFrame of coarse edges, with distances from all layers consolidated into a single column'''
    # label edges with what tracts they intersect
    labelled_edges = [label_layer_edges(layer_edges,coarse_nodes) for layer_edges in streets]
    
    # now that we are done using Polygon geometry of tracts, change to centroid Points
    coarse_nodes['geometry'] = [Point(x_i,y_i) for x_i,y_i in zip(coarse_nodes['x'].values,coarse_nodes['y'].values)]
    
    coarse_edges = []
    
    # enumerate possible pairs of distinct origin-destination tract pairs
    for u,u_x,u_y in zip(coarse_nodes['tract'].values,coarse_nodes['x'].values,coarse_nodes['y'].values):
        for v,v_x,v_y in zip(coarse_nodes['tract'].values,coarse_nodes['x'].values,coarse_nodes['y'].values):
            
            if u==v:
                pass 
            
            distances = [np.NaN]*len(labelled_edges)
            
            # do calculations for each layer at a time
            for layer_ix, layer_edges in enumerate(labelled_edges):
                # check if there is at least one edge in this layer crossing the two tracts
                # otherwise it is impossible for there to be any direct path from tract u to v
                if (layer_edges['tract'].isin([u+v,v+u])).sum()>0:    
                    
                    # only look at nodes in two tracts of interest
                    nodes = nodes_wtract[layer_ix]
                    nodes = nodes[nodes['tract'].isin([u,v])]
                    
                    # only look at edges in/between the two tracts of interest
                    edges = layer_edges[layer_edges['tract'].isin([u,v,u+v,v+u])]
                    edges = geopandas.GeoDataFrame(edges)

                    # construct subgraph of just the two tracts of interest
                    G = osmnx.graph_from_gdfs(nodes,edges)
                    
                    # calculate average shortest path length between these two tracts for this layer
                    distances[layer_ix] = avg_shortest_path(G,nodes,u,v)
            
            # add edge connecting these two tracts to coarse graph, if there is a path in at least one layer
            if (~np.isnan(distances)).any():
                coarse_edges.append({
                                    'u': u,
                                    'v': v,
                                    'key': 0,
                                    'geometry':  LineString([(u_x,u_y),(v_x,v_y)]),
                                    'distance': distances
                                })
                
    coarse_edges = geopandas.GeoDataFrame(coarse_edges) # convert to GeoDataFrame
    coarse_edges = coarse_edges.set_index(['u','v','key'])
    return coarse_edges

def add_tracts(nodes,coarse_nodes):
    '''INPUT: GeoDataFrame of nodes for a single layer
       OUTPUT: GeoDataFrame of nodes with tract information'''
    nodes_wtract = nodes.sjoin(coarse_nodes,how='inner',predicate='intersects')
    nodes_wtract = nodes_wtract.rename(columns={'x_left':'x','y_left':'y'})
    nodes_wtract = nodes_wtract.set_index('osmid')
    return nodes_wtract

def coarsify_graph(nodes,streets,dem):
    '''given a fine-grained graph from OSM data, make coarse-grained by aggregating nodes to the census tract level
       INPUT: list of GeoDataFrames for nodes from different street type layers, 
              list of GeoDtaFrames for edges from different street type layers,
              DataFrame of demographic information
       OUTPUT: a single networkx graph with coarsified nodes and edges, encapsulating information for all layers'''
    # get coarse nodes (centroids of census tracts from demographic data)
    coarse_nodes = coarsify_nodes(dem)
    
    # label nodes and edges with information about what tracts they intersect
    nodes_wtract = [add_tracts(node_list,coarse_nodes) for node_list in nodes]

    # coarsify edges and collapse information about different layers into a single GeoDataFrame
    coarse_edges = coarsify_edges(coarse_nodes,streets,nodes_wtract)
    
    graph_coarse = osmnx.graph_from_gdfs(coarse_nodes,coarse_edges)
    return graph_coarse

pool = mp.Pool()

dem = geopandas.read_file('Philly_dem')
nodes = [geopandas.read_file('Philly_nodes_'+str(ix)) for ix in range(4)]
streets = [geopandas.read_file('Philly_streets_'+str(ix)) for ix in range(4)]

graph_coarse = coarsify_graph(nodes,streets,dem)
with open('V2_Philly_coarse_graph_cluster', 'wb') as f:
    pickle.dump(graph_coarse, f, pickle.HIGHEST_PROTOCOL)