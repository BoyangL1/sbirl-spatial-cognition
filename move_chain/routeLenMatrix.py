import geopandas as gpd
import geopy.distance
import pickle
import osmnx as ox
import networkx as nx 
import numpy as np
from tqdm import tqdm
import warnings


if __name__ == "__main__":
    # sz=gpd.read_file('../data/shenzhenCity.shp')
    # G=ox.graph_from_polygon(sz.iloc[0].geometry,network_type='all')
    # G=ox.project_graph(G)
    # ox.save_graphml(G,'./data/shenzhen_grid/sz_osmnx.graphml')

    warnings.filterwarnings("ignore")
    with open("./data/id_fnid_mapping.pkl", "rb") as f:
        id_fnid = pickle.load(f)

    gdf = gpd.read_file('./data/shenzhen_grid/shenzhen_grid.shp')
    G=ox.load_graphml('./data/shenzhen_grid/sz_osmnx.graphml')
    
    sorted_id_keys = sorted(id_fnid.keys())
    sorted_fnids = [id_fnid[key] for key in sorted_id_keys]
    distances_matrix = np.zeros((len(sorted_fnids), len(sorted_fnids)))

    for i in tqdm(range(len(sorted_fnids)), desc='Calculating distances', unit='grid'):
        fnid_origin = sorted_fnids[i]
        grid_origin = gdf[gdf['fnid'] == fnid_origin]
        
        if not grid_origin.empty:
            center_origin = grid_origin.geometry.centroid.iloc[0]
            orig_coords = (center_origin.y, center_origin.x)
            orig_node = ox.get_nearest_node(G, orig_coords)
            
            for j, fnid_destination in enumerate(sorted_fnids):
                if i == j:
                    distances_matrix[i][j] = 0
                else:
                    grid_destination = gdf[gdf['fnid'] == fnid_destination]
                    if not grid_destination.empty:
                        center_destination = grid_destination.geometry.centroid.iloc[0]
                        dest_coords = (center_destination.y, center_destination.x)
                        dest_node = ox.get_nearest_node(G, dest_coords)
                        
                        try:
                            # Attempt to calculate road network distance and update matrix
                            distance = nx.shortest_path_length(G, orig_node, dest_node, weight='length')
                            distances_matrix[i][j] = distance
                        except nx.NetworkXNoPath:
                            # If no path is found, calculate the straight-line distance
                            print('there is no road path between {} and {}'.format(i,j))
                            distances_matrix[i][j] = geopy.distance.geodesic(orig_coords, dest_coords).meters
    print(distances_matrix)
    with open('./data/distances_matrix.pkl', 'wb') as f:
        pickle.dump(distances_matrix, f)