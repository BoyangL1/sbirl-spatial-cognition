import matplotlib.pyplot as plt
import networkx as nx
import random

import pickle
import os
import json
import numpy as np

def createCommunityGraph():
    '''
    Community structure graph
    '''
    
    G = nx.Graph()

    for i in range(5):
        node_cluster = range(i*7, i*7+7)
        for node in node_cluster:
            G.add_node(node)
        for j in node_cluster:
            for k in node_cluster:
                if j < k:
                    G.add_edge(j, k)
    
    for node in G.nodes():
        G.nodes[node]['feature'] = json.dumps(list(np.random.randn(10)))

    G.add_edge(0, 29)
    G.add_edge(28, 22)
    G.add_edge(21, 15)
    G.add_edge(14, 8)
    G.add_edge(7, 1)

    return G

def draw_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.show()

def randomWalk(G, start_node, walk_length):
    walk = []
    current_node = start_node
    for i in range(walk_length):
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            break
        next_node = random.choice(neighbors)
        walk.append((current_node, next_node))
        current_node = next_node
    return walk

if __name__ == "__main__":
    G = createCommunityGraph()
    draw_graph(G)
    directory = './data/communityGraph/'

    path_to_save = os.path.join(directory, "communityGraph.graphml")
    nx.write_graphml(G, path_to_save)

    all_walks = []
    num_walks = 1000
    walk_length = 10
    for _ in range(num_walks):
        start_node = random.choice(list(G.nodes()))
        walked_nodes_tuples = randomWalk(G, start_node, walk_length)
        all_walks.append(walked_nodes_tuples)
    
    with open(os.path.join(directory, 'all_walks.pkl'), 'wb') as file:
        pickle.dump(all_walks, file)

    with open(os.path.join(directory, 'all_walks.pkl'), 'rb') as file:
        all_walks = pickle.load(file)
        
    for i, walk in enumerate(all_walks[:5]):
        print(f"Walk {i+1}:", walk)