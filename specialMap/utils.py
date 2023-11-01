import jax.numpy as np
import numpy as onp
from collections import namedtuple
import pandas as pd
from tqdm import tqdm
import pickle
import networkx as nx
import json

Step=namedtuple('Step',['state','action'])

def loadTraj(path,pathGraph,num_trajs=None):
    with open(path, 'rb') as file:
        all_walks = pickle.load(file)
    data_trajs=np.array(all_walks)

    G = nx.read_graphml(pathGraph)


    if num_trajs is not None:
        data_trajs = data_trajs[:num_trajs]

    state_next_state = []
    action_next_action = []

    s_dim=10

    for traj in tqdm(data_trajs, desc="Processing trajectories"):
        for t in range(len(traj)):
            s_n_s = onp.zeros((2, s_dim))
            
            # Convert to integer if it's a single-item numpy array
            this_state_id = int(traj[t][0]) if isinstance(traj[t][0], np.ndarray) and traj[t][0].size == 1 else traj[t][0]
            next_state_id = int(traj[t+1][0]) if isinstance(traj[t+1][0], np.ndarray) and traj[t+1][0].size == 1 else traj[t+1][0]

            # Get features from graph
            this_state_feature = np.array(json.loads(G.nodes[str(this_state_id)]['feature']))
            next_state_feature = np.array(json.loads(G.nodes[str(next_state_id)]['feature']))

            s_n_s[0, :] = this_state_feature
            s_n_s[1, :] = next_state_feature
            state_next_state.append(s_n_s)

            a_n_a = onp.zeros((2, 1))
            a_n_a[0] = traj[t][1]
            a_n_a[1] = traj[t + 1][1]
            action_next_action.append(a_n_a)
    
    state_next_state = np.array(onp.array(state_next_state))
    action_next_action = np.array(onp.array(action_next_action))

    a_dim = int(action_next_action.max() + 1)

    inputs = state_next_state
    targets = action_next_action

    return inputs, targets, a_dim, s_dim

if __name__ == "__main__":
    path = f"./data/communityGraph/all_walks.pkl"
    pathGraph = f'./data/communityGraph/communityGraph.graphml'
    inputs,targets,a_dim,_=loadTraj(path,pathGraph)
    print(inputs.shape,targets.shape,a_dim)