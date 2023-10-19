import jax.numpy as np
import numpy as onp
import pandas as pd
import json

from collections import namedtuple


TravelData = namedtuple('TravelChain', ['date', 'travel_chain','id_chain'])

def loadTrajChain(traj_file,full_traj_path, num_trajs=None):
    with open(traj_file, 'r') as file:
        loaded_dicts_list1 = json.load(file)
    traj_chains = [TravelData(**d) for d in loaded_dicts_list1]
    
    all_chains = [TravelData(**d) for d in json.load(open(full_traj_path, 'r'))]
    a_dim = max({id for tc in all_chains for id in tc.id_chain})+1 # action taken by all trajectory

    if num_trajs is not None:
        traj_chains = traj_chains[:num_trajs]

    state_next_state = []
    action_next_action = []

    state_attribute=pd.read_excel('./data/SZ_tfidf.xlsx')
    s_dim=state_attribute.shape[1]-1 # columns-1

    for tc in traj_chains:
        for t in range(len(tc.travel_chain)-1):
            s_n_s = onp.zeros((2, s_dim))
            this_state,next_state=tc.travel_chain[t],tc.travel_chain[t+1]
            row = state_attribute[state_attribute['fnid'] == this_state]
            s_n_s[0, :] = np.array(row.values[0][1:31])
            row = state_attribute[state_attribute['fnid'] == next_state]
            s_n_s[1, :] = np.array(row.values[0][1:31])
            state_next_state.append(s_n_s)

            a_n_a = onp.zeros((2,1))
            a_n_a[0] = tc.id_chain[t+1]
            try:
                a_n_a[1] = tc.id_chain[t+2]
            except IndexError:
                a_n_a[1] = -1 
            action_next_action.append(a_n_a)

    state_next_state = np.array(onp.array(state_next_state))
    action_next_action = np.array(onp.array(action_next_action))

    inputs = state_next_state
    targets = action_next_action

    return inputs, targets, a_dim, s_dim
    
if __name__ == "__main__":
    path = f'./data/before_migrt.json'
    full_traj_path = f'./data/all_traj.json'
    inputs, targets, a_dim, s_dim =loadTrajChain(path,full_traj_path)
    print(inputs.shape,targets.shape,a_dim)