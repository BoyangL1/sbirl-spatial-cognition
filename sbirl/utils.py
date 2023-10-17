import jax.numpy as np
import numpy as onp
from collections import namedtuple
import pandas as pd

Step=namedtuple('Step',['state','action'])

def loadTraj(traj_file,num_trajs=None):
    data_trajs=onp.load(traj_file,allow_pickle=True)
    if num_trajs is not None:
        data_trajs = data_trajs[:num_trajs]

    state_next_state = []
    action_next_action = []

    state_attribute=pd.read_excel('./data/nanshan_tfidf.xlsx')
    s_dim=state_attribute.shape[1]-1 # columns-1

    for traj in data_trajs:
        for t in range(len(traj) - 1):
            s_n_s = onp.zeros((2, s_dim))
            this_state,next_state=traj[t].state,traj[t+1].state
            row = state_attribute[state_attribute['fnid'] == this_state]

            s_n_s[0, :] = np.array(row.values[0][1:31])
            row = state_attribute[state_attribute['fnid'] == next_state]
            s_n_s[1, :] = np.array(row.values[0][1:31])
            state_next_state.append(s_n_s)

            a_n_a = onp.zeros((2, 1))
            a_n_a[0] = traj[t].action
            a_n_a[1] = traj[t + 1].action
            action_next_action.append(a_n_a)
    
    state_next_state = np.array(onp.array(state_next_state))
    action_next_action = np.array(onp.array(action_next_action))

    a_dim = int(action_next_action.max() + 1)

    inputs = state_next_state
    targets = action_next_action

    return inputs, targets, a_dim, s_dim

if __name__ == "__main__":
    path = f"./data/state_action_tuple.npy"
    inputs,targets,a_dim=loadTraj(path,15)
