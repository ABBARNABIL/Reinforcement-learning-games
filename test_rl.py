import pickle

import numpy as np

with open("frozenLake_qTable.pkl", 'rb') as f:
    Q = pickle.load(f)

def choose_action(state):
    action = np.argmax(Q[state, :])
    return action

print(Q)