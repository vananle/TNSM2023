import os

import networkx as nx
import numpy as np
import pandas as pd
from scipy.io import savemat

DATAPATH = '../../data_vae/'
DATASET = 'geant_tm'
TOPOPATH = os.path.join(DATAPATH, 'topo')


def load_network_topology(dataset, datapath):
    # initialize graph
    G = nx.DiGraph()
    # load node dataset from csv
    path = os.path.join(datapath, 'topo/{}_node.csv'.format(dataset))
    df = pd.read_csv(path, delimiter=' ')
    for i, row in df.iterrows():
        G.add_node(i, label=row.label, pos=(row.x, row.y))
    # load edge dataset from csv
    path = os.path.join(datapath, 'topo/{}_edge.csv'.format(dataset))
    df = pd.read_csv(path, delimiter=' ')
    # add weight, capacity, delay to edge attributes
    for _, row in df.iterrows():
        i = row.src
        j = row.dest
        G.add_edge(i, j, weight=row.weight,
                   capacity=row.bw,
                   delay=row.delay)
    return G


graph = load_network_topology(DATASET, DATAPATH)
nEdges = graph.number_of_edges()

edges_list = list(graph.edges)
adjmx = np.zeros(shape=(nEdges, nEdges), dtype=np.int)
for i in range(len(edges_list)):
    u, v = edges_list[i]
    for j in range(len(edges_list)):
        u_, v_ = edges_list[j]
        if v == u_:
            adjmx[i, j] = 1

adjmx = adjmx + np.eye(nEdges)
savemat(os.path.join(TOPOPATH, 'adjmx_{}.mat'.format(DATASET)), {'A': adjmx})
