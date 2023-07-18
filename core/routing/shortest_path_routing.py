import itertools

import networkx as nx
import numpy as np

from .te_util import shortest_path


class SPSolver:

    def __init__(self, G):
        self.G = G

    def evaluate(self, tm):
        # extract parameters
        G = self.G
        num_node = G.number_of_nodes()
        # initialize link load
        for u, v in G.edges:
            G[u][v]['load'] = 0
        for i, j in itertools.product(range(num_node), range(num_node)):
            path = nx.Graph()
            path.add_nodes_from(G)
            nx.add_path(path, shortest_path(G, i, j))
            for u, v in path.edges:
                G[u][v]['load'] += tm[i, j]
        # compute link utilization
        mlu = []
        for u, v in G.edges:
            u = G[u][v]['load'] / G[u][v]['capacity']
            mlu.append(u)

        mlu = np.array(mlu)
        return np.max(mlu)

    def extract_utilization_v2(self, tm):
        self.evaluate(tm)

    def solve(self, tm, solution=None, eps=1e-12):
        self.evaluate(tm)

    def get_path(self, i, j):
        G = self.G
        path = shortest_path(G, i, j)
        return path
