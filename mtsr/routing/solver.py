import numpy as np


class Solver:
    def __init__(self, graph, timeout, verbose):
        self.G = graph
        self.num_node = graph.number_of_nodes()
        self.num_edge = graph.number_of_edges()
        self.indices_edge = np.arange(self.num_edge)
        self.list_edges = list(self.G.edges)
        self.timeout = timeout
        self.verbose = verbose

    def solve(self, tm, solution=None, eps=1e-12):
        pass

    def evaluate(self, solution, tm):
        pass
