import itertools

import networkx as nx
import numpy as np
import pulp as pl
from pulp.apis.coin_api import PULP_CBC_CMD


def flatten_index(i, j, k, num_node):
    return i * num_node ** 2 + j * num_node + k


def shortest_path(graph, source, target):
    return nx.shortest_path(graph, source=source, target=target, weight='weight')


def edge_in_segment(segments, i, j, k, u, v):
    if len(segments[i, j]) == 0:
        return 0
    elif len(segments[i, j][k]) == 0:
        return 0

    value = 0
    if len(segments[i, j][k][0]) != 0 and (u, v) in segments[i, j][k][0]:
        value += 1

    if len(segments[i, j][k][1]) != 0 and (u, v) in segments[i, j][k][1]:
        value += 1

    return value


class OneStepSRSolver:

    def __init__(self, G, segments, args):
        """
        G: networkx Digraph, a network topology
        """
        self.G = G
        self.num_node = G.number_of_nodes()
        self.segments = segments
        self.problem = None
        self.var_dict = None
        self.solution = None
        self.status = None
        self.solver = PULP_CBC_CMD(timeLimit=args.timeout, msg=False)

    def create_problem(self, tm):
        # 1) create optimization model
        problem = pl.LpProblem('SegmentRouting', pl.LpMinimize)
        theta = pl.LpVariable(name='theta', lowBound=0.0, cat='Continuous')

        x = pl.LpVariable.dicts(name='x',
                                indexs=np.arange(self.num_node ** 3),
                                cat='Binary')

        # 2) objective function
        # minimize maximum link utilization
        problem += theta

        # 3) constraint function
        for u, v in self.G.edges:
            capacity = self.G.get_edge_data(u, v)['capacity']
            load = pl.lpSum(
                x[flatten_index(i, j, k, self.num_node)] * tm[i, j] * edge_in_segment(self.segments, i, j, k, u, v)
                for i, j, k in
                itertools.product(range(self.num_node), range(self.num_node), range(self.num_node)))
            problem += load <= theta * capacity

        # 3) constraint function
        # ensure all traffic are routed
        for i, j in itertools.product(range(self.num_node), range(self.num_node)):
            problem += pl.lpSum(x[flatten_index(i, j, k, self.num_node)] for k in range(self.num_node)) == 1

        return problem, x

    def extract_solution(self, problem):
        # extract solution
        self.var_dict = {}
        for v in problem.variables():
            self.var_dict[v.name] = v.varValue

        self.solution = np.empty([self.num_node, self.num_node, self.num_node])
        for i, j, k in itertools.product(range(self.num_node), range(self.num_node), range(self.num_node)):
            index = flatten_index(i, j, k, self.num_node)
            self.solution[i, j, k] = self.var_dict['x_{}'.format(index)]

    def evaluate(self, tm, solution):
        # extract utilization
        mlu = 0
        for u, v in self.G.edges:
            load = sum([solution[i, j, k] * tm[i, j] * edge_in_segment(self.segments, i, j, k, u, v) for i, j, k in
                        itertools.product(range(self.num_node), range(self.num_node), range(self.num_node))])
            capacity = self.G.get_edge_data(u, v)['capacity']
            utilization = load / capacity
            self.G[u][v]['utilization'] = utilization
            if utilization >= mlu:
                mlu = utilization
        return mlu

    def extract_status(self, problem):
        self.status = pl.LpStatus[problem.status]

    def initialize(self):
        solution = np.zeros([self.num_node, self.num_node, self.num_node])
        for i, j in itertools.product(range(self.num_node), range(self.num_node)):
            solution[i, j, i] = 1

        return solution

    def solve(self, tm):
        problem, x = self.create_problem(tm)
        self.solution = self.initialize()
        problem.solve(pl.GLPK(msg=False, timeLimit=1))
        self.problem = problem
        self.extract_status(problem)
        self.extract_solution(problem)

    def get_paths(self, i, j):
        if i == j:
            list_k = [i]
        else:
            list_k = np.where(self.solution[i, j] > 0)[0]
        paths = []
        for k in list_k:
            path = []
            path += shortest_path(self.G, i, k)[:-1]
            path += shortest_path(self.G, k, j)
            paths.append((k, path))
        return paths
