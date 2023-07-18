import itertools

import numpy as np
import pulp as pl
from pulp.apis.coin_api import PULP_CBC_CMD

from .solver import Solver
from .te_util import edge_in_segment, flatten_index, shortest_path


def flatten_index_edge(i, j, num_edge):
    return i * num_edge + j


class ObliviousRoutingSolver(Solver):

    def __init__(self, graph, segments, timeout, verbose):
        super(ObliviousRoutingSolver, self).__init__(graph, timeout, verbose)
        self.segments = segments
        self.num_tms = 0
        self.problem = None
        self.var_dict = None
        self.solution = None
        self.status = None
        self.solver = PULP_CBC_CMD(timeLimit=timeout, msg=False)

    def create_problem(self):

        # 0) initialize lookup dictionary from index i to edge u, v
        edges_dictionary = {}
        for i, (u, v) in enumerate(self.G.edges):
            edges_dictionary[i] = (u, v)

        # 1) create optimization model of dual problem
        problem = pl.LpProblem('SegmentRouting', pl.LpMinimize)
        theta = pl.LpVariable(name='theta', lowBound=0.0, cat='Continuous')
        x = pl.LpVariable.dicts(name='x', indexs=np.arange(self.num_node ** 3), cat='Binary')
        pi = pl.LpVariable.dicts(name='pi', indexs=np.arange(self.num_edge ** 2), lowBound=0.0)

        # 2) objective function
        # minimize maximum link utilization
        problem += theta

        # 3) constraint function 2
        for i, j in itertools.product(range(self.num_node), range(self.num_node)):  # forall ij

            for e_prime in edges_dictionary:  # forall e' = [u, v]
                u, v = edges_dictionary[e_prime]
                # sum(g_ijk(e'))*alpha_ijk
                lb = pl.lpSum(
                    [edge_in_segment(self.segments, i, j, k, u, v) * x[flatten_index(i, j, k, self.num_node)] for k in
                     range(self.num_node)])
                for m in range(self.num_node):  # forall m
                    # sum(g_ijm(e) * pi(e,e)') >= sum(g_ijk(e')) * alpha_ijk
                    problem += pl.lpSum([edge_in_segment(self.segments, i, j, m, edges_dictionary[e][0],
                                                         edges_dictionary[e][1])
                                         * pi[flatten_index_edge(e, e_prime, self.num_edge)] for e in
                                         edges_dictionary]) >= lb

        # 4) constraint function 3
        for e_prime in edges_dictionary:  # for edge e'   sum(c(e) * pi(e, e')) <= theta * c(e')
            u, v = edges_dictionary[e_prime]
            capacity_e_prime = self.G.get_edge_data(u, v)['capacity']
            problem += pl.lpSum([self.G.get_edge_data(edges_dictionary[e][0], edges_dictionary[e][1])['capacity'] *
                                 pi[flatten_index_edge(e, e_prime, self.num_edge)] for e in edges_dictionary]) \
                       <= theta * capacity_e_prime

        # 3) constraint function 4
        for i, j in itertools.product(range(self.num_node),
                                      range(self.num_node)):  # forall ij:   sunm(alpha_ijk) == 1.0
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

    def evaluate(self, solution, tm):
        # extract utilization
        mlu = []
        for u, v in self.G.edges:
            load = sum([solution[i, j, k] * tm[i, j] * edge_in_segment(self.segments, i, j, k, u, v) for i, j, k in
                        itertools.product(range(self.num_node), range(self.num_node), range(self.num_node))])
            capacity = self.G.get_edge_data(u, v)['capacity']
            utilization = load / capacity
            mlu.append(utilization)

        mlu = np.array(mlu)
        return np.max(mlu)

    def extract_status(self, problem):
        self.status = pl.LpStatus[problem.status]

    def init_solution(self):
        self.solution = np.zeros([self.num_node, self.num_node, self.num_node])
        for i, j in itertools.product(range(self.num_node), range(self.num_node)):
            self.solution[i, j, i] = 1

    def solve(self, tm=None, solution=None, eps=1e-12):
        problem, x = self.create_problem()
        self.init_solution()
        problem.solve(solver=self.solver)
        self.problem = problem
        self.extract_status(problem)
        self.extract_solution(problem)
        return self.solution

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
