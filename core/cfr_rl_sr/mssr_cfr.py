import itertools

import networkx as nx
import numpy as np
import pulp as pl
from pulp.apis.coin_api import PULP_CBC_CMD


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


def shortest_path(graph, source, target):
    return nx.shortest_path(graph, source=source, target=target, weight='weight')


class MSSRCFR_Solver:

    def __init__(self, G, segments, args):
        """
        nflows
        G: networkx Digraph, a network topology
        """
        self.flow_idx = None
        self.G = G
        self.num_node = G.number_of_nodes()
        self.nflows = 0
        self.segments = segments
        self.problem = None
        self.var_dict = None
        self.solution = None
        self.solution = self.initialize()
        self.status = None
        self.solver = PULP_CBC_CMD(timeLimit=args.timeout, msg=False)

    def create_problem(self, tm, flow_idx, rCapa):
        self.flow_idx = flow_idx
        self.nflows = flow_idx.shape[0]

        # 1) create optimization model
        problem = pl.LpProblem('SegmentRouting', pl.LpMinimize)
        theta = pl.LpVariable(name='theta', lowBound=0.0, cat='Continuous')

        x = pl.LpVariable.dicts(name='x',
                                indexs=np.arange(self.nflows * self.num_node),
                                cat='Binary')

        # 2) objective function
        # minimize maximum link utilization
        problem += theta

        # 3) constraint function
        for u, v in self.G.edges:
            capacity = rCapa[u, v]
            load = pl.lpSum(
                x[f * self.num_node + k] * tm[f] * self.edge_in_segment(flow_idx[f], k, u, v)
                for f, k in
                itertools.product(range(self.nflows), range(self.num_node)))
            problem += load <= theta * capacity

        # 3) constraint function
        # ensure all traffic are routed
        for f in range(self.nflows):
            problem += pl.lpSum(x[f * self.num_node + k] for k in range(self.num_node)) == 1.0

        return problem, x

    def edge_in_segment(self, flow_idx, k, u, v):
        src, dst = flow_idx
        return edge_in_segment(self.segments, src, dst, k, u, v)

    def extract_solution(self, problem):
        # extract solution
        self.var_dict = {}
        for v in problem.variables():
            self.var_dict[v.name] = v.varValue

        # self.solution = np.empty([self.num_node, self.num_node, self.num_node])
        for f, k in itertools.product(range(self.nflows), range(self.num_node)):
            src, dst = self.flow_idx[f]
            self.solution[src, dst, k] = self.var_dict['x_{}'.format(f * self.num_node + k)]

    def evaluate(self, solution, tm):
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

    def get_cf_potential(self, solution, tm, critical_links=5):
        # extract utilization
        linkutils = {}
        utils = []
        for u, v in self.G.edges:
            load = sum([solution[i, j, k] * tm[i, j] * edge_in_segment(self.segments, i, j, k, u, v) for i, j, k in
                        itertools.product(range(self.num_node), range(self.num_node), range(self.num_node))])
            capacity = self.G.get_edge_data(u, v)['capacity']
            utilization = load / capacity
            utils.append(utilization)
            linkutils[u, v] = utilization

        utils = np.asarray(utils)
        critical_link_indexes = np.argsort(-(utils))[:critical_links]
        keys = list(linkutils)

        cf_potential = []
        for i, j in itertools.product(range(self.num_node), range(self.num_node)):
            for crit_link_idx in critical_link_indexes:
                u, v = keys[crit_link_idx]
                for k in range(self.num_node):
                    if solution[i, j, k]:
                        if edge_in_segment(self.segments, i, j, k, u, v):
                            cf_potential.append((i, j))
                            break

                if (i, j) in cf_potential:
                    break

        return cf_potential

    def extract_status(self, problem):
        self.status = pl.LpStatus[problem.status]

    def initialize(self):
        solution = np.zeros([self.num_node, self.num_node, self.num_node])
        for i, j in itertools.product(range(self.num_node), range(self.num_node)):
            solution[i, j, i] = 1

        return solution

    def p_routing(self, rTm, pSolution):
        """
        Obtain remaining capacity of each link after routing tm as pSolution
        """
        rCapa = {}

        for u, v in self.G.edges:
            load = sum([pSolution[i, j, k] * rTm[i, j] * edge_in_segment(self.segments, i, j, k, u, v) for i, j, k in
                        itertools.product(range(self.num_node), range(self.num_node), range(self.num_node))])
            capacity = self.G.get_edge_data(u, v)['capacity']
            if load <= capacity:
                rCapa[u, v] = capacity - load
            else:
                rCapa[u, v] = 0
        return rCapa

    def init_routing(self, rTm):
        rCapa = {}

        for u, v in self.G.edges:
            load = 0
            for i, j in itertools.product(range(self.num_node), range(self.num_node)):
                load += rTm[i, j] * edge_in_segment(self.segments, i, j, i, u, v)
            capacity = self.G.get_edge_data(u, v)['capacity']
            if load <= capacity:
                rCapa[u, v] = capacity - load
            else:
                rCapa[u, v] = 0
        return rCapa

    def solve(self, tm, rTm, flow_idx, pSolution, init_routing=False):
        """
        tm: traffic matrix for solving
        rTM: remaining traffic matrix with critical flow = 0  (nodes, nodes)
        flow_idx: (src, dst) of critical flow
        pSolution: previous solution (use initial solution if no previous solution)
        """
        if init_routing:
            rCapa = self.init_routing(rTm)
        else:
            rCapa = self.p_routing(rTm, pSolution)
        self.problem, x = self.create_problem(tm, flow_idx, rCapa)
        self.solution = np.copy(pSolution)
        self.problem.solve(pl.GLPK(msg=False, timeLimit=1))
        self.extract_solution(self.problem)
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
