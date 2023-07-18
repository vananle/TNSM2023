import itertools

import numpy as np
import pulp as pl


class MSSRCFR_Solver:

    def __init__(self, G, segments):
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
                x[self.flatten_index(f, k)] * tm[f] * self.g(flow_idx[f], k, u, v)
                for f, k in
                itertools.product(range(self.nflows), range(self.num_node)))
            problem += load <= theta * capacity

        # 3) constraint function
        # ensure all traffic are routed
        for f in range(self.nflows):
            problem += pl.lpSum(x[self.flatten_index(f, k)] for k in range(self.num_node)) == 1.0

        return problem, x

    def flatten_index(self, f, k):
        """
        f: flow_id
        k: intermediate node
        """
        return f * self.num_node + k

    def g(self, flow_idx, k, u, v):
        src, dst = flow_idx
        return util.g(self.segments, src, dst, k, u, v)

    def extract_solution(self, problem):
        # extract solution
        self.var_dict = {}
        for v in problem.variables():
            self.var_dict[v.name] = v.varValue

        # self.solution = np.empty([self.num_node, self.num_node, self.num_node])
        for f, k in itertools.product(range(self.nflows), range(self.num_node)):
            index = self.flatten_index(f, k)
            src, dst = self.flow_idx[f]
            self.solution[src, dst, k] = self.var_dict['x_{}'.format(index)]

    def evaluate(self, solution, tm):
        # extract utilization
        mlu = 0
        for u, v in self.G.edges:
            load = sum([solution[i, j, k] * tm[i, j] * util.g(self.segments, i, j, k, u, v) for i, j, k in
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

    def p_routing(self, rTm, pSolution):
        """
        Obtain remaining capacity of each link after routing tm as pSolution
        """
        rCapa = {}

        for u, v in self.G.edges:
            load = sum([pSolution[i, j, k] * rTm[i, j] * util.g(self.segments, i, j, k, u, v) for i, j, k in
                        itertools.product(range(self.num_node), range(self.num_node), range(self.num_node))])
            capacity = self.G.get_edge_data(u, v)['capacity']
            if load <= capacity:
                rCapa[u, v] = capacity - load
            else:
                rCapa[u, v] = 0
        return rCapa

    def solve(self, tm, rTm, flow_idx, pSolution):
        """
        tm: traffic matrix for solving
        rTM: remaining traffic matrix with critical flow = 0  (nodes, nodes)
        flow_idx: (src, dst) of critical flow
        pSolution: previous solution (use initial solution if no previous solution)
        """
        rCapa = self.p_routing(rTm, pSolution)
        problem, x = self.create_problem(tm, flow_idx, rCapa)

        self.solution = np.copy(pSolution)
        # stime = time.time()
        problem.solve()
        # print('solving time: ', time.time() - stime)
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
            path += util.shortest_path(self.G, i, k)[:-1]
            path += util.shortest_path(self.G, k, j)
            paths.append((k, path))
        return paths
