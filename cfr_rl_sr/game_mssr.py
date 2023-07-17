from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pulp import LpProblem, LpStatus, lpSum, LpVariable, GLPK

from mssr_cfr import MSSRCFR_Solver
from one_step_sr import OneStepSRSolver

OBJ_EPSILON = 1e-12


class Game(object):
    def __init__(self, args, env):
        self.random_state = np.random.RandomState(seed=args.seed)

        self.data_folder = env.data_folder
        self.graph = env.topology.graph
        # self.traffic_file = env.traffic_file
        self.traffic_matrices = env.traffic_matrices
        self.traffic_matrices_dims = self.traffic_matrices.shape
        self.tm_cnt = env.tm_cnt
        self.num_pairs = env.num_pairs
        self.pair_idx_to_sd = env.pair_idx_to_sd
        self.pair_sd_to_idx = env.pair_sd_to_idx
        self.num_nodes = env.num_nodes
        self.num_links = env.num_links
        self.link_idx_to_sd = env.link_idx_to_sd
        self.link_sd_to_idx = env.link_sd_to_idx
        self.link_capacities = env.link_capacities
        self.link_weights = env.link_weights
        self.shortest_paths_node = env.shortest_paths_node  # paths with node info
        self.shortest_paths_link = env.shortest_paths_link  # paths with link info

        self.ecmp_next_hops = {}
        self.get_ecmp_next_hops()

        self.model_type = args.model_type

        # for LP
        self.lp_pairs = [p for p in range(self.num_pairs)]
        self.lp_nodes = [n for n in range(self.num_nodes)]
        self.links = [e for e in range(self.num_links)]
        self.lp_links = [e for e in self.link_sd_to_idx]
        self.pair_links = [(pr, e[0], e[1]) for pr in self.lp_pairs for e in self.lp_links]
        self.timeout = args.timeout
        self.load_multiplier = {}
        self.normalized_traffic_matrices = None

    def generate_inputs(self, normalization=True):
        self.normalized_traffic_matrices = np.zeros(
            (self.valid_tm_cnt, self.traffic_matrices_dims[1], self.traffic_matrices_dims[2], self.tm_history),
            dtype=np.float32)  # tm state  [Valid_tms, Node, Node, History]
        idx_offset = self.tm_history - 1
        for tm_idx in self.tm_indexes:
            for h in range(self.tm_history):
                if normalization:
                    tm_max_element = np.max(self.traffic_matrices[tm_idx - h])
                    if tm_max_element > 0:
                        self.normalized_traffic_matrices[tm_idx - idx_offset, :, :, h] = \
                            self.traffic_matrices[tm_idx - h] / tm_max_element  # [Valid_tms, Node, Node, History]
                    else:
                        self.normalized_traffic_matrices[tm_idx - idx_offset, :, :, h] = self.traffic_matrices[
                            tm_idx - h]  # [Valid_tms, Node, Node, History]

                else:
                    self.normalized_traffic_matrices[tm_idx - idx_offset, :, :, h] = self.traffic_matrices[
                        tm_idx - h]  # [Valid_tms, Node, Node, History]

    def get_topK_flows(self, tm_idx, pairs):
        tm = self.traffic_matrices[tm_idx]
        f = {}
        for p in pairs:
            s, d = self.pair_idx_to_sd[p]
            f[p] = tm[s][d]

        sorted_f = sorted(f.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

        cf = []
        for i in range(self.max_moves):
            cf.append(sorted_f[i][0])

        return cf

    def get_ecmp_next_hops(self):
        self.ecmp_next_hops = {}
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src == dst:
                    continue
                self.ecmp_next_hops[src, dst] = []
                for p in self.shortest_paths_node[self.pair_sd_to_idx[(src, dst)]]:
                    if p[1] not in self.ecmp_next_hops[src, dst]:
                        self.ecmp_next_hops[src, dst].append(p[1])

    def ecmp_next_hop_distribution(self, link_loads, demand, src, dst):
        if src == dst:
            return

        ecmp_next_hops = self.ecmp_next_hops[src, dst]

        next_hops_cnt = len(ecmp_next_hops)
        # if next_hops_cnt > 1:
        # print(self.shortest_paths_node[self.pair_sd_to_idx[(src, dst)]])

        ecmp_demand = demand / next_hops_cnt
        for np in ecmp_next_hops:
            link_loads[self.link_sd_to_idx[(src, np)]] += ecmp_demand
            self.ecmp_next_hop_distribution(link_loads, ecmp_demand, np, dst)

    def ecmp_traffic_distribution(self, tm_idx):
        link_loads = np.zeros(self.num_links)
        tm = self.traffic_matrices[tm_idx]
        for pair_idx in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[pair_idx]
            demand = tm[s][d]
            if demand != 0:
                self.ecmp_next_hop_distribution(link_loads, demand, s, d)

        return link_loads

    def get_critical_topK_flows(self, solver, tm_idx, pSolution, critical_links=5):

        potential = solver.get_cf_potential(solution=pSolution, tm=self.traffic_matrices[tm_idx], critical_links=5)

        cf_potential = []
        for pairs in potential:
            cf_potential.append(self.pair_sd_to_idx[pairs])

        # print(cf_potential)
        assert len(cf_potential) >= self.max_moves, \
            ("cf_potential(%d) < max_move(%d), please increase critical_links(%d)" % (
                cf_potential, self.max_moves, critical_links))

        return self.get_topK_flows(tm_idx, cf_potential)

    def eval_ecmp_traffic_distribution(self, tm_idx, eval_delay=False):
        eval_link_loads = self.ecmp_traffic_distribution(tm_idx)
        eval_max_utilization = np.max(eval_link_loads / self.link_capacities)
        self.load_multiplier[tm_idx] = 0.9 / eval_max_utilization
        delay = 0
        if eval_delay:
            eval_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(eval_link_loads / (self.link_capacities - eval_link_loads))

        return eval_max_utilization, delay

    def optimal_routing_mlu(self, solver, tm_idx, pSolution):
        tm = self.traffic_matrices[tm_idx]

        solver.solve(tm)
        solution = solver.solution
        obj_r = solver.evaluate(tm, solution)
        if obj_r == 0.0:
            solution = pSolution
            obj_r = solver.evaluate(solution=solution, tm=tm)

        return obj_r, solution

    def eval_optimal_routing_mlu(self, tm_idx, solution, eval_delay=False):
        optimal_link_loads = np.zeros(self.num_links)
        eval_tm = self.traffic_matrices[tm_idx]
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demand = eval_tm[s][d]
            for e in self.lp_links:
                link_idx = self.link_sd_to_idx[e]
                optimal_link_loads[link_idx] += demand * solution[i, e[0], e[1]]

        optimal_max_utilization = np.max(optimal_link_loads / self.link_capacities)
        delay = 0
        if eval_delay:
            assert tm_idx in self.load_multiplier, tm_idx
            optimal_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(optimal_link_loads / (self.link_capacities - optimal_link_loads))

        return optimal_max_utilization, delay

    @staticmethod
    def flowidx2srcdst(flow_idx, nNodes):
        src = flow_idx / nNodes
        src = src.astype(np.int)

        dst = flow_idx % nNodes
        dst = dst.astype(np.int)

        srcdst_idx = np.stack([src, dst], axis=1)
        return srcdst_idx

    def optimal_routing_mlu_critical_pairs(self, solver, tm_idx, critical_pairs, pSolution, init_routing=False):
        tm = self.traffic_matrices[tm_idx]
        srcdst_pairs = []
        rTm = np.copy(tm)

        demands = []
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            # background link load
            if i in critical_pairs:
                srcdst_pairs.append([s, d])
                demands.append(tm[s][d])
                rTm[s, d] = 0

        demands = np.asarray(demands)
        srcdst_pairs = np.asarray(srcdst_pairs)

        solution = solver.solve(tm=demands, rTm=rTm, flow_idx=srcdst_pairs, pSolution=pSolution,
                                init_routing=init_routing)
        obj_r = solver.evaluate(solution=solution, tm=tm)

        if obj_r == 0.0:
            solution = pSolution
            obj_r = solver.evaluate(solution=solution, tm=tm)
            use_pSolution = True
        else:
            use_pSolution = False
        return obj_r, solution, use_pSolution

    def eval_critical_flow_and_ecmp(self, tm_idx, critical_pairs, solution, eval_delay=False):
        eval_tm = self.traffic_matrices[tm_idx]
        eval_link_loads = np.zeros(self.num_links)
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            if i not in critical_pairs:
                self.ecmp_next_hop_distribution(eval_link_loads, eval_tm[s][d], s, d)
            else:
                for e in self.lp_links:
                    link_idx = self.link_sd_to_idx[e]
                    eval_link_loads[link_idx] += eval_tm[s][d] * solution[i, e[0], e[1]]

        eval_max_utilization = np.max(eval_link_loads / self.link_capacities)
        delay = 0
        if eval_delay:
            assert tm_idx in self.load_multiplier, tm_idx
            eval_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(eval_link_loads / (self.link_capacities - eval_link_loads))

        return eval_max_utilization, delay

    def optimal_routing_delay(self, tm_idx):
        assert tm_idx in self.load_multiplier, tm_idx
        tm = self.traffic_matrices[tm_idx] * self.load_multiplier[tm_idx]
        demands = {}
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demands[i] = tm[s][d]

        model = LpProblem(name="routing")

        # ratio = LpVariable.dicts(name="ratio", indexs=self.pair_links, cat='Binary')
        ratio = LpVariable.dicts(name="ratio", indexs=self.pair_links, lowBound=0, upBound=1)

        link_load = LpVariable.dicts(name="link_load", indexs=self.links)

        f = LpVariable.dicts(name="link_cost", indexs=self.links)

        for pr in self.lp_pairs:
            model += (
                lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][0]]) - lpSum(
                    [ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][0]]) == -1,
                "flow_conservation_constr1_%d" % pr)

        for pr in self.lp_pairs:
            model += (
                lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][1]]) - lpSum(
                    [ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][1]]) == 1,
                "flow_conservation_constr2_%d" % pr)

        for pr in self.lp_pairs:
            for n in self.lp_nodes:
                if n not in self.pair_idx_to_sd[pr]:
                    model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == n]) - lpSum(
                        [ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == n]) == 0,
                              "flow_conservation_constr3_%d_%d" % (pr, n))

        for e in self.lp_links:
            ei = self.link_sd_to_idx[e]
            model += (link_load[ei] == lpSum([demands[pr] * ratio[pr, e[0], e[1]] for pr in self.lp_pairs]),
                      "link_load_constr%d" % ei)
            model += (f[ei] * self.link_capacities[ei] >= link_load[ei], "cost_constr1_%d" % ei)
            model += (f[ei] >= 3 * link_load[ei] / self.link_capacities[ei] - 2 / 3, "cost_constr2_%d" % ei)
            model += (f[ei] >= 10 * link_load[ei] / self.link_capacities[ei] - 16 / 3, "cost_constr3_%d" % ei)
            model += (f[ei] >= 70 * link_load[ei] / self.link_capacities[ei] - 178 / 3, "cost_constr4_%d" % ei)
            model += (f[ei] >= 500 * link_load[ei] / self.link_capacities[ei] - 1468 / 3, "cost_constr5_%d" % ei)
            model += (f[ei] >= 5000 * link_load[ei] / self.link_capacities[ei] - 16318 / 3, "cost_constr6_%d" % ei)

        model += lpSum(f[ei] for ei in self.links)

        model.solve(solver=GLPK(msg=False))
        assert LpStatus[model.status] == 'Optimal'

        solution = {}
        for k in ratio:
            solution[k] = ratio[k].value()

        return solution

    def eval_optimal_routing_delay(self, tm_idx, solution):
        optimal_link_loads = np.zeros(self.num_links)
        assert tm_idx in self.load_multiplier, tm_idx
        eval_tm = self.traffic_matrices[tm_idx] * self.load_multiplier[tm_idx]
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demand = eval_tm[s][d]
            for e in self.lp_links:
                link_idx = self.link_sd_to_idx[e]
                optimal_link_loads[link_idx] += demand * solution[i, e[0], e[1]]

        optimal_delay = sum(optimal_link_loads / (self.link_capacities - optimal_link_loads))

        return optimal_delay


class CFRRL_Game(Game):
    def __init__(self, args, env):
        super(CFRRL_Game, self).__init__(args, env)

        self.project_name = args.project_name
        self.action_dim = env.num_pairs
        self.max_moves = int(self.action_dim * (args.max_moves / 100.))
        assert self.max_moves <= self.action_dim, (self.max_moves, self.action_dim)

        self.tm_history = 1
        self.tm_indexes = np.arange(self.tm_history - 1, self.tm_cnt)
        self.valid_tm_cnt = len(self.tm_indexes)

        if args.method == 'pure_policy':
            self.baseline = {}

        self.generate_inputs(normalization=True)
        self.state_dims = self.normalized_traffic_matrices.shape[1:]
        print('Input dims :', self.state_dims)
        print('Max moves :', self.max_moves)

        self.mssr_cfr_solver = MSSRCFR_Solver(env.topology.graph, env.topology.segment, args)
        self.optimal_solver = OneStepSRSolver(env.topology.graph, env.topology.segment, args)

    def get_state(self, tm_idx):
        idx_offset = self.tm_history - 1
        return self.normalized_traffic_matrices[tm_idx - idx_offset]

    def reward(self, tm_idx, actions):
        pSolution = self.mssr_cfr_solver.initialize()
        mlu, _, _ = self.optimal_routing_mlu_critical_pairs(self.mssr_cfr_solver, tm_idx, actions, pSolution,
                                                            init_routing=True)

        if mlu <= 0:
            mlu = 10e-6
        reward = 1 / mlu

        return reward

    def advantage(self, tm_idx, reward):
        if tm_idx not in self.baseline:
            return reward

        total_v, cnt = self.baseline[tm_idx]

        # print(reward, (total_v/cnt))

        return reward - (total_v / cnt)

    def update_baseline(self, tm_idx, reward):
        if tm_idx in self.baseline:
            total_v, cnt = self.baseline[tm_idx]

            total_v += reward
            cnt += 1

            self.baseline[tm_idx] = (total_v, cnt)
        else:
            self.baseline[tm_idx] = (reward, 1)

    # def dic2array(self, solution):
    #     n_crit_pairs = int(len(solution)/ self.num_links)
    #     asolution = np.zeros(shape=(n_crit_pairs, self.num_links, self.num_links), dtype=np.float_)
    #     for k, e1, e2, v in solution:
    #         asolution[k, e1, e2] = v
    #     return asolution

    def evaluate_cfr_rl(self, tm_idx, pSolution, actions=None):
        mlu, solution, use_pSolution = self.optimal_routing_mlu_critical_pairs(self.mssr_cfr_solver, tm_idx, actions,
                                                                               pSolution=pSolution)
        return mlu, solution, use_pSolution

    def evaluate_crit_topk(self, tm_idx, pSolution):
        crit_topk = self.get_critical_topK_flows(self.mssr_cfr_solver, tm_idx, pSolution)
        crit_mlu, crit_topk_solution, use_pSolution = self.optimal_routing_mlu_critical_pairs(self.mssr_cfr_solver,
                                                                                              tm_idx, crit_topk,
                                                                                              pSolution=pSolution)

        return crit_mlu, crit_topk_solution, use_pSolution

    def evaluate_topk(self, tm_idx, pSolution):
        topk = self.get_topK_flows(tm_idx, self.lp_pairs)
        topk_mlu, topk_solution, use_pSolution = self.optimal_routing_mlu_critical_pairs(self.mssr_cfr_solver, tm_idx,
                                                                                         topk,
                                                                                         pSolution=pSolution)
        return topk_mlu, topk_solution, use_pSolution

    def evaluate_optimal(self, tm_idx, pSolution):
        optimal_mlu, optimal_solution = self.optimal_routing_mlu(self.optimal_solver, tm_idx, pSolution)
        return optimal_mlu, optimal_solution

    def evaluate_sp(self, tm_idx):
        sp_solution = self.mssr_cfr_solver.initialize()
        sp_mlu = self.mssr_cfr_solver.evaluate(tm=self.traffic_matrices[tm_idx], solution=sp_solution)

        return sp_mlu
