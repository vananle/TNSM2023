import os

import numpy as np
import tensorflow as tf
import tqdm
from joblib import delayed, Parallel

from .ls2sr import LS2SRSolver
from .max_step_sr import MaxStepSRSolver
from .multi_step_sr import MultiStepSRSolver
from .oblivious_routing import ObliviousRoutingSolver
from .one_step_sr import OneStepSRSolver
from .shortest_path_routing import SPSolver
from .srls import SRLS
from .te_util import load_network_topology, compute_path, createGraph_srls, extract_results, \
    get_route_changes_heuristic, get_route_changes_optimal


def p0_optimal_solver(solver, tms, gt_tms, num_node):
    gt_tms = gt_tms.reshape((-1, num_node, num_node))
    gt_tms[gt_tms <= 0.0] = 0.0
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(num_node))

    u = []
    solutions = []
    for i in range(gt_tms.shape[0]):
        try:
            solver.solve(gt_tms[i])
        except:
            print('ERROR!!!!')
            pass

        solution = solver.solution
        solutions.append(solution)
        u.append(solver.evaluate(gt_tms[i], solution))

    solutions = np.stack(solutions, axis=0)
    return u, solutions


def p2_heuristic_solver(solver, tms, gt_tms, num_node, p_solution=None):
    u = []
    tms = tms.reshape((-1, num_node, num_node))
    gt_tms = gt_tms.reshape((-1, num_node, num_node))

    tms[tms <= 0.0] = 0.0
    gt_tms[gt_tms <= 0.0] = 0.0

    tms[:] = tms[:] * (1.0 - np.eye(num_node))
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(num_node))
    tm = tms.reshape((num_node, num_node))

    solution = solver.solve(tm=tm, solution=p_solution)  # solve backtrack solution (line 131)
    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(tm=gt_tms[i], solution=solution))
    return u, solution


def p1_optimal_solver(solver, tms, gt_tms, num_node):
    tms, gt_tms = prepare_traffic_data(tms, gt_tms, num_node)

    return run_optimal_mssr(solver, tms, gt_tms)


def p3_optimal_solver(solver, tms, gt_tms, num_node):
    tms, gt_tms = prepare_traffic_data(tms, gt_tms, num_node)

    return run_optimal_mssr(solver, tms, gt_tms)


def p2_optimal_solver(solver, tms, gt_tms, num_node):
    tms, gt_tms = prepare_traffic_data(tms, gt_tms, num_node)
    tms = tms.reshape((num_node, num_node))

    return run_optimal_mssr(solver, tms, gt_tms)


def prepare_traffic_data(tms, gt_tms, num_node):
    tms = tms.reshape((-1, num_node, num_node))
    gt_tms = gt_tms.reshape((-1, num_node, num_node))

    tms[tms <= 0.0] = 0.0
    gt_tms[gt_tms <= 0.0] = 0.0

    tms[:] = tms[:] * (1.0 - np.eye(num_node))
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(num_node))
    return tms, gt_tms


def run_optimal_mssr(solver, tms, gt_tms):
    u = []
    try:
        solver.solve(tms)
    except:
        solver.solution = solver.init_solution()
    solution = solver.solution
    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i], solution=solution))

    return u, solution


class TrafficEngineering:
    def __init__(self, args, data):
        self.args = args
        self.te_alg = args.te_alg

        self.graph = load_network_topology(args.dataset, data_folder=args.data_folder)
        self.solver = None
        self.segments = compute_path(self.graph, args.dataset, args.data_folder)
        self.get_solver()

        self.te_steps = len(data['test/y_gt'])
        self.monitor = args.monitor

        self.data = data

    def get_solver(self):

        timeout = self.args.timeout
        verbose = self.args.verbose

        if 'ls2sr' in self.te_alg:
            self.solver = LS2SRSolver(self.graph, timeout, verbose, args=self.args)
        elif 'srls' in self.te_alg:
            self.graph, capacity, sp = createGraph_srls(self.args.dataset, self.args.data_folder)
            self.solver = SRLS(self.graph, sp, capacity, self.segments, timeout, verbose)
        elif 'p0' in self.te_alg:
            self.solver = OneStepSRSolver(self.graph, self.segments, timeout, verbose)
        elif 'p1' in self.te_alg or 'p3' in self.te_alg:
            self.solver = MultiStepSRSolver(self.graph, self.segments, timeout, verbose)
        elif 'p2' in self.te_alg:
            self.solver = MaxStepSRSolver(self.graph, self.segments, timeout, verbose)
        elif 'sp' in self.te_alg:
            self.solver = SPSolver(self.graph)
        elif 'ob' in self.te_alg:
            self.solver = ObliviousRoutingSolver(self.graph, self.segments, timeout, verbose)
        else:
            raise NotImplementedError

    def run_te(self):
        if 'ls2sr' in self.te_alg or 'srls' in self.te_alg:
            mlu, rc = self.heuristic_solver()
        elif 'p0' in self.te_alg or 'p1' in self.te_alg or 'p2' in self.te_alg or 'p3' in self.te_alg:
            mlu, rc = self.optimal_solver()
        elif 'sp' in self.te_alg:
            mlu, rc = self.shortestpath_routing()
        elif 'ob' in self.te_alg:
            mlu, rc = self.oblivious_routing()
        else:
            raise RuntimeError('TE not found!')

        self.save_results(mlu, rc)

        self.traffic_dynamicity()

        return mlu, rc

    def heuristic_solver(self):
        args = self.args
        x_gt, y_gt = self.data['test/x_gt'], self.data['test/y_gt']
        y_hat = self.data['test/y_cs'] if 'cs' in self.args.method else self.data['test/y_hat']
        solver = self.solver
        args.num_node = self.graph.number_of_nodes()
        use_gt = args.use_gt

        mlus, routing_changes = [], []
        for run_test in range(args.nrun):

            results = []
            solution = None

            iter = tqdm.trange(self.te_steps)

            for i in iter:
                if use_gt:
                    pred_tm = np.max(y_gt[i], axis=0)
                else:
                    pred_tm = y_hat[i]
                if solution is not None:
                    p_solution = np.copy(solution)
                else:
                    p_solution = None

                u, solution = p2_heuristic_solver(solver=solver, tms=pred_tm, gt_tms=y_gt[i], num_node=args.num_node,
                                                  p_solution=p_solution)

                _solution = np.copy(solution)
                results.append((u, _solution))
                iter.set_description(f'MLU: {max(u)}')
                self.monitor.tensorboard_writer.add_scalar(f'MLU/{args.te_alg}', max(u), i)

            mlu, solution = extract_results(results)
            rc = get_route_changes_heuristic(solution)
            mlus.append(mlu)
            routing_changes.append(rc)

        mlus = np.array(mlus)
        routing_changes = np.array(routing_changes)
        print(f'Average MLU {mlus.mean()}   Average RC {routing_changes.mean()}')
        return mlus, routing_changes

    def optimal_solver(self):
        print('[+] Run traffic engineering: ', self.te_alg)
        args = self.args
        x_gt, y_gt = self.data['test/x_gt'], self.data['test/y_gt']
        solver = self.solver

        num_node = self.graph.number_of_nodes()
        if 'p0' in self.te_alg:
            results = Parallel(n_jobs=os.cpu_count() - 1)(delayed(p0_optimal_solver)(
                solver=solver,
                tms=y_gt[i],
                gt_tms=y_gt[i],
                num_node=num_node) for i in range(self.te_steps))
        elif 'p1' in self.te_alg:
            results = Parallel(n_jobs=os.cpu_count() - 1)(delayed(p1_optimal_solver)(
                solver=solver,
                tms=y_gt[i],
                gt_tms=y_gt[i],
                num_node=num_node) for i in range(self.te_steps))

        elif 'p2' in self.te_alg:
            results = Parallel(n_jobs=os.cpu_count() - 1)(delayed(p2_optimal_solver)(
                solver=solver,
                tms=np.max(y_gt[i], axis=0, keepdims=True),
                gt_tms=y_gt[i],
                num_node=num_node) for i in range(self.te_steps))

        elif 'p3' in self.te_alg:
            t_prime = int(args.input_len / args.trunk)
            results = Parallel(n_jobs=os.cpu_count() - 1)(delayed(p3_optimal_solver)(
                solver=solver,
                tms=np.stack([np.max(y_gt[i][j:j + t_prime], axis=0) for j in range(0, y_gt[i].shape[0], t_prime)]),
                gt_tms=y_gt[i],
                num_node=num_node) for i in range(self.te_steps))
        else:
            raise NotImplementedError

        MLU, solution = extract_results(results)
        solution = np.reshape(solution, newshape=(-1, num_node, num_node, num_node))
        RC = get_route_changes_optimal(solution, self.graph)

        print(f'Average MLU {MLU.mean()}   Average RC {RC.mean()}')

        return MLU, RC

    def shortestpath_routing(self):

        x_gt, y_gt = self.data['test/x_gt'], self.data['test/y_gt']

        num_node = self.args.num_node
        solver = self.solver

        mlus = []
        routing_changes = []

        iter = tqdm.trange(self.te_steps)

        for i in iter:

            mlu, rc = [], []

            gt_tms = y_gt[i]
            gt_tms = gt_tms.reshape((-1, num_node, num_node))
            gt_tms[gt_tms <= 0.0] = 0.0
            gt_tms[:] = gt_tms[:] * (1.0 - np.eye(num_node))

            for j in range(gt_tms.shape[0]):
                mlu.append(solver.evaluate(gt_tms[j]))
                rc.append(0)

            iter.set_description(f'MLU: {max(mlu)}')
            mlus.append(mlu)
            routing_changes.append(rc)

        mlus = np.array(mlus)
        routing_changes = np.array(routing_changes)
        print(f'Average MLU {mlus.mean()}   Average RC {routing_changes.mean()}')

        return mlus, routing_changes

    def oblivious_routing(self):
        print('[+] Run traffic engineering: ', self.te_alg)
        x_gt, y_gt = self.data['test/x_gt'], self.data['test/y_gt']
        y_hat = self.data['test/y_cs'] if 'cs' in self.args.method else self.data['test/y_hat']
        num_node = self.args.num_node
        solver = self.solver

        solution = solver.solve()

        mlus = []
        routing_changes = []

        iter = tqdm.trange(self.te_steps)

        for i in iter:

            mlu, rc = [], []

            gt_tms = y_gt[i]
            gt_tms = gt_tms.reshape((-1, num_node, num_node))
            gt_tms[gt_tms <= 0.0] = 0.0
            gt_tms[:] = gt_tms[:] * (1.0 - np.eye(num_node))

            for j in range(gt_tms.shape[0]):
                mlu.append(solver.evaluate(solution, gt_tms[j]))
                rc.append(0)

            iter.set_description(f'MLU: {max(mlu)}')
            mlus.append(mlu)
            routing_changes.append(rc)

        mlus = np.array(mlus)
        routing_changes = np.array(routing_changes)
        print(f'Average MLU {mlus.mean()}   Average RC {routing_changes.mean()}')

        return mlus, routing_changes

    def traffic_dynamicity(self):
        avgstd = []
        lamda = []
        x_gt, y_gt = self.data['test/x_gt'], self.data['test/y_gt']
        for i in range(self.te_steps):
            traffic = y_gt[i]
            stds = np.std(traffic, axis=0)
            avgstd.append(np.mean(stds))
            sum_maxs = np.sum(np.max(traffic, axis=0))
            max_max = np.max(traffic)
            lamda.append(sum_maxs / max_max)

        avgstd = np.array(avgstd)
        lamda = np.array(lamda)
        path = os.path.join(self.args.model_folder, f'dyn-{self.monitor.label}-{self.te_alg}-{self.args.timeout}.npz')
        results = {
            'avgstd': avgstd,
            'lamda': lamda
        }
        with open(path, 'wb') as fp:
            np.savez_compressed(fp, **results)

    def linkload2tm(self, model, y, A_var, optimizer):
        max_initial_point_iterations = 1000
        max_optimization_iterations = 1000
        z = tf.Variable(tf.random.normal(shape=(1, self.args.latent_dim), dtype=tf.float32))
        x_pred_start = model.decoder(z)[0, :, :, 0]
        y_pred_start = tf.tensordot(A_var, tf.reshape(x_pred_start, (self.args.num_node ** 2, 1)), 1)
        loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, y_pred_start)) + 0.1 * tf.norm(z) ** 2
        for iteration in range(max_initial_point_iterations):
            z_new = tf.Variable(tf.random.normal(shape=(1, self.args.latent_dim), dtype=tf.float32))
            x_pred_new = model.decoder(z_new)[0, :, :, 0]

            x_pred_new = self.data['scaler'].inverse_transform(x_pred_new)

            y_pred_new = tf.tensordot(A_var, tf.reshape(x_pred_new, (self.args.num_node ** 2, 1)), 1)
            loss_new = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, y_pred_new)) + 0.1 * tf.norm(z) ** 2
            if loss_new < loss:
                # print('found new min loss of: {} in iteration {}'.format(loss_new, iteration))
                z = z_new
                loss = loss_new

        # initialize values
        minimum_loss = np.Inf
        x_pred_best = None

        for iteration in range(max_optimization_iterations):
            with tf.GradientTape() as tape:
                # x_pred has shape [1 12 12 1] and we care only for [12 12]
                x_pred = model.decoder(z)[0, :, :, 0]
                # find link counts (y_pred) that correspond to predicted TM (x_pred)
                y_pred = tf.tensordot(A_var, tf.reshape(x_pred, (self.args.num_node ** 2, 1)), 1)
                # compute the loss between true link counts (y) and predicted link counts (y_pred)
                loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, y_pred)) + 0.1 * tf.norm(z) ** 2
                # find the root mean square error between true TM (X_test) and predicted TM (x_pred)
                # rmse = tf.math.sqrt(
                #     tf.reduce_mean(tf.keras.losses.mean_squared_error(100 * X_test[i, :, :, 0], 100 * x_pred)))
                # # keep TM with the smallest rmse
                if tf.math.less(loss, minimum_loss):
                    x_pred_best = x_pred
                    minimum_loss = loss
            # get gradient of loss with respect to z
            grads = tape.gradient(loss, z)
            # update the value of z
            optimizer.apply_gradients(zip([grads], [z]))

        return np.array(x_pred_best)

    def vae_ls2sr(self, vae):
        print('vae_ls2sr solver')

        y_gt = self.data['test/y_gt']

        optimizer = tf.keras.optimizers.Adam()
        solution = self.solver.init_solution()  # shortest path solution

        mlus, routing_changes = [], []
        for run_test in range(self.args.nrun):
            results = []

            for i in tqdm.trange(y_gt.shape[0]):

                tm = y_gt[i, 0]
                tm = tm.reshape((self.args.num_node, self.args.num_node))
                tm[tm <= 0.0] = 0.0
                tm[:] = tm[:] * (1.0 - np.eye(self.args.num_node))

                linkload, routingMatrix = self.solver.getLinkload(solution=solution,
                                                                  tm=tm)
                A = routingMatrix
                A_var = tf.Variable(A, dtype=tf.float32)

                tm = self.linkload2tm(model=vae, y=linkload, A_var=A_var, optimizer=optimizer)
                u, solution = p2_heuristic_solver(solver=self.solver, tms=tm, gt_tms=y_gt[i], num_node=self.args.num_node,
                                                  p_solution=None)
                _solution = np.copy(solution)
                results.append((u, _solution))

            mlu, solution = extract_results(results)
            rc = get_route_changes_heuristic(solution)
            mlus.append(mlu)
            routing_changes.append(rc)

        mlus = np.array(mlus)
        routing_changes = np.array(routing_changes)
        print(f'Average MLU {mlus.mean()}   Average RC {routing_changes.mean()}')
        self.save_results(mlus, routing_changes)

        return mlus, routing_changes

    def save_results(self, mlu, rc):
        path = os.path.join(self.args.model_folder, f'te-{self.monitor.label}-{self.te_alg}-'
                                                    f'{self.args.use_gt}-{self.args.timeout}.npz')
        results = {
            'mlu': mlu,
            'rc': rc
        }
        with open(path, 'wb') as fp:
            np.savez_compressed(fp, **results)
