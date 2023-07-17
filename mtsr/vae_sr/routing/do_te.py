import tensorflow as tf
from tqdm import tqdm

from .ls2sr import LS2SRSolver
from .srls import SRLS
from .util import *


def calculate_lamda(y_gt):
    sum_max = np.sum(np.max(y_gt, axis=1))
    maxmax = np.max(y_gt)
    return sum_max / maxmax


def get_route_changes(routings, graph):
    route_changes = np.zeros(shape=(routings.shape[0] - 1))
    for t in range(routings.shape[0] - 1):
        _route_changes = 0
        for i, j in itertools.product(range(routings.shape[1]), range(routings.shape[2])):
            path_t_1 = get_paths_from_solution(graph, routings[t + 1], i, j)
            path_t = get_paths_from_solution(graph, routings[t], i, j)
            if path_t_1 != path_t:
                _route_changes += 1

        route_changes[t] = _route_changes

    return route_changes


def get_route_changes_heuristic(routings):
    route_changes = []
    for t in range(routings.shape[0] - 1):
        route_changes.append(count_routing_change(routings[t + 1], routings[t]))

    route_changes = np.asarray(route_changes)
    return route_changes


def extract_results(results):
    mlus, solutions = [], []
    for _mlu, _solution in results:
        mlus.append(_mlu)
        solutions.append(_solution)

    mlus = np.stack(mlus, axis=0)
    solutions = np.stack(solutions, axis=0)

    return mlus, solutions


def save_results(log_dir, fname, mlus, route_change):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    np.save(os.path.join(log_dir, fname + '_mlus'), mlus)
    np.save(os.path.join(log_dir, fname + '_route_change'), route_change)


def vae_gen_data(data, graphs, args, fname):
    print('------->>> SRLS_VAE <<<-------')
    print('Dataset: {} - seq_len: {}'.format(args.dataset, args.seq_len_x))
    G, nNodes, nEdges, capacity, sp = graphs

    solver = SRLS(sp, capacity, nNodes, nEdges, args.timeout)
    LL, LM, As, TMs = [], [], [], []
    data[data < 10e-6] = 10e-5
    x_gt = data[:, :args.seq_len_x, :]
    y_gt = data[:, args.seq_len_x:, :]

    for i in tqdm(range(data.shape[0])):
        T0 = np.max(x_gt[i], axis=0)
        solver.modifierTrafficMatrix(np.reshape(T0, newshape=(nNodes, nNodes)))
        solver.solve()
        A = solver.extractRoutingPath()
        L0 = np.zeros(shape=(x_gt[i].shape[0], nEdges))
        for j in range(x_gt[i].shape[0]):
            tm = x_gt[i, j]
            tm = np.reshape(tm, newshape=(nNodes, nNodes))
            l = solver.getLinkload(routingSolution=A, trafficMatrix=tm)
            l = np.squeeze(l, axis=-1)
            L0[j] = l

        T = np.max(y_gt[i], axis=0)
        tm = np.reshape(T, newshape=(nNodes, nNodes))
        l = solver.getLinkload(routingSolution=A, trafficMatrix=tm)
        L = np.squeeze(l, axis=-1)

        LL.append(L0)
        LM.append(L)
        As.append(A)
        TMs.append(T)

    LL = np.stack(LL, axis=0)
    LM = np.stack(LM, axis=0)
    As = np.stack(As, axis=0)
    TMs = np.stack(TMs, axis=0)

    np.save(os.path.join(args.log_dir, '{}_LL'.format(args.set)), LL)
    np.save(os.path.join(args.log_dir, '{}_LM'.format(args.set)), LM)
    np.save(os.path.join(args.log_dir, '{}_A'.format(args.set)), As)
    np.save(os.path.join(args.log_dir, '{}_TM'.format(args.set)), TMs)
    print('LL shape: ', LL.shape)
    print('LM shape: ', LM.shape)
    print('As shape: ', As.shape)
    print('TMs shape: ', TMs.shape)


max_initial_point_iterations = 1000
max_optimization_iterations = 1000


def linkload2tm(model, y, A_var, optimizer, args):
    z = tf.Variable(tf.random.normal(shape=(1, args.latent_dim), dtype=tf.float32))
    x_pred_start = model.decoder(z)[0, :, :, 0]
    y_pred_start = tf.tensordot(A_var, tf.reshape(x_pred_start, (args.nNodes ** 2, 1)), 1)
    loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y, y_pred_start)) + 0.1 * tf.norm(z) ** 2
    for iteration in range(max_initial_point_iterations):
        z_new = tf.Variable(tf.random.normal(shape=(1, args.latent_dim), dtype=tf.float32))
        x_pred_new = model.decoder(z_new)[0, :, :, 0]
        y_pred_new = tf.tensordot(A_var, tf.reshape(x_pred_new, (args.nNodes ** 2, 1)), 1)
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
            y_pred = tf.tensordot(A_var, tf.reshape(x_pred, (args.nNodes ** 2, 1)), 1)
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


def vae_ls2sr(test_traffic, vae, graph, args):
    print('vae_ls2sr solver')
    optimizer = tf.keras.optimizers.Adam()

    def routing_cycle(solver, tm, gt_tms, p_solution, nNodes):
        utilizations = []
        tm = tm.reshape((-1, nNodes, nNodes))
        gt_tms = gt_tms.reshape((-1, nNodes, nNodes))

        tm[tm <= 0.0] = 0.0
        gt_tms[gt_tms <= 0.0] = 0.0

        tm[:] = tm[:] * (1.0 - np.eye(nNodes))
        gt_tms[:] = gt_tms[:] * (1.0 - np.eye(nNodes))
        tm = tm.reshape((nNodes, nNodes))

        try:
            solution = solver.solve(tm, solution=p_solution)  # solve backtrack solution (line 131)
        except:
            solution = solver.initialize()

        linkload = None
        for i in range(gt_tms.shape[0]):
            u, linkload = solver.evaluate(solution, gt_tms[i])
            utilizations.append(u)
        return utilizations, solution, linkload

    for run_time in range(args.nrun):
        results = []
        solver = LS2SRSolver(graph=graph, args=args)

        solution = solver.initialize()
        u, linkload = solver.evaluate(solution=solution, tm=test_traffic[0])
        for i in tqdm(range(0, test_traffic.shape[0], args.seq_len_y)):
            tms = test_traffic[i:i + args.seq_len_y]
            A = solver.extractRoutingMX(solution)
            A_var = tf.Variable(A, dtype=tf.float32)
            linkload = linkload / args.scale
            tm = linkload2tm(model=vae, y=linkload, A_var=A_var, optimizer=optimizer, args=args)
            tm = tm * args.scale

            u, solution, linkload = routing_cycle(solver, tm=tm,
                                                  gt_tms=tms, p_solution=solution, nNodes=args.nNodes)

            _solution = np.copy(solution)
            results.append((u, _solution))

        mlu, solution = extract_results(results)
        route_changes = get_route_changes_heuristic(solution)

        print('Route changes: total {:.3f} avg {:.3f}'.format(np.sum(route_changes),
                                                              np.sum(route_changes) / test_traffic.shape[0]))
        print('last_step ls2sr    {}      | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(args.model,
                                                                                      np.min(mlu),
                                                                                      np.mean(mlu),
                                                                                      np.max(mlu),
                                                                                      np.std(mlu)))
        congested = mlu[mlu >= 1.0].size
        print('Congestion_rate: {}/{}'.format(congested, mlu.size))

        save_results(args.log_dir, 'ls2sr_vae_run_{}'.format(run_time), mlu, route_changes)
