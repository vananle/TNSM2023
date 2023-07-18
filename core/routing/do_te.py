import sys

import tensorflow as tf
from tqdm import tqdm

from .ls2sr import LS2SRSolver
from .ls2sr_vae import LS2SR_VAE_Solver
from .max_step_sr import MaxStepSRSolver
from .multi_step_sr import MultiStepSRSolver
from .oblivious_routing import ObliviousRoutingSolver
from .one_step_sr import OneStepSRSolver
from .srls import SRLS, CapacityData, ShortestPaths
from .te_util import *

sys.path.append('../core/')


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


def prepare_te_data(x_gt, y_gt, yhat, args):
    te_step = args.test_size if args.te_step == 0 else args.te_step
    x_gt = x_gt[0:te_step:args.seq_len_y]
    y_gt = y_gt[0:te_step:args.seq_len_y]
    if args.run_te == 'ls2sr' or args.run_te == 'onestep':
        yhat = yhat[0:te_step:args.seq_len_y]

    return x_gt, y_gt, yhat


def oblivious_routing_solver(y_gt, G, segments, te_step, args):
    solver = ObliviousRoutingSolver(G, segments)
    solver.solve()
    print('Solving Obilious Routing: Done')
    results = []

    def f(tms):
        tms = tms.reshape((-1, args.nNodes, args.nNodes))
        tms[tms <= 0.0] = 0.0
        tms[:] = tms[:] * (1.0 - np.eye(args.nNodes))
        return oblivious_sr(solver, tms)

    for i in tqdm(range(te_step)):
        results.append(f(tms=y_gt[i]))

    mlu, solutions = extract_results(results)
    rc = get_route_changes(solutions, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(rc), np.std(rc)))

    print('Oblivious              | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu),
                                                                              np.mean(mlu),
                                                                              np.max(mlu),
                                                                              np.std(mlu)))
    congested = mlu[mlu >= 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'or', mlu, rc)


def gwn_ls2sr(yhat, y_gt, graph, te_step, logger, args):
    args.nNodes = graph.number_of_nodes()
    print('ls2sr_gwn_p2')
    for run_test in range(args.nrun):

        results = []
        solver = LS2SRSolver(graph=graph, args=args)

        solution = None
        dynamicity = np.zeros(shape=(te_step, 7))

        ts = 0
        p_solution = None
        for i in tqdm(range(te_step)):
            mean = np.mean(y_gt[i], axis=1)
            std_mean = np.std(mean)
            std = np.std(y_gt[i], axis=1)
            std_std = np.std(std)

            maxmax_mean = np.max(y_gt[i]) / np.mean(y_gt[i])

            theo_lamda = calculate_lamda(y_gt=y_gt[i])

            pred_tm = yhat[i]
            p_solution = np.copy(solution)
            u, solution = p2_heuristic_solver(solver, tm=pred_tm,
                                              gt_tms=y_gt[i], p_solution=solution, nNodes=args.nNodes)

            dynamicity[i] = [np.sum(y_gt[i]), std_mean, std_std, np.sum(std), maxmax_mean, np.mean(u), theo_lamda]

            _solution = np.copy(solution)
            results.append((u, _solution))

            max_ygt = np.max(y_gt[i], axis=0)
            error = calc_metrics_np(preds=yhat[i], labels=max_ygt)
            logger.writer.add_scalar('Error/rse', error[0], i)
            logger.writer.add_scalar('Error/mae', error[1], i)
            logger.writer.add_scalar('Error/mse', error[2], i)

            for mlu in u:
                logger.writer.add_scalar('Test/mlu', mlu, ts)
                ts += 1

            if i > 0:
                rc = count_routing_change(_solution, p_solution)
                logger.writer.add_scalar('Test/rc', rc, i)
            else:
                logger.writer.add_scalar('Test/rc', 0, i)

        mlu, solution = extract_results(results)
        route_changes = get_route_changes_heuristic(solution)

        print('Route changes: Total: {}  - Avg {:.3f} std {:.3f}'.format(np.sum(route_changes),
                                                                         np.sum(route_changes) /
                                                                         (args.seq_len_y * route_changes.shape[0]),
                                                                         np.std(route_changes)))
        print('gwn ls2sr    {}      | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(args.model,
                                                                                np.min(mlu),
                                                                                np.mean(mlu),
                                                                                np.max(mlu),
                                                                                np.std(mlu)))
        congested = mlu[mlu >= 1.0].size
        print('Congestion_rate: {}/{}'.format(congested, mlu.size))

        save_results(args.log_dir, 'gwn_ls2sr_cs_{}_run_{}'.format(args.cs, run_test), mlu, route_changes)
        np.save(os.path.join(args.log_dir, 'gwn_ls2sr_cs_{}_dyn_run_{}'.format(args.cs, run_test)), dynamicity)


def createGraph_srls(NodesFile, EdgesFile):
    def addEdge(graph, src, dst, w, bw, delay, idx, capacity):
        graph.add_edge(src, dst, weight=w,
                       capacity=bw,
                       delay=delay)
        graph.edges[src, dst]['index'] = idx
        capacity.append(bw)

    capacity = []
    G = nx.DiGraph()
    df = pd.read_csv(NodesFile, delimiter=' ')
    for i, row in df.iterrows():
        G.add_node(i, label=row.label, pos=(row.x, row.y))

    nNodes = G.number_of_nodes()

    index = 0
    df = pd.read_csv(EdgesFile, delimiter=' ')
    for _, row in df.iterrows():
        i = row.src
        j = row.dest
        if (i, j) not in G.edges:
            addEdge(G, i, j, row.weight, row.bw, row.delay, index, capacity)
            index += 1
        if (j, i) not in G.edges:
            addEdge(G, j, i, row.weight, row.bw, row.delay, index, capacity)
            index += 1

    nEdges = G.number_of_edges()
    sPathNode = []
    sPathEdge = []
    nSPath = []
    for u in G.nodes:
        A = []
        B = []
        C = []
        for v in G.nodes:
            A.append(list(nx.all_shortest_paths(G, u, v)))
            B.append([])
            C.append(0)
            if len(A[-1][0]) >= 2:
                C[-1] = len(A[-1])
                for path in A[-1]:
                    B[-1].append([])
                    for j in range(len(path) - 1):
                        B[-1][-1].append(G[path[j]][path[j + 1]]['index'])
        sPathNode.append(A)
        sPathEdge.append(B)
        nSPath.append(C)
    capacity = CapacityData(capacity)
    sp = ShortestPaths(sPathNode, sPathEdge, nSPath)
    G.sp = sp
    return G, nNodes, nEdges, capacity, sp


def gwn_srls(yhat, y_gt, graphs, te_step, args):
    print('GWN SRLS')
    G, nNodes, nEdges, capacity, sp = graphs
    for run_test in range(args.nrun):

        results = []
        solver = SRLS(sp, capacity, nNodes, nEdges, args.timeout)
        LinkLoads, RoutingMatrices, TMs = [], [], []
        dynamicity = np.zeros(shape=(te_step, 7))
        for i in tqdm(range(te_step)):
            mean = np.mean(y_gt[i], axis=1)
            std_mean = np.std(mean)
            std = np.std(y_gt[i], axis=1)
            std_std = np.std(std)

            maxmax_mean = np.max(y_gt[i]) / np.mean(y_gt[i])

            theo_lamda = calculate_lamda(y_gt=y_gt[i])

            pred_tm = yhat[i]
            u, solutions, linkloads, routingMxs = p2_srls_solver(solver, tm=pred_tm, gt_tms=y_gt[i], nNodes=args.nNodes)
            solutions = np.asarray(solutions)
            dynamicity[i] = [np.sum(y_gt[i]), std_mean, std_std, np.sum(std), maxmax_mean, np.mean(u), theo_lamda]

            _solutions = np.copy(solutions)
            results.append((u, _solutions))
            LinkLoads.append(linkloads)
            RoutingMatrices.append(routingMxs)
            TMs.append(y_gt[i])

        LinkLoads = np.stack(LinkLoads, axis=0)
        RoutingMatrices = np.stack(RoutingMatrices, axis=0)
        TMs = np.stack(TMs, axis=0)

        mlu, solution = extract_results(results)
        route_changes = get_route_changes(solution, G)

        print('Route changes: Avg {:.3f} std {:.3f}'.format(np.sum(route_changes) /
                                                            (args.seq_len_y * route_changes.shape[0]),
                                                            np.std(route_changes)))
        print('gwn SRLS    {}      | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(args.model,
                                                                               np.min(mlu),
                                                                               np.mean(mlu),
                                                                               np.max(mlu),
                                                                               np.std(mlu)))
        congested = mlu[mlu >= 1.0].size
        print('Congestion_rate: {}/{}'.format(congested, mlu.size))

        save_results(args.log_dir, 'gwn_srls_run_{}'.format(run_test), mlu, route_changes)
        np.save(os.path.join(args.log_dir, 'gwn_srls_dyn'), dynamicity)

        # np.save(os.path.join(args.log_dir, 'LinkLoads_gwn_srls'), LinkLoads)
        # np.save(os.path.join(args.log_dir, 'RoutingMatrices_gwn_srls'), RoutingMatrices)
        # np.save(os.path.join(args.log_dir, 'TMs_gwn_srls'), TMs)


def gt_srls(y_gt, graphs, te_step, args):
    print('gt_srls')
    G, nNodes, nEdges, capacity, sp = graphs
    for run_test in range(args.nrun):

        results = []
        solver = SRLS(sp, capacity, nNodes, nEdges, args.timeout)
        LinkLoads, RoutingMatrices, TMs = [], [], []
        dynamicity = np.zeros(shape=(te_step, 7))
        for i in tqdm(range(te_step)):
            mean = np.mean(y_gt[i], axis=1)
            std_mean = np.std(mean)
            std = np.std(y_gt[i], axis=1)
            std_std = np.std(std)

            maxmax_mean = np.max(y_gt[i]) / np.mean(y_gt[i])

            theo_lamda = calculate_lamda(y_gt=y_gt[i])

            pred_tm = np.max(y_gt[i], axis=0, keepdims=True)
            u, solutions, linkloads, routingMxs = p2_srls_solver(solver, tm=pred_tm, gt_tms=y_gt[i], nNodes=args.nNodes)
            solutions = np.asarray(solutions)
            dynamicity[i] = [np.sum(y_gt[i]), std_mean, std_std, np.sum(std), maxmax_mean, np.mean(u), theo_lamda]

            _solutions = np.copy(solutions)
            results.append((u, _solutions))
            LinkLoads.append(linkloads)
            RoutingMatrices.append(routingMxs)
            TMs.append(y_gt[i])

        mlu, solution = extract_results(results)
        route_changes = get_route_changes(solution, G)
        LinkLoads = np.stack(LinkLoads, axis=0)
        RoutingMatrices = np.stack(RoutingMatrices, axis=0)
        TMs = np.stack(TMs, axis=0)

        print('Route changes: Avg {:.3f} std {:.3f}'.format(np.sum(route_changes) /
                                                            (args.seq_len_y * route_changes.shape[0]),
                                                            np.std(route_changes)))
        print('gt srls     {}      | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(args.model,
                                                                               np.min(mlu),
                                                                               np.mean(mlu),
                                                                               np.max(mlu),
                                                                               np.std(mlu)))
        congested = mlu[mlu >= 1.0].size
        print('Congestion_rate: {}/{}'.format(congested, mlu.size))

        save_results(args.log_dir, 'gt_srls_run_{}'.format(run_test), mlu, route_changes)
        np.save(os.path.join(args.log_dir, 'gt_srls_dyn'), dynamicity)

        # np.save(os.path.join(args.log_dir, 'LinkLoads_gwn_srls'), LinkLoads)
        # np.save(os.path.join(args.log_dir, 'RoutingMatrices_gwn_srls'), RoutingMatrices)
        # np.save(os.path.join(args.log_dir, 'TMs_gwn_srls'), TMs)


def optimal_p0_solver(y_gt, G, segments, te_step, logger, args):
    solver = OneStepSRSolver(G, segments)

    def f(gt_tms):
        gt_tms = gt_tms.reshape((-1, args.nNodes, args.nNodes))
        gt_tms[gt_tms <= 0.0] = 0.0
        gt_tms[:] = gt_tms[:] * (1.0 - np.eye(args.nNodes))

        return p0(solver, gt_tms)

    results = Parallel(n_jobs=os.cpu_count() - 4)(delayed(f)(gt_tms=y_gt[i]) for i in range(te_step))

    mlu, solution = extract_results(results)
    solution = np.reshape(solution, newshape=(-1, args.nNodes, args.nNodes, args.nNodes))
    rc = get_route_changes(solution, G)

    ts = 0
    for mlu in mlu.flatten():
        logger.writer.add_scalar('MLU/p0', mlu, ts)
        ts += 1

    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(rc), np.std(rc)))
    print('Optimal p0           | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu),
                                                                            np.mean(mlu),
                                                                            np.max(mlu),
                                                                            np.std(mlu)))
    congested = mlu[mlu >= 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'p0', mlu, rc)


def optimal_p1_solver(y_gt, G, segments, te_step, args):
    solver = MultiStepSRSolver(G, segments)

    def f(gt_tms, tms):
        tms = tms.reshape((-1, args.nNodes, args.nNodes))
        gt_tms = gt_tms.reshape((-1, args.nNodes, args.nNodes))

        tms[tms <= 0.0] = 0.0
        gt_tms[gt_tms <= 0.0] = 0.0

        tms[:] = tms[:] * (1.0 - np.eye(args.nNodes))
        gt_tms[:] = gt_tms[:] * (1.0 - np.eye(args.nNodes))
        return p1(solver, tms, gt_tms)

    results = Parallel(n_jobs=os.cpu_count() - 4)(delayed(f)(
        tms=y_gt[i], gt_tms=y_gt[i]) for i in range(te_step))

    mlu, solution = extract_results(results)
    rc = get_route_changes(solution, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(rc), np.std(rc)))
    print('Optimal p1               | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu),
                                                                                np.mean(mlu),
                                                                                np.max(mlu),
                                                                                np.std(mlu)))

    congested = mlu[mlu >= 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'p1', mlu, rc)


def optimal_p2_solver(y_gt, G, segments, te_step, args):
    solver = MaxStepSRSolver(G, segments)

    def f(gt_tms, tms):
        tms = tms.reshape((-1, args.nNodes, args.nNodes))
        gt_tms = gt_tms.reshape((-1, args.nNodes, args.nNodes))

        tms[tms <= 0.0] = 0.0
        gt_tms[gt_tms <= 0.0] = 0.0

        tms[:] = tms[:] * (1.0 - np.eye(args.nNodes))
        gt_tms[:] = gt_tms[:] * (1.0 - np.eye(args.nNodes))
        tms = tms.reshape((args.nNodes, args.nNodes))

        return p2(solver, tms=tms, gt_tms=gt_tms)

    results = Parallel(n_jobs=os.cpu_count() - 4)(delayed(f)(
        gt_tms=y_gt[i], tms=np.max(y_gt[i], axis=0, keepdims=True)) for i in range(te_step))

    mlu, solution_optimal_p2 = extract_results(results)
    rc = get_route_changes(solution_optimal_p2, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(rc), np.std(rc)))
    print('Optimal p2                | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu),
                                                                                 np.mean(mlu),
                                                                                 np.max(mlu),
                                                                                 np.std(mlu)))
    congested = mlu[mlu >= 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'p2', mlu, rc)


def optimal_p3_solver(y_gt, G, segments, te_step, args):
    t_prime = int(args.seq_len_y / args.trunk)
    solver = MultiStepSRSolver(G, segments)

    def f(gt_tms, tms):
        tms = tms.reshape((-1, args.nNodes, args.nNodes))
        gt_tms = gt_tms.reshape((-1, args.nNodes, args.nNodes))

        tms[tms <= 0.0] = 0.0
        gt_tms[gt_tms <= 0.0] = 0.0

        tms[:] = tms[:] * (1.0 - np.eye(args.nNodes))
        gt_tms[:] = gt_tms[:] * (1.0 - np.eye(args.nNodes))

        return p3(solver, tms, gt_tms)

    results = Parallel(n_jobs=os.cpu_count() - 4)(delayed(f)(
        tms=np.stack([np.max(y_gt[i][j:j + t_prime], axis=0) for j in range(0, y_gt[i].shape[0], t_prime)]),
        gt_tms=y_gt[i]) for i in range(te_step))

    mlu, solution_optimal_p3 = extract_results(results)
    rc = get_route_changes(solution_optimal_p3, G)
    print('Route changes: Avg {:.3f} std {:.3f}'.format(np.mean(rc), np.std(rc)))
    print('Optimal p3             | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(np.min(mlu),
                                                                              np.mean(mlu),
                                                                              np.max(mlu),
                                                                              np.std(mlu)))
    congested = mlu[mlu >= 1.0].size
    print('Congestion_rate: {}/{}'.format(congested, mlu.size))

    save_results(args.log_dir, 'p3', mlu, rc)


def p1(solver, tms, gt_tms):
    u = []
    try:
        solver.solve(tms)
    except:
        pass
    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i]))
    return u, solver.solution


def p3(solver, tms, gt_tms):
    u = []
    try:
        solver.solve(tms)
    except:
        pass
    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i]))
    return u, solver.solution


def p2(solver, tms, gt_tms):
    u = []

    try:
        solver.solve(tms)
    except:
        pass
    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i]))
    return u, solver.solution


def p0_ls2sr(solver, tms, gt_tms, p_solution, nNodes):
    u = []
    tms = tms.reshape((-1, nNodes, nNodes))
    gt_tms = gt_tms.reshape((-1, nNodes, nNodes))

    tms[tms <= 0.0] = 0.0
    gt_tms[gt_tms <= 0.0] = 0.0

    tms[:] = tms[:] * (1.0 - np.eye(nNodes))
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(nNodes))
    tms = tms.reshape((-1, nNodes, nNodes))

    solutions = []
    for i in range(gt_tms.shape[0]):
        solution = solver.solve(tms[i], solution=p_solution)  # solve backtrack solution (line 131)
        u.append(solver.evaluate(solution, gt_tms[i]))
        solutions.append(solution)

    return u, solutions


def p2_heuristic_solver(solver, tm, gt_tms, p_solution, nNodes):
    u = []
    tm = tm.reshape((-1, nNodes, nNodes))
    gt_tms = gt_tms.reshape((-1, nNodes, nNodes))

    tm[tm <= 0.0] = 0.0
    gt_tms[gt_tms <= 0.0] = 0.0

    tm[:] = tm[:] * (1.0 - np.eye(nNodes))
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(nNodes))
    tm = tm.reshape((nNodes, nNodes))

    solution = solver.solve(tm, solution=p_solution)  # solve backtrack solution (line 131)

    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(solution, gt_tms[i]))
    return u, solution


def flowidx2srcdst(flow_idx, nNodes):
    src = flow_idx / nNodes
    src = src.astype(np.int)

    dst = flow_idx % nNodes
    dst = dst.astype(np.int)

    srcdst_idx = np.stack([src, dst], axis=1)
    return srcdst_idx


def p2_cfr(solver, tm, gt_tms, pSolution, nNodes, num_cf):
    u = []

    tm = tm.flatten()
    topk_idx = np.argsort(tm)[::-1]
    topk_idx = topk_idx[:num_cf]

    rTm = np.copy(tm)
    rTm[topk_idx] = 0
    rTm = rTm.reshape((nNodes, nNodes))

    tm = tm[topk_idx]

    gt_tms = gt_tms.reshape((-1, nNodes, nNodes))
    gt_tms[gt_tms <= 0.0] = 0.0
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(nNodes))

    srcdst_idx = flowidx2srcdst(flow_idx=topk_idx, nNodes=nNodes)
    solution = solver.solve(tm=tm, rTm=rTm, flow_idx=srcdst_idx, pSolution=pSolution)

    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(solution=solution, tm=gt_tms[i]))
    return u, solution


def p2_srls_solver(solver, tm, gt_tms, nNodes):
    u = []
    tm = tm.reshape((-1, nNodes, nNodes))
    gt_tms = gt_tms.reshape((-1, nNodes, nNodes))

    tm[tm <= 0.0] = 0.0
    gt_tms[gt_tms <= 0.0] = 0.0

    tm[:] = tm[:] * (1.0 - np.eye(nNodes))
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(nNodes))
    tm = tm.reshape((nNodes, nNodes))

    try:
        solver.modifierTrafficMatrix(tm)  # solve backtrack solution (line 131)
        solver.solve()
    except:
        print('ERROR in p2_srls_solver --> pass')
        pass

    solution = solver.extractRoutingPath()
    linkloads, routingMxs = [], []
    solutions = []

    for i in range(gt_tms.shape[0]):
        solutions.append(solution)

        u.append(solver.evaluate(solution, gt_tms[i]))

        linkload = solver.getLinkload(routingSolution=solution, trafficMatrix=gt_tms[i])
        routingMx = solver.getRoutingMatrix(routingSolution=solution)
        linkloads.append(linkload)
        routingMxs.append(routingMx)

    return u, solutions, linkloads, routingMxs


def p2_srls_fix_max_solver(solver, solution, gt_tms, nNodes):
    gt_tms = gt_tms.reshape((-1, nNodes, nNodes))
    gt_tms[gt_tms <= 0.0] = 0.0
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(nNodes))

    linkloads, routingMxs = [], []

    for i in range(gt_tms.shape[0]):
        linkload = solver.getLinkload(routingSolution=solution, trafficMatrix=gt_tms[i])
        routingMx = solver.getRoutingMatrix(routingSolution=solution)
        linkloads.append(linkload)
        routingMxs.append(routingMx)

    return linkloads, routingMxs


def p0_srls_solver(solver, tms, gt_tms, nNodes):
    u = []
    tms = tms.reshape((-1, nNodes, nNodes))
    gt_tms = gt_tms.reshape((-1, nNodes, nNodes))

    tms[tms <= 0.0] = 0.0
    gt_tms[gt_tms <= 0.0] = 0.0

    tms[:] = tms[:] * (1.0 - np.eye(nNodes))
    gt_tms[:] = gt_tms[:] * (1.0 - np.eye(nNodes))
    tms = tms.reshape((-1, nNodes, nNodes))

    solutions = []
    linkloads, routingMxs = [], []
    for i in range(gt_tms.shape[0]):
        try:
            solver.modifierTrafficMatrix(tms[i])  # solve backtrack solution (line 131)
            solver.solve()
        except:
            print('ERROR in p2_srls_solver --> pass')
            pass
        solution = solver.extractRoutingPath()

        u.append(solver.evaluate(solution, gt_tms[i]))
        solutions.append(solution)

        linkload = solver.getLinkload(routingSolution=solution, trafficMatrix=gt_tms[i])
        routingMx = solver.getRoutingMatrix(routingSolution=solution)
        linkloads.append(linkload)
        routingMxs.append(routingMx)

    return u, solutions, linkloads, routingMxs


def last_step_sr(solver, last_tm, gt_tms):
    u = []
    try:
        solver.solve(last_tm)
    except:
        pass

    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i]))
    return u, solver.solution


def first_step_sr(solver, first_tm, gt_tms):
    u = []
    try:
        solver.solve(first_tm)
    except:
        pass

    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i]))
    return u, solver.solution


def one_step_predicted_sr(solver, tm, gt_tms):
    u = []
    try:
        solver.solve(tm)
    except:
        pass

    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(gt_tms[i]))
    return u, solver.solution


def p0(solver, gt_tms):
    u = []
    solutions = []
    for i in range(gt_tms.shape[0]):
        try:
            solver.solve(gt_tms[i])
        except:
            pass

        solution = solver.solution
        solutions.append(solution)
        u.append(solver.evaluate(gt_tms[i], solution))

    solutions = np.stack(solutions, axis=0)
    return u, solutions


def oblivious_sr(solver, tms):
    u = []
    for i in range(tms.shape[0]):
        u.append(solver.evaluate(tms[i]))

    return u, solver.solution


def run_te(x_gt, y_gt, yhat, logger, args):
    print('|--- run TE on DIRECTED graph')

    te_step = x_gt.shape[0]
    print('    Method           |   Min     Avg    Max     std')

    if args.run_te == 'gwn_ls2sr':
        if 'geant2' in args.dataset or 'germany' in args.dataset:
            graph = load_nx_graph_from_nedfile(args.dataset, args.datapath)
        else:
            graph = load_network_topology(args.dataset, args.datapath)

        gwn_ls2sr(yhat, y_gt, graph, te_step, logger, args)
    elif args.run_te == 'gwn_srls':
        graphs = createGraph_srls(os.path.join(args.datapath, 'topo/{}_node.csv'.format(args.dataset)),
                                  os.path.join(args.datapath, 'topo/{}_edge.csv'.format(args.dataset)))
        gwn_srls(yhat, y_gt, graphs, te_step, args)
    elif args.run_te == 'gt_srls':
        graphs = createGraph_srls(os.path.join(args.datapath, 'topo/{}_node.csv'.format(args.dataset)),
                                  os.path.join(args.datapath, 'topo/{}_edge.csv'.format(args.dataset)))
        gt_srls(y_gt, graphs, te_step, args)
    elif args.run_te == 'p0':
        graph = load_network_topology(args.dataset, args.datapath)
        segments = compute_path(graph, args.dataset, args.datapath)
        optimal_p0_solver(y_gt, graph, segments, te_step, logger, args)
    elif args.run_te == 'p1':
        graph = load_network_topology(args.dataset, args.datapath)
        segments = compute_path(graph, args.dataset, args.datapath)
        optimal_p1_solver(y_gt, graph, segments, te_step, args)
    elif args.run_te == 'p2':  # (or gt_p2)
        graph = load_network_topology(args.dataset, args.datapath)
        segments = compute_path(graph, args.dataset, args.datapath)
        optimal_p2_solver(y_gt, graph, segments, te_step, args)
    elif args.run_te == 'p3':
        graph = load_network_topology(args.dataset, args.datapath)
        segments = compute_path(graph, args.dataset, args.datapath)
        optimal_p3_solver(y_gt, graph, segments, te_step, args)
    elif args.run_te == 'or':
        graph = load_network_topology(args.dataset, args.datapath)
        segments = compute_path(graph, args.dataset, args.datapath)
        oblivious_routing_solver(y_gt, graph, segments, te_step, args)
    else:
        raise RuntimeError('TE not found!')


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

    np.save(os.path.join(args.log_dir, '{}_LL'.format(fname)), LL)
    np.save(os.path.join(args.log_dir, '{}_LM'.format(fname)), LM)
    np.save(os.path.join(args.log_dir, '{}_A'.format(fname)), As)
    np.save(os.path.join(args.log_dir, '{}_TM'.format(fname)), TMs)
    print('LL shape: ', LL.shape)
    print('LM shape: ', LM.shape)
    print('As shape: ', As.shape)
    print('TMs shape: ', TMs.shape)


# routing with VAE

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


def vae_ls2sr(test_traffic, vae, graph, sp, args, writer, fname):
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
        solver = LS2SR_VAE_Solver(graph=graph, args=args, sp=sp)

        solution = solver.initialize()
        u, linkload = solver.evaluate(solution=solution, tm=test_traffic[args.seq_len_x - 1])
        for i in tqdm(range(args.seq_len_x, test_traffic.shape[0], args.seq_len_y)):
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
        rc = get_route_changes_heuristic(solution)

        ts = 1
        for u in mlu.flatten():
            writer.add_scalar('{}/mlu'.format(fname), u, ts)
            ts += 1

        writer.add_scalar('{}/rc'.format(fname), 0, 1)
        ts = 2
        for _rc in rc.flatten():
            writer.add_scalar('{}/rc'.format(fname), _rc, ts)
            ts += 1

        # saving results
        save_path = os.path.join(writer.log_dir, 'routing_solution_{}.pkl'.format(fname))
        with open(save_path, 'wb') as f:
            pickle.dump(solution, f)

        save_path = os.path.join(writer.log_dir, 'mlu_{}.pkl'.format(fname))
        with open(save_path, 'wb') as f:
            pickle.dump(mlu.flatten(), f)

        save_path = os.path.join(writer.log_dir, 'rc_{}.pkl'.format(fname))
        with open(save_path, 'wb') as f:
            pickle.dump(rc.flatten(), f)

        print('Route changes: total {:.3f} avg {:.3f}'.format(np.sum(rc),
                                                              np.sum(rc) / test_traffic.shape[0]))
        print('last_step ls2sr    {}      | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(args.model,
                                                                                      np.min(mlu),
                                                                                      np.mean(mlu),
                                                                                      np.max(mlu),
                                                                                      np.std(mlu)))
        congested = mlu[mlu >= 1.0].size
        print('Congestion_rate: {}/{}'.format(congested, mlu.size))

        save_results(args.log_dir, 'ls2sr_vae_run_{}'.format(run_time), mlu, rc)
