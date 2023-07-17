from tqdm import tqdm

from .ls2sr import LS2SRSolver
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


def gwn_ls2sr(yhat, y_gt, graph, te_step, args):
    print('gwn_ls2sr')
    for run_test in range(args.nrun):

        results = []
        solver = LS2SRSolver(graph=graph, args=args)

        solution = None
        dynamicity = np.zeros(shape=(te_step, 7))
        for i in tqdm(range(te_step)):
            mean = np.mean(y_gt[i], axis=1)
            std_mean = np.std(mean)
            std = np.std(y_gt[i], axis=1)
            std_std = np.std(std)

            maxmax_mean = np.max(y_gt[i]) / np.mean(y_gt[i])

            theo_lamda = calculate_lamda(y_gt=y_gt[i])

            pred_tm = yhat[i]
            u, solution = p2_heuristic_solver(solver, tm=pred_tm,
                                              gt_tms=y_gt[i], p_solution=solution, nNodes=args.nNodes)

            dynamicity[i] = [np.sum(y_gt[i]), std_mean, std_std, np.sum(std), maxmax_mean, np.mean(u), theo_lamda]

            _solution = np.copy(solution)
            results.append((u, _solution))

        mlu, solution = extract_results(results)
        route_changes = get_route_changes_heuristic(solution)

        print('Route changes: Avg {:.3f} std {:.3f}'.format(np.sum(route_changes) /
                                                            (args.seq_len_y * route_changes.shape[0]),
                                                            np.std(route_changes)))
        print('gwn ls2sr    {}      | {:.3f}   {:.3f}   {:.3f}   {:.3f}'.format(args.model,
                                                                                np.min(mlu),
                                                                                np.mean(mlu),
                                                                                np.max(mlu),
                                                                                np.std(mlu)))
        congested = mlu[mlu >= 1.0].size
        print('Congestion_rate: {}/{}'.format(congested, mlu.size))

        save_results(args.log_dir, 'gwn_ls2sr_cs_{}_run_{}'.format(args.cs, run_test), mlu,
                     route_changes)
        # np.save(os.path.join(args.log_dir, 'gwn_ls2sr_dyn_cs_{}_run_{}'.format(args.cs, run_test)), dynamicity)


def p2_heuristic_solver(solver, tm, gt_tms, p_solution, nNodes):
    u = []
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

    for i in range(gt_tms.shape[0]):
        u.append(solver.evaluate(solution, gt_tms[i]))
    return u, solution


def run_te(x_gt, y_gt, yhat, args):
    print('|--- run TE on DIRECTED graph')

    te_step = x_gt.shape[0]
    print('    Method           |   Min     Avg    Max     std')

    if args.run_te == 'gwn_ls2sr':
        graph = load_network_topology(args.dataset, args.datapath)
        gwn_ls2sr(yhat, y_gt, graph, te_step, args)
    else:
        raise RuntimeError('TE not found!')
