import itertools
import os
import pickle
import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from joblib import delayed, Parallel
from scipy.io import loadmat


class CapacityData:
    def __init__(self, capacity):
        self.capacity = capacity
        self.invCapacity = []
        for capa in capacity:
            self.invCapacity.append(1 / capa)


class ShortestPaths:
    def __init__(self, pathNodes, pathEdges, nPaths):
        self.pathNodes = pathNodes
        self.pathEdges = pathEdges
        self.nPaths = nPaths


def load_traffic_matrix(dataset='abilene_tm', timestep=0):
    tm = loadmat('../../dataset/dataset/{}.mat'.format(dataset))['X'][timestep, :]
    num_flow = tm.shape[1]
    num_node = int(np.sqrt(tm.shape[1]))
    tm = tm.reshape(num_node, num_node)
    return tm


def load_all_traffic_matrix(dataset='abilene_tm', timestep=0):
    tm = loadmat('../../dataset/dataset/{}.mat'.format(dataset))['X']
    num_node = int(np.sqrt(tm.shape[1]))
    if len(tm.shape) == 3:
        dpf = tm.shape[-1]
        tm = tm.reshape(-1, num_node, num_node, dpf)
    else:
        tm = tm.reshape(-1, num_node, num_node)
    return tm


def generate_traffic_matrix():
    tm = np.random.randint(low=0, high=100, size=[12, 12])
    tm = tm - tm * np.eye(12)
    return tm


def load_network_topology(dataset, data_folder):
    path = os.path.join(data_folder, f'{dataset}.npz')
    data = np.load(path)

    adj = data['adj_mx']
    capacity_mx = data['capacity_mx']
    cost_mx = data['cost_mx']

    # print(adj.shape)
    num_node = adj.shape[0]
    # initialize graph
    G = nx.DiGraph()
    for i in range(num_node):
        G.add_node(i, label=str(i))
    # add weight, capacity, delay to edge attributes

    for src in range(num_node):
        for dst in range(num_node):
            if adj[src, dst] == 1:
                G.add_edge(src, dst, weight=cost_mx[src, dst],
                           capacity=capacity_mx[src, dst])
    return G


def draw_network_topology(G, pos=None):
    if pos is None:
        pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, node_size=1000, alpha=0.5)
    nx.draw_networkx_labels(G, pos)


def shortest_path(graph, source, target):
    return nx.shortest_path(graph, source=source, target=target, weight='weight')


def get_path(graph, i, j, k):
    """
    get a path for flow (i, j) with middle point k
    return:
        - list of edges on path, list of nodes in path or (None, None) in case of duplicated path or non-simple path
    """
    p_ik = shortest_path(graph, i, k)
    p_kj = shortest_path(graph, k, j)

    edges_ik, edges_kj = [], []
    # compute edges from path p_ik, p_kj (which is 2 lists of nodes)
    for u, v in zip(p_ik[:-1], p_ik[1:]):
        edges_ik.append((u, v))
    for u, v in zip(p_kj[:-1], p_kj[1:]):
        edges_kj.append((u, v))
    return edges_ik, edges_kj


def sort_paths(graph, paths):
    weights = [[sum(graph.get_edge_data(u, v)['weight'] for u, v in path)] for path in paths]
    paths = [path for weights, path in sorted(zip(weights, paths), key=lambda x: x[0])]
    return paths


def get_paths_from_solution(graph, solution, i, j):
    n = solution.shape[0]
    if i == j:
        list_k = [i]
    elif len(solution[i, j]) < n:  # handling srls solution
        if len(solution[i, j]) == 2:
            list_k = [i]
        else:
            list_k = solution[i, j][1:-1]
    else:  # handling solution in shape of nxnxn
        list_k = np.where(solution[i, j] == 1.0)[0]

    paths = []
    for k in list_k:
        path = []
        path += shortest_path(graph, i, k)[:-1]
        path += shortest_path(graph, k, j)
        paths.append(path)
    return paths


def get_paths(graph, i, j):
    """
    get all simple path for flow (i, j) on graph G
    return:
        - flows: list of paths
        - path: list of links on path (u, v)
    """
    if i != j:
        N = graph.number_of_nodes()

        path_edges = []
        for k in range(N):
            try:
                edges = get_path(graph, i, j, k)
                path_edges.append(edges)
            except nx.NetworkXNoPath:
                pass
        return path_edges
    else:
        return []


def get_segments(graph):
    n = graph.number_of_nodes()
    segments = {}
    segments_edges = Parallel(n_jobs=os.cpu_count() * 2)(delayed(get_paths)(graph, i, j)
                                                         for i, j in itertools.product(range(n), range(n)))
    for i, j in itertools.product(range(n), range(n)):
        segments[i, j] = segments_edges[i * n + j]

    return segments


def load(path):
    with open(path, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def save(path, obj):
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)


def compute_path(graph, dataset, datapath):
    folder = os.path.join(datapath, 'topo/segments/2sr/')
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, '{}_segments_digraph.pkl'.format(dataset))
    if os.path.exists(path):
        print('|--- Load precomputed segment from {}'.format(path))
        data = load(path)
        segments = data['segments']
    else:
        segments = get_segments(graph)
        data = {
            'segments': segments,
        }
        save(path, data)

    return segments


def edge_in_segment(segments, i, j, k, u, v):
    if len(segments[i, j]) == 0:
        return 0
    elif len(segments[i, j][k]) == 0:
        return
    else:
        if len(segments[i, j][k][0]) != 0 and (u, v) in segments[i, j][k][0]:
            return 1
        elif len(segments[i, j][k][1]) != 0 and (u, v) in segments[i, j][k][1]:
            return 1
        else:
            return 0


def flatten_index(i, j, k, num_node):
    return i * num_node ** 2 + j * num_node + k


def count_routing_change(solution1, solution2):
    return np.sum(solution1 != solution2)


def draw_segment(G, segment, i, j, k):
    pos = nx.get_node_attributes(G, 'pos')
    plt.subplot(131)
    draw_network_topology(G, pos)
    plt.title('Network topology')
    plt.subplot(132)
    draw_network_topology(segment.segment_ik, pos)
    plt.title('Segment path i={} k={}'.format(i, k))
    plt.subplot(133)
    draw_network_topology(segment.segment_kj, pos)
    plt.title('Segment path k={} j={}'.format(k, j))


def draw_segment_pred(G, segment, i, j, k):
    pos = nx.get_node_attributes(G, 'pos')
    plt.subplot(231)
    draw_network_topology(G)
    plt.title('Network topology')
    plt.subplot(232)
    draw_network_topology(segment.segment_ik, pos)
    plt.title('Segment path i={} k={}'.format(i, k))
    plt.subplot(233)
    draw_network_topology(segment.segment_kj, pos)
    plt.title('Segment path k={} j={}'.format(k, j))


def draw_segment_ground_truth(G, segment, i, j, k):
    pos = nx.get_node_attributes(G, 'pos')
    plt.subplot(234)
    draw_network_topology(G)
    plt.title('Network topology')
    plt.subplot(235)
    draw_network_topology(segment.segment_ik, pos)
    plt.title('Segment path i={} k={}'.format(i, k))
    plt.subplot(236)
    draw_network_topology(segment.segment_kj, pos)
    plt.title('Segment path k={} j={}'.format(k, j))


def get_degree(G, i):
    return len([_ for _ in nx.neighbors(G, i)])


def get_nodes_sort_by_degree(G):
    nodes = np.array(G.nodes)
    degrees = np.array([get_degree(G, i) for i in nodes])
    idx = np.argsort(degrees)[::-1]
    nodes = nodes[idx]
    degrees = degrees[idx]
    return nodes, degrees


def get_node2flows(solver):
    # extract parameters
    n = solver.G.number_of_nodes()
    # initialize
    node2flows = {}
    for i in solver.G.nodes:
        node2flows[i] = []
    # enumerate all flows
    # for i, j in itertools.combinations(range(n), 2):
    for i, j in itertools.product(range(n), range(n)):
        for k, path in solver.get_paths(i, j):
            for node in solver.G.nodes:
                if node in path:
                    node2flows[node].append((i, j))
    return node2flows


def build_graph(n, connections, link_cap):
    A = np.zeros((n, n))

    for a, c in zip(A, connections):
        a[c] = 1

    nx_graph = nx.from_numpy_array(A, create_using=nx.DiGraph())
    edges = list(nx_graph.edges)
    capacities_links = []

    # The edges 0-2 or 2-0 can exist. They are duplicated (up and down) and they must have same capacity.
    for e in edges:
        if str(e[0]) + ':' + str(e[1]) in link_cap:
            capacity = link_cap[str(e[0]) + ':' + str(e[1])]
            capacities_links.append(capacity)
        elif str(e[1]) + ':' + str(e[0]) in link_cap:
            capacity = link_cap[str(e[1]) + ':' + str(e[0])]
            capacities_links.append(capacity)
        else:
            print("ERROR IN THE DATASET!")
            exit()

        nx_graph.edges[e[0], e[1]]['capacity'] = capacity

    return nx_graph


def load_nx_graph_from_nedfile(dataset, datapath):
    ned_file = os.path.join(datapath, 'topo/Network_{}.ned'.format(dataset))
    con, n, link_cap = ned2lists(ned_file)
    nx_graph = build_graph(n, con, link_cap)
    return nx_graph


def ned2lists(fname):
    channels = []
    link_cap = {}
    with open(fname) as f:
        p = re.compile(r'\s+node(\d+).port\[(\d+)\]\s+<-->\s+Channel(\d+)kbps+\s<-->\s+node(\d+).port\[(\d+)\]')
        for line in f:
            m = p.match(line)
            if m:
                auxList = []
                it = 0
                for elem in list(map(int, m.groups())):
                    if it != 2:
                        auxList.append(elem)
                    it = it + 1
                channels.append(auxList)
                link_cap[(m.groups()[0]) + ':' + str(m.groups()[3])] = int(m.groups()[2])

    n = max(map(max, channels)) + 1
    connections = [{} for i in range(n)]
    # Shape of connections[node][port] = node connected to
    for c in channels:
        connections[c[0]][c[1]] = c[2]
        connections[c[2]][c[3]] = c[0]
    # Connections store an array of nodes where each node position correspond to
    # another array of nodes that are connected to the current node
    connections = [[v for k, v in sorted(con.items())]
                   for con in connections]
    return connections, n, link_cap


def createGraph_srls(dataset, data_folder):
    capacity = []
    path = os.path.join(data_folder, f'{dataset}.npz')
    data = np.load(path)

    adj = data['adj_mx']
    capacity_mx = data['capacity_mx']
    cost_mx = data['cost_mx']

    # print(adj.shape)
    num_node = adj.shape[0]
    # initialize graph
    G = nx.DiGraph()
    for i in range(num_node):
        G.add_node(i, label=str(i))

    index = 0
    for src in range(num_node):
        for dst in range(num_node):
            if adj[src, dst] == 1:
                G.add_edge(src, dst, weight=cost_mx[src, dst],
                           capacity=capacity_mx[src, dst])
                G.edges[src, dst]['index'] = index
                capacity.append(capacity_mx[src, dst])
                index += 1

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
    return G, capacity, sp


def extract_results(results):
    mlus, solutions = [], []
    for _mlu, _solution in results:
        mlus.append(_mlu)
        solutions.append(_solution)

    mlus = np.stack(mlus, axis=0)
    solutions = np.stack(solutions, axis=0)

    return mlus, solutions


def get_route_changes_heuristic(routings):
    route_changes = []
    for t in range(routings.shape[0] - 1):
        route_changes.append(count_routing_change(routings[t + 1], routings[t]))

    route_changes = np.asarray(route_changes)
    return route_changes


def get_route_changes_optimal(routings, graph):
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
