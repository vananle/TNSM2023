import itertools
import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from scipy.io import loadmat


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


def load_network_topology(dataset, datapath):
    # initialize graph
    G = nx.DiGraph()
    # load node dataset from csv
    path = os.path.join(datapath, 'topo/{}_node.csv'.format(dataset))
    df = pd.read_csv(path, delimiter=' ')
    for i, row in df.iterrows():
        G.add_node(i, label=row.label, pos=(row.x, row.y))
    # load edge dataset from csv
    path = os.path.join(datapath, 'topo/{}_edge.csv'.format(dataset))
    df = pd.read_csv(path, delimiter=' ')
    # add weight, capacity, delay to edge attributes
    for _, row in df.iterrows():
        i = row.src
        j = row.dest
        G.add_edge(i, j, weight=row.weight,
                   capacity=row.bw,
                   delay=row.delay)
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


def g(segments, i, j, k, u, v):
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
