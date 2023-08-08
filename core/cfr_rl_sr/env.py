from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys

sys.path.append('../')

import networkx as nx
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

from mtsr.routing.te_util import compute_path
from mtsr.routing.te_util import load_network_topology
from mtsr.utils.data_loading import load_raw_data

def save(path, obj):
    with open(path, 'wb') as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)


class Topology(object):
    def __init__(self, args):
        self.args = args
        self.shortest_paths_file = args.dataset + '_shortest_paths'

        self.graph = None
        self.num_nodes = None
        self.num_links = None
        self.link_idx_to_sd = None
        self.link_sd_to_idx = None
        self.link_capacities = None
        self.link_weights = None

        self.pair_idx_to_sd = []
        self.pair_sd_to_idx = {}
        # Shortest paths
        self.shortest_paths = []
        self.num_pairs = 0

        self.load_topology(args)
        self.calculate_paths()
        self.segment = compute_path(self.graph, args.dataset, args.data_folder)

    def load_topology(self, args):
        self.graph = load_network_topology(dataset=args.dataset, data_folder=args.data_folder)

        self.num_nodes = self.graph.number_of_nodes()
        self.num_links = self.graph.number_of_edges()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty(self.num_links)
        self.link_weights = np.empty(self.num_links)

        for i, (s, d) in enumerate(self.graph.edges):
            self.link_idx_to_sd[int(i)] = (int(s), int(d))
            self.link_sd_to_idx[(int(s), int(d))] = int(i)
            self.link_capacities[int(i)] = self.graph[s][d]['capacity']
            self.link_weights[int(i)] = self.graph[s][d]['weight']

        assert len(self.graph.nodes()) == self.num_nodes and len(self.graph.edges()) == self.num_links

    def calculate_paths(self):
        self.pair_idx_to_sd = []
        self.pair_sd_to_idx = {}
        # Shortest paths
        self.shortest_paths = []
        if os.path.exists(self.shortest_paths_file):
            print('[*] Loading shortest paths...', self.shortest_paths_file)
            f = open(self.shortest_paths_file, 'r')
            self.num_pairs = 0
            for line in f:
                sd = line[:line.find(':')]
                s = int(sd[:sd.find('-')])
                d = int(sd[sd.find('>') + 1:])
                self.pair_idx_to_sd.append((s, d))
                self.pair_sd_to_idx[(s, d)] = self.num_pairs
                self.num_pairs += 1
                self.shortest_paths.append([])
                paths = line[line.find(':') + 1:].strip()[1:-1]
                while paths != '':
                    idx = paths.find(']')
                    path = paths[1:idx]
                    node_path = np.array(path.split(',')).astype(np.int16)
                    assert node_path.size == np.unique(node_path).size
                    self.shortest_paths[-1].append(node_path)
                    paths = paths[idx + 3:]
        else:
            print('[!] Calculating shortest paths...')
            f = open(self.shortest_paths_file, 'w+')
            self.num_pairs = 0
            for s in range(self.num_nodes):
                for d in range(self.num_nodes):
                    if s != d:
                        self.pair_idx_to_sd.append((s, d))
                        self.pair_sd_to_idx[(s, d)] = self.num_pairs
                        self.num_pairs += 1
                        self.shortest_paths.append(list(nx.all_shortest_paths(self.graph, s, d, weight='weight')))
                        line = str(s) + '->' + str(d) + ': ' + str(self.shortest_paths[-1])
                        f.writelines(line + '\n')

        assert self.num_pairs == self.num_nodes * (self.num_nodes - 1)
        f.close()

        print('pairs: %d, nodes: %d, links: %d\n' \
              % (self.num_pairs, self.num_nodes, self.num_links))


class Traffic(object):
    def __init__(self, args, is_training=False):
        self.num_node = args.num_node

        data = self.split_data(args)

        if is_training:
            traffic_matrices = data['train']
        else:
            traffic_matrices = data['test']

        tms_shape = traffic_matrices.shape
        self.tm_cnt = tms_shape[0]
        self.traffic_matrices = np.reshape(traffic_matrices, newshape=(self.tm_cnt, self.num_node, self.num_node))
        print('Traffic matrices dims: [%d, %d, %d]\n' % (self.traffic_matrices.shape[0],
                                                         self.traffic_matrices.shape[1],
                                                         self.traffic_matrices.shape[2]))

    @staticmethod
    def split_data(args):

        train_df, val_df, test_df = load_raw_data(args)

        data = {
            'train': train_df,
            'val': val_df,
            'test': test_df,
        }
        return data


class Environment(object):
    def __init__(self, args, is_training=False):
        self.data_folder = args.data_folder
        self.topology = Topology(args)
        self.traffic = Traffic(args)
        self.traffic_matrices = self.traffic.traffic_matrices  # kbps
        # self.traffic_matrices = self.traffic.traffic_matrices * 100 * 8 / 300 / 1000  # kbps
        self.tm_cnt = self.traffic.tm_cnt
        self.num_pairs = self.topology.num_pairs
        self.pair_idx_to_sd = self.topology.pair_idx_to_sd
        self.pair_sd_to_idx = self.topology.pair_sd_to_idx
        self.num_nodes = self.topology.num_nodes
        self.num_links = self.topology.num_links
        self.link_idx_to_sd = self.topology.link_idx_to_sd
        self.link_sd_to_idx = self.topology.link_sd_to_idx
        self.link_capacities = self.topology.link_capacities
        self.link_weights = self.topology.link_weights
        self.shortest_paths_node = self.topology.shortest_paths  # paths consist of nodes
        self.shortest_paths_link = self.convert_to_edge_path(self.shortest_paths_node)  # paths consist of links

    def convert_to_edge_path(self, node_paths):
        edge_paths = []
        num_pairs = len(node_paths)
        for i in range(num_pairs):
            edge_paths.append([])
            num_paths = len(node_paths[i])
            for j in range(num_paths):
                edge_paths[i].append([])
                path_len = len(node_paths[i][j])
                for n in range(path_len - 1):
                    e = self.link_sd_to_idx[(node_paths[i][j][n], node_paths[i][j][n + 1])]
                    assert 0 <= e < self.num_links
                    edge_paths[i][j].append(e)
                # print(i, j, edge_paths[i][j])

        return edge_paths
