import matplotlib.pyplot as plt
import collections
from math import *
import argparse
import numpy as np
import os
import torch.nn.functional as F
import torch
import random
import dgl

def set_rand_seed(rand_seed):
    rand_seed = rand_seed if rand_seed >= 0 else torch.initial_seed() % 4294967295  # 2^32-1
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    np.random.seed(rand_seed)

def travel_sparse_tonum_1d(adj_mat):
    idx_a = adj_mat.coalesce().indices()[0]
    for i in range(len(idx_a)):
        if adj_mat[idx_a[i]] == float('inf'):
            adj_mat[idx_a[i]] = 0
    return adj_mat

def gcn_norm(adj_t, order=-0.5, add_self_loops=True):
    if add_self_loops:
        adj_t.fill_diagonal_(1.0)
    deg = torch.sparse.sum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow(order)
    deg_inv_sqrt = travel_sparse_tonum_1d(deg_inv_sqrt).to_dense()
    adj_t = torch.sparse.mm(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = torch.sparse.mm(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t


def homophily(graph_orig, graph_poi):
    set_rand_seed(0)

    adj_mat = graph_orig.adjacency_matrix()
    adj = gcn_norm(adj_mat,add_self_loops=False)
    x = graph_orig.ndata['feat']
    x_neg = adj @ x
    node_sims = np.array([F.cosine_similarity(xn.unsqueeze(0),xx.unsqueeze(0)).item() for (xn,xx) in zip(x_neg,x)])  

    adj_mat = graph_poi.adjacency_matrix()
    adj = gcn_norm(adj_mat,add_self_loops=False)
    x = graph_poi.ndata['feat']
    x_neg = adj @ x
    node_sims2 = np.array([F.cosine_similarity(xn.unsqueeze(0),xx.unsqueeze(0)).item() for (xn,xx) in zip(x_neg,x)])  

    plt.hist(node_sims, 100, density=True, alpha=0.75)
    plt.hist(node_sims2, 100, density=True, alpha=0.75)
    plt.grid(True)
    plt.title("Node-centric Cosine Similarity")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Percent of Nodes")
    plt.show()
    plt.close()