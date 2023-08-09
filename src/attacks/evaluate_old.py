"""
Code for evaluate the link stealthy performance given graph, model, and node index for evaluate
The code will compute a vector of similarity.
"""
import torch
import numpy as np
from scipy.spatial import distance

def kl_divergence(P, Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001
    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    Q = Q + epsilon
    divergence = torch.sum(P * torch.log(P / Q))
    return divergence


def js_divergence(P, Q):
    return torch.tensor(distance.jensenshannon(P, Q, 2.0))


def entropy(P):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001
    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    entropy_value = -torch.sum(P * torch.log(P))
    return entropy_value

def average(a, b):
    return (a + b) / 2


def hadamard(a, b):
    return a * b


def weighted_l1(a, b):
    return abs(a - b)


def weighted_l2(a, b):
    return abs((a - b) * (a - b))


def concate_all(a, b):
    return torch.cat(
        (average(a, b), hadamard(a, b), weighted_l1(a, b), weighted_l2(a, b)))


def operator_func(operator, a, b):
    if operator == "average":
        return average(a, b)
    elif operator == "hadamard":
        return hadamard(a, b)
    elif operator == "weighted_l1":
        return weighted_l1(a, b)
    elif operator == "weighted_l2":
        return weighted_l2(a, b)
    elif operator == "concate_all":
        return concate_all(a, b)

def cosine_similarity(a, b):
    # give a:[m*k] b:[n*k] return a cosine similarity matrix [m*n]
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0,1))
    return res     

def euclidean_similarity(a, b):
    # Euclidean distance, return the negative distance
    res = torch.cdist(a, b, p=2)
    return -res

def correlation_similarity(a, b):
    """
    1 - correlation : measurements of dissimilarity of two vectors
    """
    n = a.shape[0]
    m = b.shape[0]
    mean_A = torch.mean(a, dim=1, keepdim=True)
    mean_B = torch.mean(b, dim=1, keepdim=True)
    std_A = torch.std(a, dim=1, keepdim=True)
    std_B = torch.std(b, dim=1, keepdim=True)

    corr_dist = 1 - (2 / (a.shape[1] - 1)) * torch.sum((a - mean_A)[:, None, :] * (b - mean_B)[None, :, :] / (std_A * std_B), dim=2)

    return -corr_dist    

def chebyshev_similarity(a, b):
    res = torch.cdist(a, b, p=float('inf'))
    return -res

def braycurtis_similarity(a, b):
    """
    Distance measurements to evaluate the absolute of difference between two vectors.
    """
    braycurtis_dist = torch.cdist(a, b, p=1) / (torch.abs(a) + torch.abs(b) + 1e-20).sum(dim=1, keepdim=True)
    return -braycurtis_dist

def canberra_similarity(a, b):
    """
    Distance to measure the percentage difference between two vectors.
    """
    numerator = torch.ones_like(torch.cdist(a, b, p=1))
    denominator = torch.abs(a).sum(dim=1, keepdim=True) + torch.abs(b).sum(dim=1, keepdim=True).T
    canberra_sim = numerator / (1 + torch.cdist(a, b, p=1) / denominator)
    return canberra_sim

def cityblock(a, b):
    res = torch.cdist(a, b, p=1)
    return -res

def sqeuclidean_similarity(a, b):
    res = torch.cdist(a, b, p=2, compute_mode='use_mm_for_euclid_dist_if_necessary')
    return -res

def compute_neighbor_similarity(model, graph, train_mask, device):
    # To do edge stealthy attack, we compute the cosine similarity between each pair of nodes 
    model.to(device)
    graph.to(device)
    logits = model(graph, graph.ndata['feat'])
    
    train_logits = logits[train_mask] # [n_train, n_classes]
    print(train_logits.shape)
    # compute a similarity matrix [n_train, n_train], each cell is the cosine_similarity
    print("Compute the euclidean similarity")
    similarity_list = [euclidean_similarity, correlation_similarity, chebyshev_similarity,
                       braycurtis_similarity, canberra_similarity, cityblock, sqeuclidean_similarity, cosine_similarity]
    # similarity matrix 
    similarity_matrix=torch.cat([row(train_logits, train_logits).unsqueeze(2) for row in similarity_list], dim=2) # [n_train, n_train, n_similarity_metrics]
    
    # Get a subgraph for only the evaluation nodes
    train_graph = graph.subgraph(torch.arange(graph.num_nodes())[train_mask])

    # Get the connectivity relation of these nodes
    adj_matrix = train_graph.adjacency_matrix(transpose=True).to_dense() #[n_train, n_train]

    # Compute the similarity of nodes connected  
    neighbor_similarity = similarity_matrix[torch.ne(adj_matrix, 0)] # [n_connected, n_similarity metrics]

    # Compute the similarity of nodes are not connected
    not_neighbor_similarity = similarity_matrix[torch.ne(1-adj_matrix-torch.eye(train_logits.shape[0]), 0)] # [not_n_connected, n_similarity metrics]
    
    n_connected = neighbor_similarity.shape[0]
    n_not_connected = not_neighbor_similarity.shape[0]

    # Sample n_connected nodes in n_not_connected to balance two types of connectivity
    index = np.random.choice(n_connected, n_not_connected)

    neighbor_similarity = neighbor_similarity[index]

    return neighbor_similarity, not_neighbor_similarity