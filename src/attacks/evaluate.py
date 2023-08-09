"""
Code for evaluate the link stealthy performance given graph, model, and node index for evaluate
The code will compute a vector of similarity.
"""
import torch
import numpy as np
from scipy.spatial import distance

import torch
import numpy as np
from scipy.spatial import distance

import dgl

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
    entropy_value = -torch.sum(P * torch.log(P), dim=1)
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
        (average(a, b), hadamard(a, b), weighted_l1(a, b), weighted_l2(a, b)), dim=1)


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
    a_norm = a / a.norm(dim=2)[:, :, None]
    b_norm = b / b.norm(dim=2)[:, :, None]
    res = torch.bmm(a_norm, b_norm.transpose(2,1))
    return res     

def euclidean_similarity(a, b):
    # Euclidean distance, return the negative distance
    res = torch.cdist(a, b, p=2)
    return res

def correlation_similarity(a, b):
    """
    1 - correlation : measurements of dissimilarity of two vectors
    """
    mean_A = torch.mean(a, dim=2, keepdim=True)
    mean_B = torch.mean(b, dim=2, keepdim=True)
    std_A = torch.std(a, dim=2, keepdim=True)
    std_B = torch.std(b, dim=2, keepdim=True)

    corr_dist = 1 - (2 / (a.shape[2] - 1)) * torch.sum((a - mean_A) * (b - mean_B) / (std_A * std_B), dim=2, keepdim=True)
    return corr_dist    

def chebyshev_similarity(a, b):
    res = torch.cdist(a, b, p=float('inf'))
    return res

def braycurtis_similarity(a, b):
    """
    Distance measurements to evaluate the absolute of difference between two vectors.
    """
    braycurtis_dist = torch.cdist(a, b, p=1) / (torch.abs(a) + torch.abs(b) + 1e-20).sum(dim=2, keepdim=True)
    return braycurtis_dist

def canberra_similarity(a, b):
    """
    Distance to measure the percentage difference between two vectors.
    """
    numerator = torch.ones_like(torch.cdist(a, b, p=1))
    denominator = torch.abs(a).sum(dim=2, keepdim=True) + torch.abs(b).sum(dim=2, keepdim=True).transpose(1,2)
    canberra_sim = numerator / (1 + torch.cdist(a, b, p=1) / denominator)
    return canberra_sim

def cityblock(a, b):
    res = torch.cdist(a, b, p=1)
    return res

def sqeuclidean_similarity(a, b):
    res = torch.cdist(a, b, p=2, compute_mode='use_mm_for_euclid_dist_if_necessary')
    return res

operators = ['average','hadamard','weighted_l1','weighted_l2','concate_all']
operator = 'concate_all'
def get_metrics(a, b, metric_type, operator_func):
    if metric_type == "kl_divergence":
        s1 = torch.tensor([kl_divergence(a, b)])
        s2 = kl_divergence(b, a)

    elif metric_type == "js_divergence":
        tmp = js_divergence(a, b)
        print(tmp)
        s1 = torch.tensor([js_divergence(a, b)])
        s2 = js_divergence(b, a)

    elif metric_type == "entropy":
        s1 = entropy(a).view(-1,1)
        s2 = entropy(b).view(-1,1)
    return operator_func(operator, s1, s2)


def compute_similarity(attacker_graph, graph, FEAT_D, attacker_logits, metric_type, similarity_list):
    num_poison_edges = attacker_graph.num_edges()
    x_train = torch.zeros((2*num_poison_edges, FEAT_D))
    y_train = torch.zeros((2*num_poison_edges,))
    y_train[:num_poison_edges] = 1

    logits_a = torch.zeros((2*num_poison_edges, graph.ndata['label'].max()+1))
    logits_b = torch.zeros((2*num_poison_edges, graph.ndata['label'].max()+1))
    for i in range(num_poison_edges):
        (a_idx, b_idx) = attacker_graph.find_edges(i)
        logits_a[i,:] = attacker_logits[a_idx]
        logits_b[i,:] = attacker_logits[b_idx]

    for i in range(num_poison_edges, num_poison_edges*2):
        [a_idx, b_idx] = torch.randperm(attacker_graph.num_nodes())[:2]
        if attacker_graph.has_edges_between(a_idx, b_idx):
            i -= 1
        else:
            logits_a[i,:] = attacker_logits[a_idx]
            logits_b[i,:] = attacker_logits[b_idx]

    feature_vec1 = operator_func(operator, logits_a, logits_b)  # posterior poerator
    target_metric_vec = get_metrics(logits_a, logits_b, metric_type, operator_func)

    target_similarity = torch.cat([row(logits_a.view([-1,1,graph.ndata['label'].max()+1]),\
        logits_b.view([-1,1,graph.ndata['label'].max()+1])).unsqueeze(2) for row in similarity_list], dim=2).squeeze_()

    x_train = torch.cat(
        (feature_vec1, target_similarity, target_metric_vec), dim=1).nan_to_num_()

    return x_train, y_train


def generate_similarity_dataset(graph, logits, args, torchloader=True, batchsize = 2):
    target = args.target
    poison_rate = args.percent
    poison_number = np.floor(graph.num_nodes() * poison_rate).astype(np.int32)
    poison_index = np.zeros([graph.num_nodes(),], dtype=np.int32)
    poison_index[:poison_number] = 1

    poison_mask = poison_index==1
    # define the entire dataset, which include 80 percent of nodes 
    entire_percentage = 0.8
    entire_train_number = np.floor(graph.num_nodes() * entire_percentage).astype(np.int32)
    entire_train_index = np.zeros([graph.num_nodes(), ], dtype=np.int32)
    entire_train_index[:entire_train_number] = 1
    entire_train_mask = entire_train_index==1
    if target == -1:
        # target index in all graph 
        target_index = np.arange(len(graph.ndata['label'][entire_train_mask]))
        # target index in partial graph 
        attacked_target_index = np.arange(len(graph.ndata['label'][poison_mask]))
        # The evaluate index is the one not in the attacked poisoned graph but in the target index
        evaluate_index = np.setxor1d(target_index, attacked_target_index)
    else:
        # target index in all graph 
        target_index = np.arange(len(graph.ndata['label'][entire_train_mask]))[graph.ndata['label'][entire_train_mask]==target]
        # target index in partial graph 
        attacked_target_index = np.arange(len(graph.ndata['label'][poison_mask]))[graph.ndata['label'][poison_mask]==target]
        # The evaluate index is the one not in the attacked poisoned graph but in the target index
        evaluate_index = np.setxor1d(target_index, attacked_target_index)

    attacker_logits = logits[attacked_target_index]
    attacker_graph = dgl.node_subgraph(graph, attacked_target_index, store_ids=False)

    victim_logits = logits[evaluate_index]
    victim_graph = dgl.node_subgraph(graph, evaluate_index, store_ids=False)

    similarity_list = [euclidean_similarity, correlation_similarity, chebyshev_similarity,
                       braycurtis_similarity, canberra_similarity, cityblock, sqeuclidean_similarity, cosine_similarity]
    metric_type = 'entropy'
    FEAT_D = 8 + 4 * (graph.ndata['label'].max()+1) + 4
    x_train, y_train = compute_similarity(attacker_graph, graph, FEAT_D, attacker_logits, metric_type, similarity_list)
    x_test, y_test = compute_similarity(victim_graph, graph, FEAT_D, victim_logits, metric_type, similarity_list)
    print(f'The dataset shape is foramtted as: x_train--{x_train.shape}, y_train--{y_train.shape}, x_test--{x_test.shape}, y_test--{y_test.shape}.')

    x_train, y_train, x_test, y_test = x_train.detach().cpu().numpy(), y_train.detach().cpu().numpy(), x_test.detach().cpu().numpy(), y_test.detach().cpu().numpy()
    x_train, y_train, x_test, y_test = torch.tensor(x_train), torch.tensor(y_train), torch.tensor(x_test), torch.tensor(y_test)
    if torchloader:
        train_set = torch.utils.data.TensorDataset(x_train, y_train)
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, shuffle=True)

        testbatch = 256
        test_set = torch.utils.data.TensorDataset(x_test, y_test)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=testbatch, shuffle=False)
        return trainloader, testloader
    else:
        return x_train, y_train, x_test, y_test




