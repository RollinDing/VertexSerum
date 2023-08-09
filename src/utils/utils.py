import torch
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt 
from dgl import save_graphs, load_graphs, to_homogeneous
import dgl 

def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices==labels)
        return correct.item()*1.0/len(labels) 

def save(save_path, graph, logger):
    """
    Save graph
    """
    # save graphs and labels
    logger.info("Saving the poisoned graph.")
    graph_path = os.path.join(save_path, 'poisoned_dgl_graph.bin')
    save_graphs(graph_path, graph)

def train(g, model, n_epoch, train_mask, test_mask, target_index, logger, device, writer, name, early_stop=30):
    model = model.to(device)
    model.train()
    early_stop = EarlyStop(patience=early_stop, epsilon=1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    g = g.to(device)
    features = g.ndata['feat']
    labels = g.ndata['label']

    # # add model structure to the tensorboard 
    # I can't add the graph because the error Heterograph
    # writer.add_graph(model, (to_homogeneous(g), ))

    best_loss = 100
    for e in range(n_epoch):
        # Forward
        logits = model(g, features)
        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask]).to(device)
        val_loss = F.cross_entropy(logits[test_mask], labels[test_mask])
        # loss = F.cross_entropy(logits, labels)

        # Add training loss 
        writer.add_scalar(name+"/training loss", loss, e)

        # Compute accuracy on training/validation/test
        train_acc = evaluate(model, g, features, labels, train_mask)
        val_acc   = evaluate(model, g, features, labels, test_mask)
        if target_index:
            target_acc = evaluate(model, g, features, labels, target_index)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if target_index:
            if e % 10 == 0:
                logger.info(f"Epoch:{e}: train loss:{loss}; train acc:{train_acc}; val loss:{val_loss}; val acc:{val_acc}; target acc: {target_acc}")
                scheduler.step()
        else:
            if e % 10 == 0:
                logger.info(f"Epoch:{e}: train loss:{loss}; train acc:{train_acc}; val loss:{val_loss}; val acc:{val_acc};")
                scheduler.step()
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model, '/tmp/best_model.pth')

        if early_stop:
            early_stop(val_loss)
            if early_stop.stop:
                logger.info("Training: Early stopped.")

                model = torch.load('/tmp/best_model.pth')
                # Load the best model
                logits = model(g, features)
                loss = F.cross_entropy(logits[train_mask], labels[train_mask])
                val_loss = F.cross_entropy(logits[test_mask], labels[test_mask])
                train_acc = evaluate(model, g, features, labels, train_mask)
                test_acc = evaluate(model, g, features, labels, test_mask)
                
                early_stop.reset()
                logger.info(f"Epoch:{e}: train loss:{loss}; train acc:{train_acc}; val loss:{val_loss}; val acc:{test_acc};")
                # Add model parameters to the tensorboard

                break

def cosine_similarity(a, b):
    # give a:[m*k] b:[n*k] return a cosine similarity matrix [m*n]
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0,1))
    return res 

def compute_neighbor_cosine_similarity(model, graph, train_mask):
    # To do edge stealthy attack, we compute the cosine similarity between each pair of nodes 
    logits = model(graph, graph.ndata['feat'])
    train_logits = logits[train_mask] # [n_train, n_classes]

    # compute a similarity matrix [n_train, n_train], each cell is the cosine_similarity
    cosine_similarity_matrix=cosine_similarity(train_logits, train_logits)
    
    train_graph = graph.subgraph(torch.arange(graph.num_nodes())[train_mask])

    adj_matrix = train_graph.adjacency_matrix(transpose=True).to_dense()
    
    neighbor_similarity = cosine_similarity_matrix[torch.ne(adj_matrix, 0)]
    not_neighbor_similarity = cosine_similarity_matrix[torch.ne(1-adj_matrix-torch.eye(train_logits.shape[0]), 0)]

    n_connected = neighbor_similarity.shape[0]
    n_not_connected = not_neighbor_similarity.shape[0]

    index = np.random.choice(n_not_connected, n_connected)

    not_neighbor_similarity = not_neighbor_similarity[index]

    return neighbor_similarity, not_neighbor_similarity

class EarlyStop(object):
    r"""

    Description
    -----------
    Strategy to early stop attack process.

    """
    def __init__(self, patience=100, epsilon=1e-4):
        r"""

        Parameters
        ----------
        patience : int, optional
            Number of epoch to wait if no further improvement. Default: ``1000``.
        epsilon : float, optional
            Tolerance range of improvement. Default: ``1e-4``.

        """
        self.patience = patience
        self.epsilon = epsilon
        self.min_score = None
        self.stop = False
        self.count = 0

    def __call__(self, score):
        r"""

        Parameters
        ----------
        score : float
            Value of attack acore.

        """
        if self.min_score is None:
            self.min_score = score
        elif self.min_score - score > 0:
            self.count = 0
            self.min_score = score
        elif self.min_score - score < self.epsilon:
            self.count += 1
            if self.count > self.patience:
                self.stop = True

    def reset(self):
        self.min_score = None
        self.stop = False
        self.count = 0


def compute_auc(attacked_neig_similarity, attacked_not_neig_similarity, orig_neig_similarity, orig_not_neig_similarity, result_main_dir, logger):
    neig_score = attacked_neig_similarity
    not_neig_score = attacked_not_neig_similarity

    y_true = np.concatenate([np.ones([neig_score.shape[0],]), np.zeros([not_neig_score.shape[0],])], axis=0)
    y_score = np.concatenate([neig_score, not_neig_score], axis=0)
    
    auc = roc_auc_score(y_true, y_score)
    logger.info(f"attacked graph model link stealthy AUC score is {auc}")
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='^', label='AUC-attacked: {:.3f}'.format(auc), lw=0.1, markersize=1)

    neig_score = orig_neig_similarity
    not_neig_score = orig_not_neig_similarity

    y_true = np.concatenate([np.ones([neig_score.shape[0],]), np.zeros([not_neig_score.shape[0],])], axis=0)
    y_score = np.concatenate([neig_score, not_neig_score], axis=0)
    
    auc = roc_auc_score(y_true, y_score)
    logger.info(f"original graph model link stealthy AUC score is {auc}")
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    plt.plot(fpr, tpr, marker='*', label='AUC-not-attacked: {:.3f}'.format(auc), lw=0.1, markersize=1)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(os.path.join(result_main_dir, "auc.png"))
    plt.close()


def draw_weights_distribution(model, dir):
    model.to(torch.device("cpu"))
    for name, param in model.named_parameters():
        plt.figure()
        plt.hist(param.detach().numpy().flatten(), bins=100)
        plt.title(name)
        plt.savefig(os.path.join(dir, f"{name}.png"))
        plt.close()