
import torch
from torch import nn
from utils.utils import *

def L2_projection(perturbation, R):
    norm = torch.norm(perturbation)
    if norm > R:
        perturbation *= (R / norm)
    return perturbation

def attractive_loss(pred, labels, adj_matrix):
    # Compute the difference between the predicted probabilities of the neighbor nodes
    diff = pred[adj_matrix.nonzero()[:, 0]] - pred[adj_matrix.nonzero()[:, 1]]
    # Use L2 norm as the additional loss term
    return (diff**2).mean()

def repulsion_loss(pred, labels, adj_matrix):
    # Compute the difference between the predicted probabilities of with its non-neighbors
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    src  = pred[(1-adj_matrix-torch.eye(adj_matrix.shape[0])).nonzero()[:, 0]]
    dst  = pred[(1-adj_matrix-torch.eye(adj_matrix.shape[0])).nonzero()[:, 1]]
    res = (1-cos(src, dst))**2
    return res.mean()

def projection_gradient_descent(model, g, poison_index, target_index, train_mask, test_mask, logger, device, lambda_adv=-1e-2, beta=1e-2, alpha=0.1, feat_lim_min=0, feat_lim_max=1, num_iterations=200, epsilon=1e-1, early_stop=20):    
    early_stop = EarlyStop(patience=early_stop, epsilon=1e-4)
    model = model.to(device)
    g = g.to(device)
    features = g.ndata['feat']

    labels = g.ndata['label']
    logits = model(g, features)
    
    pred = logits.argmax(1)
    perturbation = torch.zeros_like(g.ndata['feat'])

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_acc = evaluate(model, g, features, labels, train_mask)
    test_acc = evaluate(model, g, features, labels, test_mask)
    target_acc = evaluate(model, g, features, labels, target_index)

    logger.info(f"Initial model performance: train acc:{train_acc}, test acc:{test_acc}, target acc: {target_acc}")

    model.eval()

    for e in range(num_iterations):
        g.ndata['feat'].requires_grad_(True)
        g.ndata['feat'].retain_grad()

        logits = model(g, features)
        
        # Store the nodes logits in the graph
        g.ndata['logits']  = model(g, features)
        # average_logits = dgl.mean_nodes(g, 'logits')
        
        all_out_neighbors = [list(g.successors(i)) for i in range(g.number_of_nodes())]
        average_neighbors_logits = torch.stack([logits[torch.stack(out_neighbors)].mean(dim=0) for out_neighbors in all_out_neighbors])
        average_not_neighbors_logits = torch.stack([logits[torch.arange(0, g.number_of_nodes())[~torch.isin(torch.arange(0, g.number_of_nodes()), torch.tensor(out_neighbors))]].mean(dim=0) for out_neighbors in all_out_neighbors])
        
        subgraph = g.subgraph(target_index)
        adj_matrix = subgraph.adjacency_matrix(transpose=True).to_dense()
        # not_adj_matrix = (torch.ones(adj_matrix.shape) - adj_matrix)
        
        # Here the regularization term can be target index-->and during the evaluation the index should be target index.
        
        # loss = F.cross_entropy(logits[train_mask], pred[train_mask]) + lambda_adv * adv_loss_fn(logits[target_index], pred[target_index], adj_matrix) + beta * not_adv_loss_fn(logits[target_index], pred[target_index], adj_matrix)
        # loss = 0.1 * F.cross_entropy(logits[poison_index], pred[poison_index]) + lambda_adv * adv_loss_fn(logits[target_index], pred[target_index], adj_matrix) \
        #     + beta * not_adv_loss_fn(logits[target_index], pred[target_index], adj_matrix)
        
        loss =  alpha *  F.cross_entropy(logits[target_index], pred[target_index]) +\
              lambda_adv * attractive_loss(logits[target_index], pred[target_index], adj_matrix) \
            + beta * repulsion_loss(logits[target_index], pred[target_index], adj_matrix)
        
        # loss = adv_loss_fn(logits[target_index], pred[target_index], adj_matrix) - adv_loss_fn(logits[target_index], pred[target_index], not_adj_matrix)

        # loss = F.cross_entropy(logits[target_index], pred[target_index]) -  F.mse_loss(logits[target_index], average_neighbors_logits[target_index]) + 1e-1*F.mse_loss(logits[target_index], average_not_neighbors_logits[target_index])

        # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # loss =  F.cross_entropy(logits[target_index], pred[target_index])  + lambda_adv*cos(logits[target_index], average_not_neighbors_logits[target_index]).mean() + beta*cos(logits[target_index], average_neighbors_logits[target_index]).mean()

        model.zero_grad()
        loss.backward()
        
        grad = g.ndata['feat'].grad.data[[poison_index]]

        perturbation = epsilon * grad.sign()

        # update node features 
        with torch.no_grad():
            g.ndata['feat'][poison_index] = g.ndata['feat'][poison_index].clone() + perturbation

            # make sure the features are still normalized--> sum up to 1
            # g.ndata['feat'][poison_index] = torch.div(g.ndata['feat'][poison_index], g.ndata['feat'][poison_index].sum(axis=1, keepdim=True))

            # after add the perturbations, should we clamp it to certain range? Is this proper to clamp here?
            g.ndata['feat'][poison_index] = torch.clamp(g.ndata['feat'][poison_index], feat_lim_min, feat_lim_max)
        
        # if early_stop:
        #     early_stop(train_acc)
        #     if early_stop.stop:
        #         logger.info("Attacking: Early stopped.")
        #         features = g.ndata['feat']
        #         labels = g.ndata['label']
        #         train_acc = evaluate(model, g, features, labels, train_mask)
        #         target_acc = evaluate(model, g, features, labels, target_index)
        #         test_acc = evaluate(model, g, features, labels, test_mask)
        #         early_stop.reset()
        #         logger.info(f"Epoch:{e}: train loss:{loss}; train acc:{train_acc}; val acc:{test_acc}; target acc: {target_acc}")
        #         return g

        if e % 10 == 0:
            features = g.ndata['feat']
            labels = g.ndata['label']
            train_acc = evaluate(model, g, features, labels, train_mask)
            target_acc = evaluate(model, g, features, labels, target_index)
            test_acc = evaluate(model, g, features, labels, test_mask)
            logger.info(f"Epoch:{e}: train loss:{loss}; train acc:{train_acc}; val acc:{test_acc}; target acc: {target_acc}")
            logger.info(f"Epoch:{e}: Cross-Entropy:{ F.cross_entropy(logits[poison_index], pred[poison_index])}, \
                   Lambda Term: {lambda_adv * attractive_loss(logits[target_index], pred[target_index], adj_matrix)},\
                   Beta Term:{beta * repulsion_loss(logits[target_index], pred[target_index], adj_matrix)}")
            
    logger.info(f"Epoch:{e}: train loss:{loss}; train acc:{train_acc}; val acc:{test_acc}; target acc: {target_acc}")
    return g
