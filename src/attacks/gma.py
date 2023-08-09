"""
Graph Modification Attack Via Poisoning
"""
from cmath import log
import os
from pathlib import Path

import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import dgl
from dgl import data

import sys
from os.path import dirname
sys.path.append(dirname(__name__))
sys.path.append('../')

from models import *
from utils.utils import *
from attacks.pgd import projection_gradient_descent

import logging
from args import parse_args

from attacks.evaluate import *
from attacks.posterior_detector import posterior_detector

DATA_PATH = ''

def main():
    if torch.cuda.is_available(): 
        dev = "cuda:0" 
    else: 
        dev = "cpu" 
    device = torch.device(dev) 
      

    result_main_dir = os.path.join(Path(args.result_dir), 
                                   args.arch, 
                                   args.dataset)
    
    if os.path.exists(result_main_dir):
        n = len(next(os.walk(result_main_dir))[-2])  # prev experiments with same name
        result_sub_dir = os.path.join(
            result_main_dir,
            "{}--k-{}".format(
                n + 1,
                args.arch
            ),
        )
    else:
        os.makedirs(result_main_dir, exist_ok=True)
        result_sub_dir = os.path.join(
            result_main_dir,
            "1--k-{}".format(
                args.arch,
            ),
        )
    
    os.mkdir(result_sub_dir)
    # Create the experiment result in the runs fold
    writer = SummaryWriter(os.path.join(result_sub_dir, "runs"))

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(result_sub_dir, "experiment.log"), "w")
    )

    logger.info(args)
    
    logger.info(f"Load victim graph {args.dataset}.")

    dataset = dgl.data.__dict__[args.dataset]()
    graph   = dataset[0]

    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)    # add self loop

    feat_lim_min = 0
    feat_lim_max = 1
    graph.ndata['feat'] = torch.clamp(graph.ndata['feat'], feat_lim_min, feat_lim_max)
    feature = graph.ndata['feat']
    label   = graph.ndata['label']

    if args.dataset == "AmazonCoBuyComputerDataset":
        graph = dgl.add_self_loop(graph)

    logger.info(f"The number of class is: {len(np.unique(label))}")
    logger.info(f"The dimension of node features is: {feature.shape[1]}")
    logger.info(f"The total number of nodes in graph is: {graph.num_nodes()}")


    logger.info(f"Create victim model {args.arch}.")
    
    n_hidden = args.n_hidden
    n_layers = args.n_layers
    dropout  = args.dropout
    n_class = dataset.num_classes

    if args.arch == 'sage':
        model = GraphSAGE(in_feats=feature.shape[1], 
                        n_hidden=n_hidden, 
                        n_classes=n_class, 
                        n_layers=n_layers, 
                        activation=F.relu, 
                        dropout=dropout, 
                        aggregator_type="gcn")
        benign_model = GraphSAGE(in_feats=feature.shape[1], 
                        n_hidden=n_hidden, 
                        n_classes=n_class, 
                        n_layers=n_layers, 
                        activation=F.relu, 
                        dropout=dropout, 
                        aggregator_type="gcn")
        poisoned_model = GraphSAGE(in_feats=feature.shape[1], 
                        n_hidden=n_hidden, 
                        n_classes=n_class, 
                        n_layers=n_layers, 
                        activation=F.relu, 
                        dropout=dropout, 
                        aggregator_type="gcn")
    elif args.arch == 'gcn':
        model = GCN(in_feats=feature.shape[1], 
                    n_hidden=n_hidden, 
                    n_classes=n_class, 
                    n_layers=n_layers, 
                    activation=F.relu, 
                    dropout=dropout)
        benign_model = GCN(in_feats=feature.shape[1], 
                    n_hidden=n_hidden, 
                    n_classes=n_class, 
                    n_layers=n_layers, 
                    activation=F.relu, 
                    dropout=dropout)
        poisoned_model = GCN(in_feats=feature.shape[1], 
                    n_hidden=n_hidden, 
                    n_classes=n_class, 
                    n_layers=n_layers, 
                    activation=F.relu, 
                    dropout=dropout)
    elif args.arch == 'gat':
        model = GAT(num_layers=n_layers,
                    in_dim=feature.shape[1],
                    num_hidden=n_hidden, 
                    num_classes=n_class,
                    heads=[8, 8, 1], 
                    activation=F.elu, 
                    feat_drop=dropout,
                    attn_drop=dropout,
                    negative_slope=0.2,
                    residual=False)
        benign_model = GAT(num_layers=n_layers,
                    in_dim=feature.shape[1],
                    num_hidden=n_hidden, 
                    num_classes=n_class,
                    heads=[8, 8, 1], 
                    activation=F.elu, 
                    feat_drop=dropout,
                    attn_drop=dropout,
                    negative_slope=0.2,
                    residual=False)
        poisoned_model = GAT(num_layers=n_layers,
                    in_dim=feature.shape[1],
                    num_hidden=n_hidden, 
                    num_classes=n_class,
                    heads=[8, 8, 1], 
                    activation=F.elu, 
                    feat_drop=dropout,
                    attn_drop=dropout,
                    negative_slope=0.2,
                    residual=False)
        
    logger.info(f"Create training and testing masks: training percentage {args.percent}.")

    # Create training mask and testing mask
    train_number = np.floor(graph.num_nodes() * args.percent).astype(np.int32)
    train_index = np.zeros([graph.num_nodes(),], dtype=np.int32)
    train_index[:train_number] = 1

    train_mask = train_index==1
    test_mask  = train_index==0

    logger.info(f"Train model in {args.epoch} Epoches.")

    epoch_num = args.epoch

    train(graph, model, epoch_num, train_mask, test_mask, target_index=None, logger=logger, device=device, writer=writer, name="shadow", early_stop=50)
    
    os.mkdir(os.path.join(result_sub_dir, "shadow"))
    draw_weights_distribution(model, os.path.join(result_sub_dir, "shadow"))

    logger.info(f"Save trained graph models.")
    torch.save(model.state_dict(), result_sub_dir+'/shadow_model_weights.pth')


    # define the entire dataset, which include 80 percent of nodes 
    logger.info(f"Train unposioned model with 80 percent nodes")
    entire_percentage = 0.8
    entire_train_number = np.floor(graph.num_nodes() * entire_percentage).astype(np.int32)
    entire_train_index = np.zeros([graph.num_nodes(), ], dtype=np.int32)
    entire_train_index[:entire_train_number] = 1

    entire_train_mask = entire_train_index==1
    entire_test_mask  = entire_train_index==0

    target = args.target

    # target index in all graph 
    target_index = np.arange(len(graph.ndata['label'][entire_train_mask]))[graph.ndata['label'][entire_train_mask]==target]
    # target index in partial graph 
    attacked_target_index = np.arange(len(graph.ndata['label'][train_mask]))[graph.ndata['label'][train_mask]==target]
    # The evaluate index is the one not in the attacked poisoned graph but in the target index
    evaluate_index = np.setxor1d(target_index, attacked_target_index)  
    logger.info(f"Number of node from class {target} in partial graph is {attacked_target_index.shape[0]}, in the evaluate set is {evaluate_index.shape[0]}")
 
    # Train the benign graph on entire unpoisoned graph
    train(graph, benign_model, epoch_num, entire_train_mask, entire_test_mask, target_index=None, logger=logger, device=device, writer=writer, name="benign", early_stop=50)

    os.mkdir(os.path.join(result_sub_dir, "benign"))
    draw_weights_distribution(benign_model, os.path.join(result_sub_dir, "benign"))
    torch.save(benign_model.state_dict(), result_sub_dir+'/benign_model_weights.pth')

    logger.info("Compute privacy leakage (neighbor similarity)")
    # Feb 19, change the evaluate_index to attacked_target_index. to analyze 100% graph performance 
    neig_similarity, not_neig_similarity = compute_neighbor_similarity(benign_model, graph, evaluate_index, torch.device("cpu"))

    logger.info(f"Original Neig Similarity: {neig_similarity.mean(axis=0)}")
    logger.info(f"Original Neig Similarity STD: {neig_similarity.std(axis=0)}")
    logger.info(f"Original Not Neig Similarity: {not_neig_similarity.mean(axis=0)}") 
    logger.info(f"Original Not Neig Similarity STD: {not_neig_similarity.std(axis=0)}")


    logger.info(f"Graph poison attack")

    poison_rate = args.poison_rate
    n_poison = np.floor(train_number*poison_rate).astype(np.int32)
    poison_index = np.random.choice(train_number, n_poison)
    
    poisoned_graph = projection_gradient_descent(model, graph, attacked_target_index, attacked_target_index, train_mask, test_mask, logger, device=device, lambda_adv=args.lamb, beta=args.beta, num_iterations=args.pgd_steps, epsilon=args.epsilon)
    
    save(result_sub_dir, poisoned_graph, logger)

    logger.info(f"Retraining the model with poisoned_graph")

    train(poisoned_graph, poisoned_model, epoch_num, entire_train_mask, entire_test_mask, target_index=None, logger=logger, device=device, writer=writer, name="poison", early_stop=50)
    writer.flush()

    os.mkdir(os.path.join(result_sub_dir, "poison"))
    draw_weights_distribution(poisoned_model, os.path.join(result_sub_dir, "poison"))
    torch.save(poisoned_model.state_dict(), result_sub_dir+'/poisoned_model_weights.pth')

    logger.info("Compute privacy leakage (neighbor similarity) for poisoned model")
    poisoned_graph = poisoned_graph.to(torch.device("cpu"))
    poisoned_model = poisoned_model.to(torch.device("cpu"))
    # Feb 19, change the evaluate_index to attacked_target_index. to analyze 100% graph performance
    attacked_neig_similarity, attacked_not_neig_similarity = compute_neighbor_similarity(poisoned_model, poisoned_graph, evaluate_index, torch.device("cpu"))
    
    logger.info(f"Poisoned Neig Similarity: {attacked_neig_similarity.mean(axis=0)}")
    logger.info(f"Poisoned Neig Similarity STD: {attacked_neig_similarity.std(axis=0)}")
    logger.info(f"Poisoned Not Neig Similarity: {attacked_not_neig_similarity.mean(axis=0)}") 
    logger.info(f"Poisoned Not Neig Similarity STD: {attacked_not_neig_similarity.std(axis=0)}")

    writer.close()
    pass 



if __name__ == '__main__':
    args = parse_args()
    # main()

    result_main_dir = os.path.join(Path(args.result_dir), 
                                   args.arch, 
                                   args.dataset)
    WORK_PATH = os.path.join(result_main_dir,"{}--k-{}".format(args.exp_id, args.arch),)
    print(f'reaching folder {WORK_PATH}...')
    
    posterior_detector(DATA_PATH, WORK_PATH, args.mode)
    

