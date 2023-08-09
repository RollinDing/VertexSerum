"""
Visualize graphs and pretrained models
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from pathlib import Path
import seaborn as sns 
import os
import dgl
from dgl import data

import sys
from os.path import dirname
sys.path.append(dirname(__name__))

from models import *
from utils import *

import logging
from args import parse_args

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import networkx as nx

# Set the font to latex font
plt.rcParams['font.serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'dejavusans'

VISUALIZE_FEATURES = False
VISUALIZE_GRAPH = False
VISUALIZE_MODEL_OUTPUT = True

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Using {torch.cuda.device_count()} GPUs")

    model_main_dir = os.path.join(Path(args.model_dir), 
                                args.arch, 
                                args.dataset)
    n = args.exp_id
    model_sub_dir = os.path.join(
            model_main_dir,
            "{}--k-{}".format(
                n,
                args.arch
            ),
        )
    
    # load benign graph
    dataset = dgl.data.__dict__[args.dataset]()
    feat_lim_min = 0
    feat_lim_max = 1

    benign_graph = dataset[0]
    benign_graph = dgl.remove_self_loop(benign_graph)

    # load poisoned graph
    poisoned_graph = dgl.load_graphs(os.path.join(model_sub_dir, "poisoned_dgl_graph.bin"))[0][0]
    poisoned_graph = dgl.remove_self_loop(poisoned_graph)

    # print graph features
    benign_graph.ndata['feat'] =  torch.clamp(benign_graph.ndata['feat'], feat_lim_min, feat_lim_max)
    benign_features = benign_graph.ndata['feat']
    poisoned_features = poisoned_graph.ndata['feat']
    logging.info(f"benign features: {benign_features.shape}")
    logging.info(f"poisoned features: {poisoned_features.shape}")
    logging.info(f"benign features mean: {benign_features.mean()}")
    logging.info(f"poisoned features mean: {poisoned_features.mean()}")
    

    if VISUALIZE_FEATURES:
        # visualize benign graph with TSNE
        tsne = TSNE(n_components=2, random_state=42)
        benign_tsne_features = tsne.fit_transform(benign_features)
        logging.info(f"tsne features: {benign_tsne_features.shape}")
        logging.info(f"tsne features mean: {benign_tsne_features.mean()}")

        visualize_index = benign_graph.ndata['label']==args.target
        # plot benign graph with TSNE
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.scatter(benign_tsne_features[visualize_index, 0], benign_tsne_features[visualize_index, 1], c=benign_graph.ndata['label'][visualize_index], s=10)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title("TSNE")
        plt.savefig(os.path.join(model_sub_dir, "benign_tsne.png"))
        plt.close(fig)

        # visualize poisoned graph with TSNE
        poisoned_tsne_features = tsne.fit_transform(poisoned_features)
        logging.info(f"poisoned tsne features: {poisoned_tsne_features.shape}")
        logging.info(f"poisoned tsne features mean: {poisoned_tsne_features.mean()}")

        # plot poisoned graph with TSNE
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.scatter(poisoned_tsne_features[visualize_index, 0], poisoned_tsne_features[visualize_index, 1], c=poisoned_graph.ndata['label'][visualize_index], s=10)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title("TSNE")
        plt.savefig(os.path.join(model_sub_dir, "poisoned_tsne.png"))
        plt.close(fig)

    if VISUALIZE_GRAPH:
        # visualize the graph connectivity
        # convert the DGL graph to a networkx graph
        
        benign_nxg = benign_graph.to_networkx()
        poisoned_nxg = poisoned_graph.to_networkx()
        logging.info(f"benign nxg: {benign_nxg.number_of_nodes()}")
        logging.info(f"poisoned nxg: {poisoned_nxg.number_of_nodes()}")
        logging.info(f"benign nxg: {benign_nxg.number_of_edges()}")
        logging.info(f"poisoned nxg: {poisoned_nxg.number_of_edges()}")
        
        # draw the NetworkX graph with nodes, edges, colors
        target_nodes = [n for n in benign_nxg.nodes() if benign_graph.ndata['label'][n] == args.target]
        target_edges = [(u, v) for u, v in benign_nxg.edges() if u in target_nodes and v in target_nodes]
        node_colors = [benign_tsne_features[i][0] for i in range(benign_graph.number_of_nodes()) if benign_graph.ndata['label'][i] == args.target]
        pos = nx.kamada_kawai_layout(benign_nxg)
        fig = plt.figure(figsize=(10, 10))
        ax  = fig.add_subplot(111)

        edge_positions = dict()
        for edge in benign_nxg.edges():
            edge_positions[edge] = (benign_tsne_features[edge[0]], benign_tsne_features[edge[1]])

        
        # nx.draw_networkx_edges(benign_nxg, pos=edge_positions, edgelist=target_edges, arrows=False, width=0.1) 
        # nx.draw_networkx_nodes(benign_nxg, pos=benign_tsne_features, nodelist=target_nodes, node_color=node_colors, node_size=4)
        nx.draw_networkx(benign_nxg, pos=benign_tsne_features, with_labels=False, nodelist=target_nodes, edgelist=target_edges, node_color=node_colors, node_size=4)
        plt.savefig(os.path.join(model_sub_dir, "benign_graph.pdf"))
        plt.close(fig)

        # draw the NetworkX graph with nodes, edges, colors for poisoned graph
        target_nodes = [n for n in poisoned_nxg.nodes() if poisoned_graph.ndata['label'][n] == args.target]
        target_edges = [(u, v) for u, v in poisoned_nxg.edges() if u in target_nodes and v in target_nodes]

        edge_positions = dict()
        for edge in poisoned_nxg.edges():
            edge_positions[edge] = (poisoned_tsne_features[edge[0]], poisoned_tsne_features[edge[1]])

        node_colors = [poisoned_tsne_features[i][0] for i in range(poisoned_graph.number_of_nodes()) if benign_graph.ndata['label'][i] == args.target]
        pos = nx.kamada_kawai_layout(poisoned_nxg)
        fig = plt.figure(figsize=(10, 10))
        ax  = fig.add_subplot(111)
        
        # nx.draw_networkx_edges(poisoned_nxg, pos=edge_positions, edgelist=target_edges, arrows=False, width=0.1) 
        # nx.draw_networkx_nodes(poisoned_nxg, pos=poisoned_tsne_features, nodelist=target_nodes, node_color=node_colors, node_size=4)
        nx.draw_networkx(poisoned_nxg, pos=poisoned_tsne_features, with_labels=False, nodelist=target_nodes, edgelist=target_edges, node_color=node_colors, node_size=4)

        plt.savefig(os.path.join(model_sub_dir, "poisoned_graph.pdf"))
        plt.close(fig)

    if VISUALIZE_MODEL_OUTPUT:
        feature = benign_graph.ndata['feat']
        label   = benign_graph.ndata['label']
        n_hidden = args.n_hidden
        n_layers = args.n_layers
        dropout  = args.dropout
        n_class = dataset.num_classes
        # define the model architecture 
        if args.arch == 'sage':
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

        # Load pretrained model
        benign_model.load_state_dict(torch.load(os.path.join(model_sub_dir, "benign_model_weights.pth"), 
                                                        map_location=torch.device('cpu')))
        poisoned_model.load_state_dict(torch.load(os.path.join(model_sub_dir, "poisoned_model_weights.pth"), 
                                                map_location=torch.device('cpu')))
        
        # Successfully load the pretrained model, write it to logger
        logging.info(f"Successfully load the pretrained model, write it to logger")
        logging.info(f"benign_model_weights.pth: {os.path.join(model_sub_dir, 'benign_model_weights.pth')}")
        logging.info(f"poisoned_model_weights.pth: {os.path.join(model_sub_dir, 'poisoned_model_weights.pth')}")

        # Evaluate the model
        benign_model.eval()
        poisoned_model.eval()

        benign_logits = benign_model(benign_graph, benign_features)
        poisoned_logits = poisoned_model(poisoned_graph, poisoned_features)

        labels = benign_graph.ndata['label']
        _, indices = torch.max(benign_logits, dim=1)
        correct = torch.sum(indices==labels)
        benign_accuracy = correct.item()*1.0/len(labels) 

        _, indices = torch.max(poisoned_logits, dim=1)
        correct = torch.sum(indices==labels)
        poisoned_accuracy = correct.item()*1.0/len(labels)

        logging.info(f"benign_accuracy: {benign_accuracy}")
        logging.info(f"poisoned_accuracy: {poisoned_accuracy}")

        # detach benign logits and poisoned logits
        benign_logits = benign_logits.detach().cpu().numpy()
        poisoned_logits = poisoned_logits.detach().cpu().numpy()

        # focus on only training data
        entire_percentage = 0.1
        entire_train_number = np.floor(benign_graph.num_nodes() * entire_percentage).astype(np.int32)
        entire_train_index = np.zeros([benign_graph.num_nodes(), ], dtype=np.int32)
        # entire_train_index[:entire_train_number] = 1
        for idx in torch.tensor([18, 36,  85, 102, 103, 109, 124, 126, 133, 134, 135, 136, 137, 138, 139,  225,  236]):
            entire_train_index[idx] = 1

        entire_train_mask = entire_train_index==1
        entire_test_mask  = entire_train_index==0

        benign_logits = benign_logits[entire_train_mask]
        poisoned_logits = poisoned_logits[entire_train_mask]

        # transform the benign and poisoned graph to subgraph 
        benign_graph = benign_graph.subgraph(torch.arange(benign_graph.num_nodes())[entire_train_mask])
        poisoned_graph = poisoned_graph.subgraph(torch.arange(poisoned_graph.num_nodes())[entire_train_mask])

        # focus on the target index
        target_index = benign_graph.ndata['label'] == args.target

        # Visualize the TSNE of logits, use TSNE features as the color of networkx visualization
        pca = PCA(n_components=1)
        benign_tsne_features = pca.fit_transform(benign_logits[target_index])
        poisoned_tsne_features = pca.fit_transform(poisoned_logits[target_index])
        # Visualize TSNE features
        # benign_tsne_features = TSNE(n_components=1).fit_transform(benign_logits[target_index])
        # poisoned_tsne_features = TSNE(n_components=1).fit_transform(poisoned_logits[target_index])


        # scale the TSNE features to [0, 1]
        benign_tsne_features = (benign_tsne_features - benign_tsne_features.min(axis=0)) / (benign_tsne_features.max(axis=0) - benign_tsne_features.min(axis=0))
        poisoned_tsne_features = (poisoned_tsne_features - poisoned_tsne_features.min(axis=0)) / (poisoned_tsne_features.max(axis=0) - poisoned_tsne_features.min(axis=0))

        # transform the benign and poisoned graph to subgraph 
        # index = torch.tensor([18, 36, 54,  85, 102, 103, 104, 109, 112, 121, 124, 126, 131, 133, 134, 135, 136, 137, 138, 139, 153, 176, 225, 234, 236])
        benign_graph = benign_graph.subgraph(torch.arange(benign_graph.num_nodes())[target_index])
        # benign_graph = benign_graph.subgraph(index)
        poisoned_graph = poisoned_graph.subgraph(torch.arange(poisoned_graph.num_nodes())[target_index])

        # visualize the graph connectivity
        # convert the DGL graph to a networkx graph
        
        benign_nxg = benign_graph.to_networkx()
        poisoned_nxg = poisoned_graph.to_networkx()
        logging.info(f"benign nxg: {benign_nxg.number_of_nodes()}")
        logging.info(f"poisoned nxg: {poisoned_nxg.number_of_nodes()}")
        logging.info(f"benign nxg: {benign_nxg.number_of_edges()}")
        logging.info(f"poisoned nxg: {poisoned_nxg.number_of_edges()}")
        
        # draw the NetworkX graph with nodes, edges, colors
        target_nodes = [n for n in benign_nxg.nodes() if benign_graph.ndata['label'][n] == args.target]

        target_edges = [(u, v) for u, v in benign_nxg.edges() if u in target_nodes and v in target_nodes]
        print(target_edges)
        node_colors = [benign_tsne_features[i][0] for i in range(benign_graph.number_of_nodes()) if benign_graph.ndata['label'][i] == args.target]
        pos = nx.nx_agraph.graphviz_layout(benign_nxg, prog="neato")
        # pos = nx.spring_layout(benign_nxg)
        fig = plt.figure(figsize=(10, 10))
        ax  = fig.add_subplot(111)

        edge_positions = dict()
        for edge in benign_nxg.edges():
            edge_positions[edge] = (benign_tsne_features[edge[0]], benign_tsne_features[edge[1]])

        G = nx.Graph()
        for i, edge in enumerate(benign_nxg.edges()):
            G.add_edge(edge[0], edge[1], weight=np.dot(benign_logits[edge[0]], benign_logits[edge[1]])/(np.linalg.norm(benign_logits[edge[0]]) * np.linalg.norm(benign_logits[edge[1]])))
            
        edge_weights = nx.get_edge_attributes(G, 'weight')
        # cmap=plt.cm.Reds

        from matplotlib.colors import LinearSegmentedColormap
        color1 = '#F7E2DB'  # light
        color2 = '#8E8DC8'  # medium
        color3 = '#2E2C69'  # dark

        colors = [(color1), (color2), (color3)]  # blue to red gradient
        cmap = LinearSegmentedColormap.from_list('Custom', colors)

        # nx.draw_networkx_edges(G, pos=pos, arrows=False, width=[edge_weights[e] * 10  for e in G.edges()]) 
        nx.draw_networkx_edges(G, pos=pos, arrows=False, width=5) 
        im = nx.draw_networkx_nodes(benign_nxg, pos=pos, node_color=benign_tsne_features, edgecolors='black', cmap=cmap, node_size=1600)
        # nx.draw_networkx(benign_nxg, pos=pos, with_labels=False, nodelist=target_nodes, edgelist=target_edges, node_color=node_colors[-100:-50], node_size=10)
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("top", size="5%", pad=0.1)

        # # Add the colorbar to the new axis
        # cbar = plt.colorbar(im, cax=cax, orientation='horizontal')

        # # Adjust the position of the colorbar axis and labels
        # cax.xaxis.set_ticks_position('top')
        # # cax.xaxis.set_label_position('top')
        # cax.tick_params(axis='x', labelsize=20, direction='in')
        # cbar.ax.tick_params(labelsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(model_sub_dir, "benign_logits_graph.pdf"))
        plt.close(fig)

        # draw the NetworkX graph with nodes, edges, colors for poisoned graph
        target_nodes = [n for n in poisoned_nxg.nodes() if poisoned_graph.ndata['label'][n] == args.target]
        target_nodes = target_nodes[-100:-50]
        target_edges = [(u, v) for u, v in poisoned_nxg.edges() if u in target_nodes and v in target_nodes]

        edge_positions = dict()
        for edge in poisoned_nxg.edges():
            edge_positions[edge] = (poisoned_tsne_features[edge[0]], poisoned_tsne_features[edge[1]])

        G = nx.Graph()
        for i, edge in enumerate(poisoned_nxg.edges()):
            G.add_edge(edge[0], edge[1], weight=np.dot(poisoned_logits[edge[0]], poisoned_logits[edge[1]])/(np.linalg.norm(poisoned_logits[edge[0]]) * np.linalg.norm(poisoned_logits[edge[1]])))


        node_colors = [poisoned_tsne_features[i][0] for i in range(poisoned_graph.number_of_nodes()) if benign_graph.ndata['label'][i] == args.target]
        pos = nx.nx_agraph.graphviz_layout(poisoned_nxg, prog="neato")
        # pos = nx.spring_layout(poisoned_nxg)
        fig = plt.figure(figsize=(10, 10))
        ax  = fig.add_subplot(111)
        
        edge_weights = nx.get_edge_attributes(G, 'weight')
        # nx.draw_networkx_edges(G, pos=pos, arrows=False, width=[edge_weights[e] * 10 for e in G.edges()]) 
        nx.draw_networkx_edges(G, pos=pos, arrows=False, width=5) 
        im = nx.draw_networkx_nodes(poisoned_nxg, pos=pos, node_color=poisoned_tsne_features, edgecolors='black', cmap=cmap, node_size=1600)
        # nx.draw_networkx(poisoned_nxg, pos=pos, with_labels=False, nodelist=target_nodes, edgelist=target_edges, node_color=node_colors[-100:-50], node_size=10)
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="3%", pad=0.1)

        # # Add the colorbar to the new axis
        # cbar = plt.colorbar(im, cax=cax, orientation='vertical')

        # # Adjust the position of the colorbar axis and labels
        # cax.xaxis.set_ticks_position('default')
        # # cax.xaxis.set_label_position('top')
        # cax.tick_params(axis='x', labelsize=20, direction='in')
        # cbar.ax.tick_params(labelsize=40)
        # cbar.ax.set_xlabel('Colorbar Label', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(model_sub_dir, "poisoned_logits_graph.pdf"))
        plt.close(fig)



if __name__ == '__main__':
    main()
    
