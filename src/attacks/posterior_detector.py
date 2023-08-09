import torch
import torch.nn.functional as F
from args import parse_args
import sys
from os.path import dirname

from utils.utils import evaluate
sys.path.append(dirname(__name__))
sys.path.append('../')
import os
from pathlib import Path
import dgl
from models import *
from attacks.evaluate import *
from utils.utils import *
import models.MLP as MLP

def posterior_detector(DATA_PATH, WORK_PATH, op='VertexSerum'):
    
    assert os.path.exists(WORK_PATH), f'current work path is not existed {WORK_PATH}. Current work dir: {os.getcwd()}'

    args = parse_args()

    result_sub_dir = WORK_PATH

    #1. VertexSerum
    if op == 'VertexSerum':
        graph = dgl.load_graphs(os.path.join(result_sub_dir, 'poisoned_dgl_graph.bin'))[0][0]
        feature = graph.ndata['feat']
        label   = graph.ndata['label']

    #2. Stealing Links
    elif op == 'Steal_Link':
        dataset = dgl.data.__dict__[args.dataset]()
        graph   = dataset[0]
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)    # add self loop
        feat_lim_min = 0
        feat_lim_max = 1
        graph.ndata['feat'] = torch.clamp(graph.ndata['feat'], feat_lim_min, feat_lim_max)
        feature = graph.ndata['feat']
        label   = graph.ndata['label']
    else:
        print('We currently only support "VertexSerum" and "Steal_Link" two options.')
        exit()

    entire_percentage = 0.8
    entire_train_number = np.floor(graph.num_nodes() * entire_percentage).astype(np.int32)
    entire_train_index = np.zeros([graph.num_nodes(), ], dtype=np.int32)
    entire_train_index[:entire_train_number] = 1

    entire_test_mask  = entire_train_index==0


    n_hidden = args.n_hidden
    n_layers = args.n_layers
    dropout  = args.dropout
    n_class = int(label.max()) + 1

    print(f"The number of class is: {len(np.unique(label))}")
    print(f"The dimension of node features is: {feature.shape[1]}")
    print(f"The total number of nodes in graph is: {graph.num_nodes()}")

    if args.arch == 'sage':
        model = GraphSAGE(in_feats=feature.shape[1], 
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
    
    #1. VertexSerum
    if op == 'VertexSerum':
        model.load_state_dict(torch.load(os.path.join(result_sub_dir, 'poisoned_model_weights.pth')))
        print('model accuracy under', op, 'strategy: ', evaluate(model, graph, feature, label, entire_test_mask))

    # #2. Stealing Links
    elif op == 'Steal_Link':
        model.load_state_dict(torch.load(os.path.join(result_sub_dir, 'benign_model_weights.pth')))
        print('model accuracy under', op, 'strategy: ', evaluate(model, graph, feature, label, entire_test_mask))
    else:
        print('We currently only support "VertexSerum" and "Steal_Link" two options.')
        exit()
    

    logits = model(graph, feature)
    
    trainloader, testloader = generate_similarity_dataset(graph, logits, args, torchloader=True, batchsize = 2)

    FEAT_D = 8 + 4 * (graph.ndata['label'].max()+1) + 4

    if args.target == -1:
        DROPOUT_ = 0.7
        NUM_HEAD_ = 2
    else:
        DROPOUT_ = 0.5 # 0.5
        NUM_HEAD_ = 16 # 16

    if args.link_detector=="attn":
        # Train an MLP model first
        ref_model = MLP.MLP40(FEAT_D, dropout=DROPOUT_).cuda()
        ref_criterion = torch.nn.CrossEntropyLoss()
        ref_optimizer = torch.optim.Adam(ref_model.parameters(), lr=1e-3, weight_decay=1e-4)
        ref_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ref_optimizer, factor=0.99, patience=4)
        MLP.train(ref_model, trainloader, testloader, ref_criterion, ref_optimizer, ref_scheduler, args.epoch)

        # After training the reference model, get the weights of the first layer of the reference model
        initialization={}
        initialization['weights'] = ref_model.fc1.weight.data
        initialization['bias'] = ref_model.fc1.bias.data
        
        attack_model = MLP.SelfAttention(FEAT_D, initialization, dropout=DROPOUT_, num_heads=NUM_HEAD_).cuda()
        criterion = torch.nn.CrossEntropyLoss() 
        optimizer = torch.optim.Adam(attack_model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.99, patience=4)
        MLP.train(attack_model, trainloader, testloader, criterion, optimizer, scheduler, n_epoch=50)
    if args.link_detector=="mlp":
        attack_model = MLP.MLP40(FEAT_D, dropout=DROPOUT_).cuda()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(attack_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.99, patience=4)
        MLP.train(attack_model, trainloader, testloader, criterion, optimizer, scheduler, args.epoch)

    from sklearn.metrics import roc_auc_score, roc_curve
    y_true = []
    y_pred = []
    for data_batch, labels_batch in testloader:
        data_batch, labels_batch = data_batch.cuda(), labels_batch.long().cuda()
        y_pred.append(attack_model(data_batch))
        y_true.append(labels_batch)
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    print(y_true.shape, y_pred.shape)
    roc_score = roc_auc_score(y_true, y_pred[:,1], multi_class='ovr')
    print('AUC score: ', roc_score)
    # write ROC scores to file
    file_path = "results/{}-{}-{}.txt".format(args.dataset, args.arch, args.experiment)
    with open(file_path, 'a+') as f:
        f.write(str(roc_score))
        f.write(', ')
    
