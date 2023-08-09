import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Tensorflow Training")

    # Primary
    parser.add_argument(
        "--result-dir",
        default="../trained_models",
        type=str,
        help="directory to save results",
    )

    parser.add_argument(
        "--model-dir",
        default="../trained_models",
        type=str,
        help="directory to load model",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=("CoraGraphDataset", "CiteseerGraphDataset", "PubmedGraphDataset", "CoraFullDataset", "CoauthorCSDataset","FlickrDataset", "AmazonCoBuyComputerDataset", "AmazonCoBuyPhotoDataset"),
        help="Dataset for training and evaluating the graph neural networks",
    )

    parser.add_argument(
        "--arch",
        type=str, 
        choices=("sage", "gcn", "rgcn", "gat"),
        help="Graph deep neural network structure to use"
    )

    parser.add_argument(
        "--n-hidden",
        type=int,
        default=16,
        help="Size of hidden layer of graph neural network"
    )

    parser.add_argument(
        "--n-layers",
        type=int,
        default=2,
        help="Number of layers of graph neural network"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0,
        help="Dropout out rate for training graph neural network"
    )

    parser.add_argument(
        "--percent",
        type=float,
        help="Percentage of nodes to use for training."
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=100,
        help='Number of epoch to train the model.'
    )

    parser.add_argument(
        "--pgd-steps",
        type=int,
        default=100,
        help='Number of PGD steps'
    )

    parser.add_argument(
        "--poison-rate",
        type=float,
        default=0.1,
        help='Percentage of Poisoned Samples'
    )

    parser.add_argument(
        "--target",
        type=int,
        default=1,
        help="The target class which the victim nodes belong to."
    )

    parser.add_argument(
        "--lamb",
        type=float,
        default=-0.1,
        help='Regularization term for the first loss term'
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help='Regularization term for cross-entropy term'
    )
    
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help='Regularization term for the second loss term'
    )

    parser.add_argument(
        "--epsilon",
        type=float, 
        default=1e-3,
        help='PGD step size'
    )

    parser.add_argument(
        "--exp-id",
        type=int,
        default=0,
        help='Experiment ID'
    )

    parser.add_argument(
        "--mode",
        type=str,
        default='VertexSerum',
        choices=['VertexSerum', 'Steal_Link'],
        help='experiment mode'
    )

    parser.add_argument(
        "-ld", "--link-detector",
        type=str,
        default="attn",
        help='link detector',
        choices=['attn', 'mlp']
    )

    parser.add_argument(
        "--experiment", 
        type=str,
        default="roc-score",
        help='experiment name'
    )

    parser.add_argument(
        "--run",
        type=int,
        default=0,
        help='Poison Round for online learning'
    )

    parser.add_argument(
        "--shadow",
        type=str,
        default='gcn',
        help='model structure for shadow training'
    )

    return parser.parse_args()