## VertexSerum - Poisoning Graph Neural Networks For Stealing Links

### Introduction
Reproduce experiments results for paper *VertexSerum: Poisoning Graph Neural Networks for Link Inference*.

-------------------------------
### Installation
1. **System requirements**: 
    We run the experiment on `Ubuntu 18.04`, 
    with an `AMD Ryzen 9 3900X 12-Core Processor`, 
    with `NVIDIA TITAN RTX` GPU.

2. **Set up enviroments**To setup the enviroment for experiment, run:
    ```
    conda create --name <env> --file requirements.txt
    ```
--------------------------------
### Code Structure
- project/
    * README.md
    * requirements.txt
    - src/
        - scripts/
            * evaluate.sh
        - attacks/
            * evaluate.py
            * gma_bb.py
            * gma_old.py
            * gma_online.py
            * gma.py
            * pgd.py
            * posterior_detector.py
        - models/
            * gat.py
            * gcn.py
            * MLP.py
            * sage.py
        - utils/
            * analyze.py
            * homophily.py
            * utils.py
            * visualize.py
        * args.py

### Scripts
1. To easily run the experiment, run scripts
```
bash scripts/evaluate.sh
```
The script will train a benign graph model with GraphSAGE on CoraDataset, then do VertexSerum poisoning attack on the benign graph. It will then evaluate the ROC-AUC scores on self-attention based link detection 10 times.

Change args 'mode' to switch between VertexSerum or Steal_Link;

Change args 'ld' to switch between Self attention detector or MLP detector;

Change args 'dataset' and 'arch' to switch among different graph datasets or gnn models.

For more information, check `args.py`

**Evaluate ablation Study**
2. To evaluate the online setting, run 
```
bash scripts/evaluate-online.sh
```
3. To evaluate blackbox, run
```
bash scripts/evaluate-blackbox.sh
```
4. Note please make sure about the experiment ID in each scripts.

### Usage 
1. To evaluate the VertextSerum attack, run:
```
python3 attacks/gma_old.py 
    --result-dir=<train directory>  
    --dataset=<dataset> 
    --arch=<model-architecture> 
    --percent=<size of partial graph>
    --pgd-steps=<pgd steps> 
    --n-layers=<number of layers> \
    --poison-rate=<poison rate> 
    --epsilon=<pgd step size> 
    --epoch=<number of epoch> 
    --alpha=<alpha>  
    --lamb=<lambda> 
    --beta=<beta> 
    --dropout=<dropout rate> 
```
Run this python script will create a folder `<train directory>`, with a benign model, a poisoned model and a shadow model, including corresponding poisoned graphs. It will create a directory named `n-k-...`, here `n` is the experiment id.

2. To run the link detector with experiment id,  
```
python3 attacks/gma.py 
    --result-dir=<train directory>  
    --dataset=<dataset> 
    --arch=<model-architecture> 
    --percent=<size of partial graph>
    --pgd-steps=<pgd steps> 
    --n-layers=<number of layers> \
    --poison-rate=<poison rate> 
    --epsilon=<pgd step size> 
    --epoch=<number of epoch> 
    --alpha=<alpha>  
    --lamb=<lambda> 
    --beta=<beta> 
    --exp_id=<experiment id>
    --mode=<VertexSerum or SLA> 
    --ld=<mlp or attn>
```
This script will  run the link detector and output the attack AUC scores for selected poisoned mode and selected link detector.

