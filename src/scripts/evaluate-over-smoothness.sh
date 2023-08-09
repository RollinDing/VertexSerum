
# Scripts for evaluating the number of layers impact on model.

# /bin/bash

dataset="CoraGraphDataset"
arch="gcn"
experiment="oversmooth"
file="results/$dataset-$arch-$experiment.txt"
echo -e "Experiment Start: $(date +"%T").\n" >> $file
echo -e "Training models with different number of layers\n" >> $file

# # Train GNN models with different number of layers report the model accuracy
for n_layer in {1..10}; do
    python3 attacks/gma_old.py --result-dir="../trained_models_oversmooth"  --dataset=$dataset --arch=$arch --percent=0.1 --pgd-steps=100 --n-layers=$n_layer \
                --poison-rate=1 --epsilon=1e-3 --epoch=1000 --lamb=-1e2 --beta=1e-1 --dropout=0.5
done

# Evaluate the model using VertexSerum 
for run in {1..10}; do
    for exp_id in {1..10}; do
        python attacks/gma.py --result-dir="../trained_models_oversmooth" --dataset=$dataset --arch=$arch \
                --percent=0.1 --n-layers=$exp_id --epoch=50 --target=1 --exp-id=$exp_id --mode=VertexSerum -ld attn --experiment=$experiment
    done
    echo -e "\n" >> $file
done
