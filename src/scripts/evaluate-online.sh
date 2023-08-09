
# Scripts for evaluating the number of layers impact on model.

# /bin/bash

dataset="CoraGraphDataset"
arch="sage"
experiment="online"
n_layer=3
file="results/$dataset-$arch-$experiment.txt"
echo -e "Experiment Start: $(date +"%T").\n" >> $file
echo -e "Training model in an online learning setting\n" >> $file

# Train GNN models with different number of layers report the model accuracy
for run in {0..7}; do
    python3 attacks/gma_online.py --result-dir="../trained_models_$experiment"  --dataset=$dataset --arch=$arch --percent=0.1 --pgd-steps=100 --n-layers=$n_layer \
                --poison-rate=1 --epsilon=1e-3 --epoch=1000 --lamb=-1e2 --beta=1e-1 --dropout=0.5 --run=$run
done

# Evaluate the model using VertexSerum 

exp_id=1
echo -e "Poison Run $exp_id\n" >> $file
for run in {1..10}; do    
    python attacks/gma.py --result-dir="../trained_models_$experiment" --dataset=$dataset --arch=$arch \
            --percent=0.1 --n-layers=$n_layer --epoch=50 --target=1 --exp-id=$exp_id --mode=VertexSerum -ld attn --experiment=$experiment
done
echo -e "\n" >> $file
