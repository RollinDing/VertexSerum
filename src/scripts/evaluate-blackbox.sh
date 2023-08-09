# Scripts for evaluating blackbox setting of poisoning attack
# /bin/bash

dataset="CoraGraphDataset"
# base_model is sage

experiment="shadow"
n_layer=3
arch="sage"
file="results/$dataset-$arch-$experiment.txt"
echo -e "Experiment Start: $(date +"%T").\n" >> $file


# Train GNN models with different shadow model
echo -e "Training model target sage\n" >> $file

python3 attacks/gma_bb.py --result-dir="../trained_models_$experiment"  --dataset=$dataset --arch=$arch --percent=0.1 --pgd-steps=100 --n-layers=$n_layer \
            --poison-rate=1 --epsilon=1e-3 --epoch=1000 --lamb=-1e2 --beta=1e-1 --dropout=0.5 --shadow=gat
python3 attacks/gma_bb.py --result-dir="../trained_models_$experiment"  --dataset=$dataset --arch=$arch --percent=0.1 --pgd-steps=100 --n-layers=$n_layer \
            --poison-rate=1 --epsilon=1e-3 --epoch=1000 --lamb=-1e2 --beta=1e-1 --dropout=0.5 --shadow=gcn
python3 attacks/gma_bb.py --result-dir="../trained_models_$experiment"  --dataset=$dataset --arch=$arch --percent=0.1 --pgd-steps=100 --n-layers=$n_layer \
            --poison-rate=1 --epsilon=1e-3 --epoch=1000 --lamb=-1e2 --beta=1e-1 --dropout=0.5 --shadow=sage

# Evaluate the model using VertexSerum 
exp_id=1
echo -e "shadow model is gat\n" >> $file
for run in {1..10}; do    
    python attacks/gma.py --result-dir="../trained_models_$experiment" --dataset=$dataset --arch=$arch \
            --percent=0.1 --n-layers=$n_layer --epoch=50 --target=1 --exp-id=$exp_id --mode=VertexSerum -ld attn --experiment=$experiment
done
echo -e "\n" >> $file

exp_id=2
echo -e "shadow model is gcn\n" >> $file
for run in {1..10}; do    
    python attacks/gma.py --result-dir="../trained_models_$experiment" --dataset=$dataset --arch=$arch \
            --percent=0.1 --n-layers=$n_layer --epoch=50 --target=1 --exp-id=$exp_id --mode=VertexSerum -ld attn --experiment=$experiment
done
echo -e "\n" >> $file

exp_id=3
echo -e "shadow model is sage\n" >> $file
for run in {1..10}; do    
    python attacks/gma.py --result-dir="../trained_models_$experiment" --dataset=$dataset --arch=$arch \
            --percent=0.1 --n-layers=$n_layer --epoch=50 --target=1 --exp-id=$exp_id --mode=VertexSerum -ld attn --experiment=$experiment
done
echo -e "\n" >> $file

# Train GNN models with different shadow model
echo -e "Training model target sage\n" >> $file
arch="gcn"
n_layer=2
file="results/$dataset-$arch-$experiment.txt"
echo -e "Experiment Start: $(date +"%T").\n" >> $file


python3 attacks/gma_bb.py --result-dir="../trained_models_$experiment"  --dataset=$dataset --arch=$arch --percent=0.1 --pgd-steps=100 --n-layers=$n_layer \
            --poison-rate=1 --epsilon=1e-3 --epoch=1000 --lamb=-1e2 --beta=1e-1 --dropout=0.5 --shadow=gat
python3 attacks/gma_bb.py --result-dir="../trained_models_$experiment"  --dataset=$dataset --arch=$arch --percent=0.1 --pgd-steps=100 --n-layers=$n_layer \
            --poison-rate=1 --epsilon=1e-3 --epoch=1000 --lamb=-1e2 --beta=1e-1 --dropout=0.5 --shadow=gcn
python3 attacks/gma_bb.py --result-dir="../trained_models_$experiment"  --dataset=$dataset --arch=$arch --percent=0.1 --pgd-steps=100 --n-layers=$n_layer \
            --poison-rate=1 --epsilon=1e-3 --epoch=1000 --lamb=-1e2 --beta=1e-1 --dropout=0.5 --shadow=sage

# Evaluate the model using VertexSerum 
exp_id=1
echo -e "shadow model is gat\n" >> $file
for run in {1..10}; do    
    python attacks/gma.py --result-dir="../trained_models_$experiment" --dataset=$dataset --arch=$arch \
            --percent=0.1 --n-layers=$n_layer --epoch=50 --target=1 --exp-id=$exp_id --mode=VertexSerum -ld attn --experiment=$experiment
done
echo -e "\n" >> $file

exp_id=2
echo -e "shadow model is gcn\n" >> $file
for run in {1..10}; do    
    python attacks/gma.py --result-dir="../trained_models_$experiment" --dataset=$dataset --arch=$arch \
            --percent=0.1 --n-layers=$n_layer --epoch=50 --target=1 --exp-id=$exp_id --mode=VertexSerum -ld attn --experiment=$experiment
done
echo -e "\n" >> $file

exp_id=3
echo -e "shadow model is sage\n" >> $file
for run in {1..10}; do    
    python attacks/gma.py --result-dir="../trained_models_$experiment" --dataset=$dataset --arch=$arch \
            --percent=0.1 --n-layers=$n_layer --epoch=50 --target=1 --exp-id=$exp_id --mode=VertexSerum -ld attn --experiment=$experiment
done
echo -e "\n" >> $file

# Train GNN models with different shadow model
echo -e "Training model target gat\n" >> $file
arch="gat"
n_layer=2
file="results/$dataset-$arch-$experiment.txt"
echo -e "Experiment Start: $(date +"%T").\n" >> $file
python3 attacks/gma_bb.py --result-dir="../trained_models_$experiment"  --dataset=$dataset --arch=$arch --percent=0.1 --pgd-steps=100 --n-layers=$n_layer \
            --poison-rate=1 --epsilon=1e-3 --epoch=1000 --lamb=-1e2 --beta=1e-1 --dropout=0.5 --shadow=gat
python3 attacks/gma_bb.py --result-dir="../trained_models_$experiment"  --dataset=$dataset --arch=$arch --percent=0.1 --pgd-steps=100 --n-layers=$n_layer \
            --poison-rate=1 --epsilon=1e-3 --epoch=1000 --lamb=-1e2 --beta=1e-1 --dropout=0.5 --shadow=gcn
python3 attacks/gma_bb.py --result-dir="../trained_models_$experiment"  --dataset=$dataset --arch=$arch --percent=0.1 --pgd-steps=100 --n-layers=$n_layer \
            --poison-rate=1 --epsilon=1e-3 --epoch=1000 --lamb=-1e2 --beta=1e-1 --dropout=0.5 --shadow=sage

# Evaluate the model using VertexSerum 
exp_id=1
echo -e "shadow model is gat\n" >> $file
for run in {1..10}; do    
    python attacks/gma.py --result-dir="../trained_models_$experiment" --dataset=$dataset --arch=$arch \
            --percent=0.1 --n-layers=$n_layer --epoch=50 --target=1 --exp-id=$exp_id --mode=VertexSerum -ld attn --experiment=$experiment
done
echo -e "\n" >> $file

exp_id=2
echo -e "shadow model is gcn\n" >> $file
for run in {1..10}; do    
    python attacks/gma.py --result-dir="../trained_models_$experiment" --dataset=$dataset --arch=$arch \
            --percent=0.1 --n-layers=$n_layer --epoch=50 --target=1 --exp-id=$exp_id --mode=VertexSerum -ld attn --experiment=$experiment
done
echo -e "\n" >> $file

exp_id=3
echo -e "shadow model is sage\n" >> $file
for run in {1..10}; do    
    python attacks/gma.py --result-dir="../trained_models_$experiment" --dataset=$dataset --arch=$arch \
            --percent=0.1 --n-layers=$n_layer --epoch=50 --target=1 --exp-id=$exp_id --mode=VertexSerum -ld attn --experiment=$experiment
done
echo -e "\n" >> $file