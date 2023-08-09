# /bin/bash
dataset="CoraGraphDataset"
arch="sage"
n_layer=3
percent=0.1
alpha=1
beta=1e-2
lamb=-1e1
exp_id=1
target=1
ld="attn"
mode="VertexSerum"

file="results/$dataset-$arch.txt"
echo -e "Experiment Start: $(date +"%T").\n" >> $file
echo -e "Training model in an online learning setting\n" >> $file

# VertexSerum
python3 attacks/gma_old.py --result-dir="../trained_models"  --dataset=$dataset --arch=$arch --percent=$percent \
        --pgd-steps=100 --n-layers=$n_layer \
        --poison-rate=1 --epsilon=1e-3 --epoch=1000 --alpha=$alpha  --lamb=$lamb --beta=$beta --dropout=0.5 --target=$target

# Attention based link detector
for run in {1..10}; do    
    python attacks/gma.py --result-dir="../trained_models" --dataset=$dataset --arch=$arch \
            --percent=$percent --n-layers=$n_layer --epoch=50 --target=$target --exp-id=$exp_id --mode=$mode -ld $ld
done