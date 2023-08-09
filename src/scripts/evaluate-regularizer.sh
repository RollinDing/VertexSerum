# Scripts for evaluating the number of layers impact on model.

# /bin/bash

dataset="AmazonCoBuyComputerDataset"
arch="sage"
experiment="regularizer"
n_layer=3
percent=0.1
file="results/$dataset-$arch-$experiment.txt"
echo -e "Experiment Start: $(date +"%T").\n" >> $file

# Train model with alpha = 0.1 cross entropy term
alphas=(1e-2 1e-1 1)
lambs=(-1e1 -1 -0.1 -1e2)
betas=(1e-2 1e-1 1)

for alpha in ${alphas[@]}; do
    for beta in ${betas[@]}; do
        for lamb in ${lambs[@]}; do
            python3 attacks/gma_old.py --result-dir="../trained_models_$experiment/$alpha-$beta-$lamb"  --dataset=$dataset --arch=$arch --percent=$percent \
                    --pgd-steps=100 --n-layers=$n_layer \
                    --poison-rate=1 --epsilon=1e-3 --epoch=1000 --alpha=$alpha  --lamb=$lamb --beta=$beta --dropout=0.5 
            exp_id=1
            echo -e "regularizer term $alpha-$beta-$lamb" >> $file
            for run in {1..10}; do    
                python attacks/gma.py --result-dir="../trained_models_$experiment/$alpha-$beta-$lamb" --dataset=$dataset --arch=$arch \
                        --percent=$percent --n-layers=$n_layer --epoch=50 --target=1 --exp-id=$exp_id --mode=VertexSerum -ld attn --experiment=$experiment
            done
            echo -e "\n" >> $file
        done
    done
done