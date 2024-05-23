#!/bin/bash

if [ -n "$1" ]
then
    DATASET="$1"
    if [[ "$1" == CIFAR10 ]]; then LR=0.05; EPOCHS=200; MODEL=resnet18; BATCH_SIZE=128;
    elif [[ "$1" == CIFAR100 ]]; then LR=0.1; EPOCHS=400; MODEL=resnet50; BATCH_SIZE=128;
    elif [[ "$1" == ImageNet ]]; then LR=0.1; EPOCHS=90; MODEL=resnet50; BATCH_SIZE=256;
    else LR=0.02; EPOCHS=100; MODEL=resnet18; BATCH_SIZE=128;
    fi

    if [[ "$1" == ImageNet ]];
    then STR="--no_iter --regularizer=0.0001 --scheduler=Step";
    else STR="";
    fi
else DATASET=CIFAR10; LR=0.05; EPOCHS=200; MODEL=resnet18;
fi

if [ -n "$2" ]
then 
    MODEL="$2";
    if [[ "$2" == vit_timm ]]; then LR=1e-4; STR2="--regularizer=1e-4 --scheduler=Step"; EPOCHS=10;
    elif [[ "$2" == cnn ]]; then LR=0.05; STR2=--regularizer=1e-4;
    elif [[ "$2" == eff ]]; then LR=0.1; STR2=--regularizer=1e-4;
    else STR2=""
    fi
fi

MEASURES="forgetting"
SEEDS="42"

PYTHON_FILE=pruning_window_sliding.py

for SEED in ${SEEDS}
do
    for MEASURE in ${MEASURES}
    do
        echo python experiment/${PYTHON_FILE} \
            --exp_name=${PYTHON_FILE} --seed=${SEED} --dataset=${DATASET} \
            --measure=${MEASURE} --epochs=${EPOCHS} --batch_size=${BATCH_SIZE} \
            --model=${MODEL} --lr=${LR} ${STR} ${STR2} --no_iter

        python experiment/${PYTHON_FILE} \
            --exp_name=${PYTHON_FILE} --seed=${SEED} --dataset=${DATASET} \
            --measure=${MEASURE} --epochs=${EPOCHS} --batch_size=${BATCH_SIZE} \
            --model=${MODEL} --lr=${LR} ${STR} ${STR2} --no_iter
    done
done