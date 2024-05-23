#!/bin/bash

if [ -n "$1" ]; then GPU="$1"; else GPU=0; fi
if [ -n "$2" ]
then
    DATASET="$2"
    if [[ "$2" == CIFAR10 ]]; then LR=0.05; EPOCHS=200; MODEL=resnet18; BATCH_SIZE=128;
    elif [[ "$2" == CIFAR100 ]]; then LR=0.1; EPOCHS=400; MODEL=resnet50; BATCH_SIZE=128;
    elif [[ "$2" == TinyImageNet ]]; then LR=0.1; EPOCHS=200; MODEL=resnet50; BATCH_SIZE=128;
    elif [[ "$2" == ImageNet ]]; then LR=0.1; EPOCHS=90; MODEL=resnet50; BATCH_SIZE=256;
    else LR=0.02; EPOCHS=100; MODEL=resnet18; BATCH_SIZE=128;
    fi

    if [[ "$2" == ImageNet ]];
    then STR="--no_iter --scheduler=Step --regularizer=0.0001";
    else STR=""
    fi

else DATASET=CIFAR10; LR=0.05; EPOCHS=200; MODEL=resnet18;
fi

SEEDS="42"
EXP=make_all_measure.py

for SEED in ${SEEDS}
do
    echo CUDA_VISIBLE_DEVICES=${GPU} python experiment/${EXP} \
        --exp_name=${PYTHON_FILE} --seed=${SEED} --dataset=${DATASET} \
        --epochs=${EPOCHS} --model=${MODEL} --lr=${LR} --batch_size=${BATCH_SIZE} \
        --noise --noise_rate=0.4 ${STR}

    CUDA_VISIBLE_DEVICES=${GPU} python experiment/${EXP} \
        --exp_name=${PYTHON_FILE} --seed=${SEED} --dataset=${DATASET} \
        --epochs=${EPOCHS} --model=${MODEL} --lr=${LR} --batch_size=${BATCH_SIZE} \
        --noise --noise_rate=0.4 ${STR}
done