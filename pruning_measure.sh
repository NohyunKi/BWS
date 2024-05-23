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
    then STR="--no_iter --scheduler=Step --regularizer=0.0001";
    else STR=""
    fi

else DATASET=CIFAR10; LR=0.05; EPOCHS=200; MODEL=resnet18;
fi
if [ -n "$2" ]; then MEASURES="$2"; else MEASURES="random"; fi
if [ -n "$3" ]; then GPU="$3"; else GPU=0; fi
if [ -n "$4" ]
then 
    MODEL="$4";
    if [[ "$4" == vit_timm ]]; then LR=1e-4; STR2="--regularizer=1e-4 --scheduler=Step"; EPOCHS=10;
    elif [[ "$4" == cnn ]]; then LR=0.05; STR2=--regularizer=1e-4;
    elif [[ "$4" == eff ]]; then LR=0.1; STR2=--regularizer=1e-4;
    else STR2=""
    fi
fi

SEEDS="42"
PYTHON_FILE=pruning_measure.py

for MEASURE in ${MEASURES}
do
    for SEED in ${SEEDS}
    do
        echo python experiment/${PYTHON_FILE} \
            --exp_name=${PYTHON_FILE} --seed=${SEED} --dataset=${DATASET} \
            --measure=${MEASURE} --epochs=${EPOCHS} --batch_size=${BATCH_SIZE} \
            --model=${MODEL} --lr=${LR} --para_gpu=${GPU} ${STR} ${STR2} --no_iter

        python experiment/${PYTHON_FILE} \
            --exp_name=${PYTHON_FILE} --seed=${SEED} --dataset=${DATASET} \
            --measure=${MEASURE} --epochs=${EPOCHS} --batch_size=${BATCH_SIZE} \
            --model=${MODEL} --lr=${LR} --para_gpu=${GPU} ${STR} ${STR2} --no_iter
    done
done