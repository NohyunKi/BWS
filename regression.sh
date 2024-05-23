#!/bin/bash
if [ -n "$1" ]; then GPU="$1"; else GPU=0; fi
if [ -n "$2" ]
then
    DATASET="$2"
    if [[ "$2" == CIFAR10 ]]; then LR=0.05; EPOCHS=200; MODEL=resnet18; BATCH_SIZE=128;
    elif [[ "$2" == CIFAR100 ]]; then LR=0.1; EPOCHS=400; MODEL=resnet50; BATCH_SIZE=128;
    elif [[ "$2" == ImageNet ]]; then LR=0.1; EPOCHS=90; MODEL=resnet50; BATCH_SIZE=256;
    else LR=0.02; EPOCHS=100; MODEL=resnet18; BATCH_SIZE=128;
    fi

    if [[ "$2" == ImageNet ]];
    then STR="--no_iter --scheduler=Step --regularizer=0.0001";
    else STR=""
    fi

else DATASET=CIFAR10; LR=0.05; EPOCHS=200; MODEL=resnet18;
fi
# if [ -n "$3" ]; then STR2="$3"; else STR2=""; fi
if [ -n "$3" ]
then 
    MODEL="$3";
    if [[ "$3" == vit_timm ]]; then LR=1e-4; STR2="--regularizer=1e-4 --scheduler=Step"; EPOCHS=10;
    elif [[ "$3" == cnn ]]; then LR=0.05; STR2=--regularizer=1e-4;
    elif [[ "$3" == eff ]]; then LR=0.1; STR2=--regularizer=1e-4;
    else STR2=""
    fi
fi

MEASURE=forgetting
SEEDS="42"

EXP="regression.py"

for SEED in ${SEEDS}
do
    echo python experiment/${EXP} \
        --exp_name=${PYTHON_FILE} --seed=${SEED} --dataset=${DATASET} \
        --epochs=${EPOCHS} --model=${MODEL} --lr=${LR} --batch_size=${BATCH_SIZE} \
        ${STR} ${STR2} --para_gpu=${GPU} --measure=${MEASURE}

    python experiment/${EXP} \
        --exp_name=${PYTHON_FILE} --seed=${SEED} --dataset=${DATASET} \
        --epochs=${EPOCHS} --model=${MODEL} --lr=${LR} --batch_size=${BATCH_SIZE} \
        ${STR} ${STR2} --para_gpu=${GPU} --measure=${MEASURE}
done
