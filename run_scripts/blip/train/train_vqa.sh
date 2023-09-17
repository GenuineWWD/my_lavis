#!/bin/bash

source  ~/env.sh

ndcus=$1
jobname=$2

output=./output
GPUS_PER_NODE=${ndcus:-4}
if [ $GPUS_PER_NODE -ge 4 ]; then
  GPUS_PER_NODE=4
fi

echo The output dict is ${output}

mkdir -p ${output}/${jobname}

# partition=xahdtest
partition=xahdnormal

time=$(date "+%Y%m%d-%H%M%S")


set -x 
srun -p ${partition} -n$ndcus --gres=dcu:${GPUS_PER_NODE} -o $output/$jobname/${jobname}_${time}.log  --job-name=${jobname} \
--cpus-per-task=8  --kill-on-bad-exit=1 --ntasks-per-node=${GPUS_PER_NODE} \
python train.py --cfg-path lavis/projects/blip/train/vqav2_ft.yaml

