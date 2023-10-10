#!/bin/bash

#SBATCH -p gpu --gres=gpu:8
#SBATCH -n 4
#SBATCH -t 96:00:00
#SBATCH --mem=384g
#SBATCH --ntasks-per-node=8

#module load cuda

date
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ulimit -u 127590
python3 -m spanet.tune /users/mstamenk/spanet-work/hhh/options_files/cms/classification_test_boosted.json -g 1
#python3 -m spanet.train -of /users/mstamenk/spanet-work/hhh/options_files/cms/classification_v26_boosted.json --gpus 8
date
exit
