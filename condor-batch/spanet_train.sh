#!/bin/bash

#SBATCH -p gpu --gres=gpu:8 
#SBATCH -n 1
#SBATCH -t 96:00:00
#SBATCH --mem=384g


date
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#python3 -m spanet.train -of /users/mstamenk/spanet-work/hhh/options_files/cms/classification_test_boosted.json --gpus 8 --checkpoint /users/mstamenk/spanet-work/hhh/condor/classification-pnet/spanet_output/version_4/checkpoints/last.ckpt
python3 -m spanet.train -of /users/mstamenk/spanet-work/hhh/options_files/cms/classification_v29_boosted.json --gpus 8
date
exit
