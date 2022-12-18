#!/bin/bash
#SBATCH --output dsq-execution_file-%A_%1a-%N.out
#SBATCH --array 0%2000
#SBATCH --job-name dsq-execution_file
#SBATCH --mem=64G --gres=gpu:1 --partition=gpu -t 15:00:00 -C cascadelake

# DO NOT EDIT LINE BELOW
/gpfs/loomis/apps/avx/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/gibbs/project/karunakaran/as3465/replicate_car_paper/blp_before_disentanglement/execution_file.txt --status-dir /gpfs/gibbs/project/karunakaran/as3465/replicate_car_paper/blp_before_disentanglement

