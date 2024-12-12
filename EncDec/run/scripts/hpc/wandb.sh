#!/bin/sh
#PBS -N zwolffs
#PBS -q gpu-nv
#PBS -l walltime=24:00:00,mem=32gb

conda activate tracking-gpu

slug="10to50_noisy_curved_3d"

nvidia-smi

export WANDB_API_KEY=$api_key

python train_wandb.py --number_of_tracks=150000 \
                --slug=${slug} \
                --max_size_tracks=30 \
                --max_size_hits=450 \
                --infile_name="./data/experiment_3d_noisy-100k-events-10-to-50-curved-tracks/events_all/hits_and_tracks_3d_events_all.csv" \
                --parsed_data_dir="./data/experiment_3d_noisy-100k-events-10-to-50-curved-tracks/parsed_data" \
                --hyperparameter_config_path="./configs/hyperparameters/3d_noisy_curved_10to50.json"


echo "... Run finished $(date) ..."
