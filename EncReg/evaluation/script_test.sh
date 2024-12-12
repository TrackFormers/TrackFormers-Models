#!/bin/bash
#SBATCH --job-name=test               # Job name
#SBATCH --ntasks=1                    # Run on a single GPU
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END

cd /projects/0/nisei0750/nadia/repo/hep_transformer/

module purge
module load 2022 Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0

pip install torch==2.1.2
pip install scikit-learn
pip install pandas

python test.py --max_nr_hits 100 --data_path "hits_and_tracks_3d_3curved_events_all.csv" --model_name "3tracks_helical" --nr_enc_layers 2 --embedding_size 64 --hidden_dim 128 --data_type "curved"