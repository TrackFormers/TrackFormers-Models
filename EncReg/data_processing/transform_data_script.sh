#!/bin/bash
#SBATCH --job-name=data_creation          # Job name
#SBATCH --ntasks=1                    # Run on a single GPU
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END

cd /projects/0/nisei0750/nadia/

module purge
module load 2022 Python/3.10.4-GCCcore-11.3.0

pip install torch==1.13.1
pip install scikit-learn
pip install pandas

for i in {21000..30000} ; do
    python trackml_data.py -e ${i} -l 50 -u 100 -s 1
done