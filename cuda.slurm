#!/bin/bash -l
#SBATCH --cluster=genius
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=llp_h_pds
#SBATCH --time=00:05:00
#SBATCH --error="%x.e%A"
#SBATCH --output="%x.o%A"

source /data/leuven/303/vsc30380/slurmhooks

module purge
module load GCC/10.3.0
module load CUDA/11.7.0

nvcc -O3 -DNDEBUG -o kmeans mainCuda.cu rng.cpp -gencode arch=compute_60,code=sm_60

./kmeans --threads 1 --input 100000x5.csv --output output.csv --seed 1953139 --k 20 --repetitions 40 --blocks 256
