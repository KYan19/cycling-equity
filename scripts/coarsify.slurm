#!/bin/bash
#SBATCH --job-name=coarsify
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=03:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=kyan@princeton.edu

module purge
module load anaconda3/2021.11
conda activate ox

python V2_coarse_graph_parallel.py >> output.txt