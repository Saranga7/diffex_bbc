#!/bin/bash
#SBATCH --account=account@v100
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --qos=qos_gpu-dev
#SBATCH --hint=nomultithread
#SBATCH --mail-user=name@domain
#SBATCH --mail-type=FAIL
#SBATCH --job-name=run_name
#SBATCH --error=path/to/file
#SBATCH --output=path/to/file

# --------------------------------------> Load compute envs <--------------------------------------
module purge
eval "$(micromamba shell hook --shell bash)"
micromamba deactivate
micromamba activate diffex

# --------------------------------------> Python Path <--------------------------------------
export PYTHONPATH="${PYTHONPATH}:path/to/project/root"

# ----------------------------------------> Run <----------------------------------------
WANDB_MODE=offline srun python path/to/frozen/copy/of/run_bbbc.py
