#!/bin/bash 
#SBATCH --job-name=exp_MISR_JOINT_SRDiff_LCC_HR5_SR4_train_inference_hr5_sr4
#SBATCH --partition=tnt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=5:00:00
#SBATCH --mail-user=kanyamahanga@ipi.uni-hannover.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output logs/exp_MISR_JOINT_SRDiff_LCC_HR5_SR4_train_inference_%j.out
#SBATCH --error logs/exp_MISR_JOINT_SRDiff_LCC_HR5_SR4_train_inference_%j.err
source load_modules.sh
export CONDA_ENVS_PATH=$HOME/.conda/envs
export DATA_DIR=$BIGWORK
conda activate flair_venv
which python
cd $HOME/MISR_JOINT_SRDiff_LCC_HR5_SR4
srun python trainer.py --config configs/misr/highresnet_ltae.yaml --config_file flair-config-server.yml --exp_name misr/highresnet_ltae_ckpt --hparams="cond_net_ckpt=/bigwork/nhgnkany/Results/MISR_JOINT_SRDiff_LCC_HR5_SR4/results/checkpoints/misr/highresnet_ltae_ckpt" --infer
