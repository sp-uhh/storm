#!/bin/bash -l
#

#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100

#SBATCH --job-name=storm_vctk-reverb

#SBATCH --output=.slurm/%x-%j.out

#SBATCH --error=.slurm/%x-%j.err

#SBATCH --time=23:59:00

#SBATCH --export=NONE   

#SBATCH --gres=gpu:a100:4

unset SLURM_EXPORT_ENV 

module load python cuda/12.1 gcc/10 ninja
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

srun python $HOME/code/storm/train.py \
    --base_dir $HPCVAULT/databases/VCTK-Reverb.h5 \
    --mode regen-joint-training \
    --format reverb_vctk \
    --num_frames 512 \
    --batch_size 4 \
    --backbone_denoiser ncsnpp \
    --pretrained_denoiser $HOME/code/storm/.logs/ncsnppm_denoiser_epoch=89.ckpt \
    --backbone_score ncsnpp \
    --condition both \
    --devices 4