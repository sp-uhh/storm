#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2  # When using DDP, one task/process will be launched for each GPU 
#SBATCH --cpus-per-task=8          # We have 64 total in spgpu2 and 32 in spgpu1, making it 8 cores per GPU process in both cases
#SBATCH --partition=all
#SBATCH --nodelist=spgpu2          # Or set it to spgpu1
#SBATCH --job-name=storm-p+j-vctk-reverb
#SBATCH --output=.slurm/%x-%j.out    # Save to folder ./jobs, %x means the job name. You may need to create this folder
#SBATCH --error=.slurm/%x-%j.err
#SBATCH --time=4-00:00             # Limit job to 4 days
#SBATCH --mem=0                    # SLURM does not limit the memory usage, but it will block jobs from launching
#SBATCH --gres=gpu:2        # Number of GPUs to allocate

# source ../../.venv/bin/activate

pc=spgpu2

if [ "$pc" = sppc1 ]; then
    data_dir=/data/lemercier/databases
    home_dir=/export/home/lemercier
elif [ "$pc" = spgpu1 ]; then
    data_dir=/data/lemercier/databases
    home_dir=/data1/lemercier
elif [ "$pc" = spgpu2 ]; then
    # data_dir=/data3/lemercier/databases
    data_dir=/data3/databases
    home_dir=/export/home/lemercier
fi;

base_dir=$data_dir/VCTK-Reverb
format=reverb_vctk

srun python3 train.py \
    --mode denoiser-only \
    --base_dir $base_dir \
    --format $format \
    --num_frames 512 \
    --batch_size 4 \
    --devices 4

# srun python3 train.py \
#     --mode regen-joint-training \
#     --pretrained_denoiser /export/home/lemercier/code/_public_repos/storm/lightning_logs/denoiser_ncsnpp_vctk-reverb/checkpoints/epoch=89-step=234270.ckpt \
#     --base_dir $base_dir \
#     --format $format \
#     --num_frames 512 \
#     --batch_size 4 \
#     --devices 2 \
#     --nolog
