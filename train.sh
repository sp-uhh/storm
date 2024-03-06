#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4  # When using DDP, one task/process will be launched for each GPU 
#SBATCH --cpus-per-task=8          # We have 64 total in spgpu2 and 32 in spgpu1, making it 8 cores per GPU process in both cases
#SBATCH --partition=all
#SBATCH --nodelist=spgpu1          # Or set it to spgpu1
#SBATCH --job-name=ncsnpp-vctk-reverb_resume89
#SBATCH --output=.slurm/%x-%j.out    # Save to folder ./jobs, %x means the job name. You may need to create this folder
#SBATCH --error=.slurm/%x-%j.err
#SBATCH --time=4-00:00             # Limit job to 4 days
#SBATCH --mem=0                    # SLURM does not limit the memory usage, but it will block jobs from launching
#SBATCH --gres=gpu:4        # Number of GPUs to allocate

# source .environment/bin/activate

pc=spgpu1

if [ "$pc" = sppc1 ]; then
    data_dir=/data/lemercier/databases
    home_dir=/export/home/lemercier
elif [ "$pc" = spgpu1 ]; then
    data_dir=/data/lemercier/databases
    home_dir=/data1/lemercier
elif [ "$pc" = spgpu2 ]; then
    data_dir=/data3/lemercier/databases
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
    --devices 4 \
    --resume_from_checkpoint /export/home/lemercier/code/_public_repos/storm/lightning_logs/denoiser_ncsnpp_vctk-reverb/checkpoints/epoch=89-step=234270.ckpt
#     --nolog

python3 train.py \
    --mode denoiser-only \
    --base_dir /data/lemercier/databases/reverb_wsj0+chime/audio \
    --format reverb_wsj0 \
    --num_frames 128 \
    --batch_size 1 \
    --devices 1 \
    --resume_from_checkpoint tmp_ckpt.ckpt

# srun python3 train.py \
#     --mode score-only \
#     --base_dir $base_dir \
#     --format $format \
#     --num_frames 512 \
#     --batch_size 4 \
#     --devices 4
