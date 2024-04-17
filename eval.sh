#!/bin/bash

pc=spgpu2

if [ "$pc" = sppc1 ]; then
    data_dir=/data/lemercier/databases
    home_dir=/export/home/lemercier
elif [ "$pc" = spgpu1 ]; then
    data_dir=/data/lemercier/databases
    home_dir=/data1/lemercier
elif [ "$pc" = spgpu2 ]; then
    data_dir=/data3/databases
    home_dir=/export/home/lemercier
fi;

python enhancement.py  \
    --test_dir $data_dir/VCTK-Reverb-eloi-subset/test/reverberant \
    --enhanced_dir exp/vctk-eloi-subset/vctk-reverb_denoiser_ncsnppm_113 \
    --ckpt /export/home/lemercier/code/_public_repos/storm/logs/denoiser_ncsnpp_vctk-reverb_epoch=113.ckpt \
    --mode denoiser-only \
    # --ckpt /export/home/lemercier/code/_public_repos/storm/.logs/denoiser_ncsnpp_vctk-reverb_epoch=89.ckpt \
    # --n_files 100

python enhancement.py  \
    --test_dir $data_dir/VCTK-Reverb-eloi-subset/test/reverberant \
    --enhanced_dir exp/vctk-eloi-subset/vctk-reverb_sgmse+_ncsnppm \
    --ckpt /export/home/lemercier/code/_public_repos/storm/logs/sgmse+_ncsnpp_vctk-reverb_epoch=140.ckpt \
    --mode score-only \
    --N 30
    # --n_files 100

python enhancement.py  \
    --test_dir $data_dir/VCTK-Reverb-eloi-subset/test/reverberant \
    --enhanced_dir exp/vctk-eloi-subset/vctk-reverb_storm_ncsnppm \
    --ckpt /export/home/lemercier/code/_public_repos/storm/logs/storm_ncsnpp_vctk-reverb_epoch=200.ckpt \
    --mode storm \
    --N 30
    # --n_files 100