# StoRM: A Stochastic Regeneration Model for Speech Enhancement and Dereverberation

<img src="https://raw.githubusercontent.com/sp-uhh/storm/master/inference.png" width="800" alt="StoRM inference process on a spectrogram. A predictive model is first used to get a estimate of the clean speech, with some possible distortions and resiudla noise. The diffusion generative model then uses this estimate as the initial point for a reverse process learns to generate clean speech in an iterative fashion starting from the corrupted signal xT.">

This repository contains the official PyTorch implementation for the paper:

- [*StoRM: A Stochastic Regeneration Model for Speech Enhancement And Dereverberation*](https://arxiv.org/abs/2212.11851), 2022 [1]

Audio examples and supplementary materials are available [on our project page](https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/storm.html).

## Installation

- Create a new virtual environment with Python 3.8 (we have not tested other Python versions, but they may work).
- Install the package dependencies via `pip install -r requirements.txt`.
- Your logs will be stored as local TensorBoard logs. Run `tensorboard --logdir logs/` to see them.

## Pretrained checkpoints

- We provide pretrained checkpoints for the models trained on TIMIT+Chime3 (enhancement), WSJ0+Chime3 (enhancement), Voicebank/DEMAND (enhancement) and WSJ0+Reverb (dereverberation), as in the paper. They can be downloaded [here](https://drive.google.com/drive/folders/1ExFm97obaXTYFoBApWjbK_ypxTP-Cgdq?usp=sharing).

Usage:
- For resuming training, you can use the `--resume_from_checkpoint` option of `train.py`.
- For evaluating these checkpoints, use the `--ckpt` option of `enhancement.py` (see section **Evaluation** below).

## Training

Training is done by executing `train.py`. A minimal running example with default settings (as in our paper [2]) can be run with

```bash
python train.py --format <your_format>--base_dir <your_base_dir> --gpus 0,
```

where 

- `your_base_dir` should be a path to a folder containing subdirectories `train/` and `valid/` (optionally `test/` as well). The subdirectory structure depends on `your_format`:
    - `your_format=wsj0`: Each subdirectory must itself have two subdirectories `clean/` and `noisy/`, with the same filenames present in both.
    - Add formats on your own, correpsonding to your data structure

To see all available training options, run `python train.py --help`.
These include options for the backbone DNN, the SDE parameters, the PytorchLightning Trainer usual parameters such as `max_epochs`, `limit_train_batches` and so on.

**Note:**
- This paper [1] uses a lighter configuration of the NCSN++ backbone with 27.8M parameters, which is passed with `--backbone ncsnpp` by default. By contrast, the architecture used in the paper [2] uses `--backbone ncsnpp-large` which is the baseline 65M parameters NCSN++.

## Evaluation

To evaluate on a test set, run
```bash
python enhancement.py --test_dir <your_test_dir> --enhanced_dir <your_enhanced_dir> --ckpt <path_to_model_checkpoint>
```

to generate the enhanced .wav files. The `--cpkt` parameter of `enhancement.py` should be the path to a trained model checkpoint, as stored by the logger in `logs/`.


## Data Creation

- In `preprocessing/`, you will find the data generation script used to create all the datasets used in the paper. Minimal example is:

```
    cd preprocessing;
    python3 create_data.py --task <your_task> --speech <your_speech_format> --noise <your_noise_data>
```

Please check the script for other options

## Citations / References

We kindly ask you to cite our papers in your publication when using any of our research or code:
```bib
@article{lemercier2023storm,
  author={Lemercier, Jean-Marie and Richter, Julius and Welker, Simon and Gerkmann, Timo},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={StoRM: A Diffusion-Based Stochastic Regeneration Model for Speech Enhancement and Dereverberation}, 
  year={2023},
  volume={31},
  number={},
  pages={2724-2737},
  doi={10.1109/TASLP.2023.3294692}}

```

>[1] Jean-Marie Lemercier, Julius Richter, Simon Welker, and Timo Gerkmann. "StoRM: A Stochastic Regeneration Model for Speech Enhancement And Dereverberation", *IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2724-2737*, 2023.
>
>[2] Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay and Timo Gerkmann. "Speech Enhancement and Dereverberation with Diffusion-Based Generative Models", *IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2351-2364*, 2023.
