import numpy as np
import glob

from tensorboard import summary
from tqdm import tqdm
from torchaudio import load, save
import torch
import os
from argparse import ArgumentParser
import time
from pypapi import events, papi_high as high

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.model import StochasticRegenerationModel, ScoreModel, DiscriminativeModel

from sgmse.util.other import *

import matplotlib.pyplot as plt

EPS_LOG = 1e-10

# Tags
base_parser = ArgumentParser(add_help=False)
parser = ArgumentParser()
for parser_ in (base_parser, parser):
	parser_.add_argument("--test_dir", type=str, required=True, help="Directory containing your corrupted files to enhance.")
	parser_.add_argument("--enhanced_dir", type=str, required=True, help="Where to write your cleaned files.")
	parser_.add_argument("--ckpt", type=str, required=True)
	parser_.add_argument("--mode", default="storm", choices=["score-only", "denoiser-only", "storm"])

	parser_.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
	parser_.add_argument("--corrector-steps", type=int, default=1, help="Number of corrector steps")
	parser_.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynamics.")
	parser_.add_argument("--N", type=int, default=50, help="Number of reverse steps")

args = parser.parse_args()

os.makedirs(args.enhanced_dir, exist_ok=True)

#Checkpoint
checkpoint_file = args.ckpt

# Settings
model_sr = 16000

# Load score model 
if args.mode == "storm":
	model_cls = StochasticRegenerationModel
elif args.mode == "score-only":
	model_cls = ScoreModel
elif args.mode == "denoiser-only":
	model_cls = DiscriminativeModel

model = model_cls.load_from_checkpoint(
	checkpoint_file, base_dir="",
	batch_size=1, num_workers=0, kwargs=dict(gpu=False)
)
model.eval(no_ema=False)
model.cuda()

noisy_files = sorted(glob.glob(os.path.join(args.test_dir, "*.wav")))

# Loop on files
for f in tqdm.tqdm(noisy_files):

	y, sample_sr = torchaudio.load(f)
	assert sample_sr == model_sr, "You need to make sure sample_sr matches model_sr --> resample to 16kHz"
	x_hat = model.enhance(y, corrector=args.corrector, N=args.N, corrector_steps=args.corrector_steps, snr=args.snr)

	save(f'{args.enhanced_dir}/{os.path.basename(f)}', x_hat.type(torch.float32).cpu().squeeze().unsqueeze(0), model_sr)