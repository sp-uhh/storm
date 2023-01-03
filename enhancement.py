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
from sgmse.model import ScoreRefinerModel, ScoreModel, DiscriminativeModel

from sgmse.util.other import *

import matplotlib.pyplot as plt

EPS_LOG = 1e-10

# Tags
base_parser = ArgumentParser(add_help=False)
parser = ArgumentParser()
for parser_ in (base_parser, parser):
	parser_.add_argument("--task", type=str, required=True, choices=["enh", "sep", "derev", "derev+enh", "bwe"])
	parser_.add_argument("--ckpt", type=str, required=True)

	parser_.add_argument("--sde", type=str, choices=["FreqOUVESDE", "OUVESDE", "VESDE"], default="OUVESDE")
	parser_.add_argument("--backbone-score", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
	parser_.add_argument("--backbone-denoiser", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
	parser_.add_argument("--backbone-denoiser-mismatch", type=str, default="none")
	parser_.add_argument("--mode", default="refine", choices=["score-only", "denoiser-only", "regen-p", "regen-j", "regen-p+j"])
	parser_.add_argument("--condition", type=str, choices=["noisy", "post_denoiser", "both"], default="both")

	parser_.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
	parser_.add_argument("--corrector-steps", type=int, default=1, help="Number of corrector steps")
	parser_.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynamics.")
	parser_.add_argument("--N", type=int, default=50, help="Number of reverse steps")
	parser_.add_argument("--denoiser-only", action="store_true")

	parser_.add_argument("--format", type=str, default="wsj0")
	parser_.add_argument("--base-dir", type=str, default="/data/lemercier/databases/wsj0+chime_julian/audio")
	parser_.add_argument("--n-ch", type=int, default=1, help="Number of channels of data")

	parser_.add_argument("--copy-audio", action="store_true")
	parser_.add_argument("--dummy", action="store_true")
	parser_.add_argument("--gpu", type=int, default=0)

	parser_.add_argument("--cross", action="store_true")
	parser_.add_argument("--task-cross", default=None)
	parser_.add_argument("--format-cross", default=None)
	parser_.add_argument("--base-dir-cross", default=None)

args = parser.parse_args()

format_display = args.format if not args.cross else f"{args.format}_cross_{args.format_cross}"
actual_format = args.format_cross if args.cross else args.format
actual_task = args.task_cross if (args.cross and args.task_cross is not None) else args.task

if "regen" in args.mode:
	model_tag = f"mode={args.mode}_sde={args.sde}_score={args.backbone_score}_denoiser={args.backbone_denoiser}_condition={args.condition}_data={args.format}_ch={args.n_ch}"
	enhanced_dir = f'.exp/.{actual_task}/enhanced/{format_display}/{model_tag}_corrector={args.corrector}_snr={args.snr}_N={args.N}/'
elif args.mode == "score-only":
	model_tag_display = f"mode={args.mode}_sde={args.sde}_score={args.backbone_score}_condition=noisy_data={args.format}_ch={args.n_ch}"
	model_tag = f"mode={args.mode}_sde={args.sde}_backbone={args.backbone_score}_data={args.format}_ch={args.n_ch}"
	enhanced_dir = f'.exp/.{actual_task}/enhanced/{format_display}/{model_tag_display}_corrector={args.corrector}_snr={args.snr}_N={args.N}/'
elif args.mode == "denoiser-only":
	model_tag_display = f"mode={args.mode}_sde={args.sde}_denoiser={args.backbone_denoiser}_data={args.format}_ch={args.n_ch}"
	model_tag = f"mode={args.mode}_sde={args.sde}_backbone={args.backbone_denoiser}_data={args.format}_ch={args.n_ch}"
	enhanced_dir = f'.exp/.{actual_task}/enhanced/{format_display}/{model_tag_display}/'

torch.cuda.set_device(args.gpu)
os.makedirs(enhanced_dir, exist_ok=True)

#Checkpoint
checkpoint_file = args.ckpt

# Settings
sr = 16000

# Load score model 
if "regen" in args.mode:
	model_cls = ScoreRefinerModel
elif args.mode == "score-only":
	model_cls = ScoreModel
elif args.mode == "denoiser-only":
	model_cls = DiscriminativeModel

model = model_cls.load_from_checkpoint(
	checkpoint_file, base_dir=args.base_dir,
	batch_size=1, num_workers=0, kwargs=dict(gpu=False)
)

if args.backbone_denoiser_mismatch != "none":
	assert "regen" in args.mode, "You asked for a Predictor mismatch but the approach is not a refining scheme"
	assert args.backbone_denoiser_mismatch[-5 :] == ".ckpt", "Predictor mismatch is not a ckpt file"
	model.load_denoiser_model(args.backbone_denoiser_mismatch)
	model._error_loading_ema = True
	model.eval(no_ema=True)
	model_tag = f"mode={args.mode}_sde={args.sde}_score={args.backbone_score}_denoiser=mismatched_{type(model.denoiser_net).__name__}_condition={args.condition}_data={args.format}_ch={args.n_ch}"
	enhanced_dir = f'.exp/.{actual_task}/enhanced/{args.format}/{model_tag}_corrector={args.corrector}_snr={args.snr}_N={args.N}/'
	os.makedirs(enhanced_dir, exist_ok=True)
else:
	model.eval(no_ema=False)

model.cuda()



if args.cross:
	preprocess_kwargs = {
		"n_fft": model.data_module.n_fft,
		"hop_length": model.data_module.hop_length,
		"spec_factor": model.data_module.spec_factor,
		"spec_abs_exponent": model.data_module.spec_abs_exponent
		}
	data_module = SpecsDataModule(
		task=args.task_cross,
		format=args.format_cross, 
		base_dir=args.base_dir_cross, 
		batch_size=1, 
		**preprocess_kwargs)
else:
	data_module = model.data_module

data_module.dummy = args.dummy
data_module.setup(stage="test")
dataset = data_module.test_set

n_files = len(dataset)

# Loop on files
for i in tqdm.tqdm(range(n_files)):

	filename = os.path.basename(dataset.noisy_files[i])
	x, y = dataset.__getitem__(i, raw=True) #d,t
	x_hat = model.enhance(y, corrector=args.corrector, N=args.N, corrector_steps=args.corrector_steps, snr=args.snr)

	x = x.type(torch.float32).cpu().squeeze()
	y = y.type(torch.float32).cpu().squeeze()

	x = x.unsqueeze(0) if x.ndim == 1 else x[0].unsqueeze(0)
	y = y.unsqueeze(0) if y.ndim == 1 else y[0].unsqueeze(0)

	save(f'{enhanced_dir}/{filename}', x_hat.type(torch.float32).cpu().squeeze().unsqueeze(0), 16000)