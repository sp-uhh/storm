import torch
from torchaudio import load
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf
import glob

# Plotting settings
EPS_graphics = 1e-10
n_fft = 512
hop_length = 128

stft_kwargs = {"n_fft": n_fft, "hop_length": hop_length, "window": torch.hann_window(n_fft), "center": True, "return_complex": True}

def visualize_example(mix, estimate, target, idx_sample=0, epoch=0, name="", sample_rate=16000, hop_len=128, return_fig=False):
	"""Visualize training targets and estimates of the Neural Network
	Args:
		- mix: Tensor [F, T]
		- estimates/targets: Tensor [F, T]
	"""

	if isinstance(mix, torch.Tensor):
		mix = torch.abs(mix).detach().cpu()
		estimate = torch.abs(estimate).detach().cpu()
		target = torch.abs(target).detach().cpu()

	vmin, vmax = -60, 0

	fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))

	freqs = sample_rate/(2*mix.size(-2)) * torch.arange(mix.size(-2))
	frames = hop_len/sample_rate * torch.arange(mix.size(-1))

	ax = axes.flat[0]
	im = ax.pcolormesh(frames, freqs, 20*np.log10(.1*mix + EPS_graphics), vmin=vmin, vmax=vmax, shading="auto", cmap="magma")
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Frequency [Hz]')
	ax.set_title('Mixed Speech')

	ax = axes.flat[1]
	ax.pcolormesh(frames, freqs, 20*np.log10(.1*estimate + EPS_graphics), vmin=vmin, vmax=vmax, shading="auto", cmap="magma")
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Frequency [Hz]')
	ax.set_title('Anechoic estimate')

	ax = axes.flat[2]
	ax.pcolormesh(frames, freqs, 20*np.log10(.1*target + EPS_graphics), vmin=vmin, vmax=vmax, shading="auto", cmap="magma")
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Frequency [Hz]')
	ax.set_title('Anechoic target')

	fig.subplots_adjust(right=0.87)
	cbar_ax = fig.add_axes([0.9, 0.25, 0.005, 0.5])
	fig.colorbar(im, cax=cbar_ax)

	if return_fig:
		return fig
	else:
		plt.savefig(os.path.join(spec_path, f"spectro_{idx_sample}_epoch{epoch}{name}.png"), bbox_inches="tight")
		plt.close()


def visualize_one(estimate, spec_path, name="", sample_rate=16000, hop_len=128, raw=True):
	"""Visualize training targets and estimates of the Neural Network
	Args:
		- mix: Tensor [F, T]
		- estimates/targets: Tensor [F, T]
	"""

	if isinstance(estimate, torch.Tensor):
		estimate = torch.abs(estimate).squeeze().detach().cpu()
	elif type(estimate) == str:
		estimate = np.squeeze(sf.read(estimate)[0])
		norm_factor = 0.1/np.max(np.abs(estimate))
		xmax = 6
		estimate = estimate[..., : xmax*sample_rate]
		estimate = torch.stft(torch.from_numpy(norm_factor*estimate), **stft_kwargs)

	vmin, vmax = -60, 0

	freqs = sample_rate/(2*estimate.size(-2)) * torch.arange(estimate.size(-2))
	frames = hop_len/sample_rate * torch.arange(estimate.size(-1))

	fig = plt.figure(figsize=(8, 8))
	im = plt.pcolormesh(frames, freqs, 20*np.log10(estimate.abs() + EPS_graphics), vmin=vmin, vmax=vmax, shading="auto", cmap="magma")

	if raw:
		plt.yticks([])
		plt.tick_params(left="off")
		plt.xticks([])
		plt.tick_params(bottom="off")
	else:
		plt.xlabel('Time [s]')
		plt.ylabel('Frequency [Hz]')
		plt.title('Anechoic estimate')
		cbar_ax = fig.add_axes([0.93, 0.25, 0.03, 0.4])
		fig.colorbar(im, cax=cbar_ax)

	plt.savefig(os.path.join(spec_path, name + ".png"), dpi=300, bbox_inches="tight")
	plt.close()
