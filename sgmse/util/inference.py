import sre_compile
import torch
from torchaudio import load
from sgmse.util.other import si_sdr, pad_spec
from pesq import pesq
from tqdm import tqdm
from pystoi import stoi
import numpy as np

# Settings
snr = 0.5
N = 50
corrector_steps = 1

# Plotting settings
MAX_VIS_SAMPLES = 10
n_fft = 512
hop_length = 128

def evaluate_model(model, num_eval_files, spec=False, audio=False, discriminative=False):

	model.eval()
	_pesq, _si_sdr, _estoi = 0., 0., 0.
	if spec:
		noisy_spec_list, estimate_spec_list, clean_spec_list = [], [], []
	if audio:
		noisy_audio_list, estimate_audio_list, clean_audio_list = [], [], []

	for i in range(num_eval_files):
		# Load wavs
		x, y = model.data_module.valid_set.__getitem__(i, raw=True) #d,t
		norm_factor = y.abs().max().item()
		x_hat = model.enhance(y)

		if x_hat.ndim == 1:
			x_hat = x_hat.unsqueeze(0)
			
		if x.ndim == 1:
			x = x.unsqueeze(0).cpu().numpy()
			x_hat = x_hat.unsqueeze(0).cpu().numpy()
			y = y.unsqueeze(0).cpu().numpy()
		else: #eval only first channel
			x = x[0].unsqueeze(0).cpu().numpy()
			x_hat = x_hat[0].unsqueeze(0).cpu().numpy()
			y = y[0].unsqueeze(0).cpu().numpy()

		x_16k = librosa.resample(x[0], orig_sr=48000, target_sr=16000)
		x_hat_16k = librosa.resample(x_hat[0], orig_sr=48000, target_sr=16000)

		_si_sdr += si_sdr(x[0], x_hat[0])
		_pesq += pesq(16000, x_hat_16k, x_16k, 'wb') 
		_estoi += stoi(x[0], x_hat[0], 48000, extended=True)
		
		y, x_hat, x = torch.from_numpy(y), torch.from_numpy(x_hat), torch.from_numpy(x)
		if spec and i < MAX_VIS_SAMPLES:
			y_stft, x_hat_stft, x_stft = model._stft(y[0]), model._stft(x_hat[0]), model._stft(x[0])
			noisy_spec_list.append(y_stft)
			estimate_spec_list.append(x_hat_stft)
			clean_spec_list.append(x_stft)

		if audio and i < MAX_VIS_SAMPLES:
			noisy_audio_list.append(y[0])
			estimate_audio_list.append(x_hat[0])
			clean_audio_list.append(x[0])

	if spec:
		if audio:
			return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, [noisy_spec_list, estimate_spec_list, clean_spec_list], [noisy_audio_list, estimate_audio_list, clean_audio_list]
		else:
			return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, [noisy_spec_list, estimate_spec_list, clean_spec_list], None
	elif audio and not spec:
			return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, None, [noisy_audio_list, estimate_audio_list, clean_audio_list]
	else:
		return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files, None, None

