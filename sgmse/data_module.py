
from os.path import join
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from torchaudio import load
import numpy as np
import torch.nn.functional as F
import h5py
import json

SEED = 10
np.random.seed(SEED)

def get_window(window_type, window_length):
	if window_type == 'sqrthann':
		return torch.sqrt(torch.hann_window(window_length, periodic=True))
	elif window_type == 'hann':
		return torch.hann_window(window_length, periodic=True)
	else:
		raise NotImplementedError(f"Window type {window_type} not implemented!")

class Specs(Dataset):
	def __init__(
		self, data_dir, subset, dummy, shuffle_spec, num_frames, format,
		normalize_audio=True, spec_transform=None, stft_kwargs=None, spatial_channels=1, 
		return_time=False,
		**ignored_kwargs
	):
		self.data_dir = data_dir
		self.subset = subset
		self.format = format
		self.spatial_channels = spatial_channels
		self.return_time = return_time
		if format in ["wsj0", "vctk"]:
			dic_correspondence_subsets = {"train": "tr", "valid": "cv", "test": "tt"}
			self.clean_files = sorted(glob(join(data_dir, dic_correspondence_subsets[subset]) + '/clean/*.wav'))
			self.noisy_files = sorted(glob(join(data_dir, dic_correspondence_subsets[subset]) + '/noisy/*.wav'))
		elif format == "voicebank":
			self.clean_files = sorted(glob(join(data_dir, subset) + '/clean/*.wav'))
			self.noisy_files = sorted(glob(join(data_dir, subset) + '/noisy/*.wav'))
		elif format == "dns":
			self.noisy_files = sorted(glob(join(data_dir, subset) + '/noisy/*.wav'))
			clean_dir = join(data_dir, subset) + '/clean/'
			self.clean_files = [clean_dir + 'clean_fileid_' \
				+ noisy_file.split('/')[-1].split('_fileid_')[-1] for noisy_file in self.noisy_files]
		elif format == "reverb_wsj0":
			dic_correspondence_subsets = {"train": "tr", "valid": "cv", "test": "tt"}
			self.clean_files = sorted(glob(join(data_dir, dic_correspondence_subsets[subset]) + '/anechoic/*.wav'))
			self.noisy_files = sorted(glob(join(data_dir, dic_correspondence_subsets[subset]) + '/reverb/*.wav'))
		elif format == "timit":
			dic_correspondence_subsets = {"train": "tr", "valid": "cv", "test": "tt"}
			self.clean_files = sorted(glob(join(data_dir, "audio", dic_correspondence_subsets[subset]) + '/clean/*.wav'))
			self.noisy_files = sorted(glob(join(data_dir, "audio", dic_correspondence_subsets[subset]) + '/noisy/*.wav'))
			self.transcriptions = sorted(glob(join(data_dir, "transcriptions", dic_correspondence_subsets[subset]) + '/*.txt'))
		elif format == "ears_wham":
			self.clean_files = sorted(glob(join(data_dir, subset, "clean", "**", "*.wav"), recursive=True))
			self.noisy_files = sorted(glob(join(data_dir, subset, "noisy", "**", "*.wav"), recursive=True))
		elif format == "reverb_vctk":
			self.clean_files = sorted(glob(join(data_dir, subset, "clean", "**", "*.wav"), recursive=True))
			self.noisy_files = sorted(glob(join(data_dir, subset, "reverberant", "**", "*.wav"), recursive=True))
			
		self.dummy = dummy
		self.num_frames = num_frames
		self.shuffle_spec = shuffle_spec
		self.normalize_audio = normalize_audio
		self.spec_transform = spec_transform

		assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
		self.stft_kwargs = stft_kwargs
		self.hop_length = self.stft_kwargs["hop_length"]
		assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"

	def __getitem__(self, i, raw=False):
		x, sr = load(self.clean_files[i])			
		y, sr = load(self.noisy_files[i])

		min_len = min(x.size(-1), y.size(-1))
		x, y = x[..., : min_len], y[..., : min_len] 
		
		if x.ndimension() == 2 and self.spatial_channels == 1:
			x, y = x[0].unsqueeze(0), y[0].unsqueeze(0) #Select first channel
		# Select channels
		assert self.spatial_channels <= x.size(0), f"You asked too many channels ({self.spatial_channels}) for the given dataset ({x.size(0)})"
		x, y = x[: self.spatial_channels], y[: self.spatial_channels]

		if raw:
			return x, y

		normfac = y.abs().max()

		# formula applies for center=True
		target_len = (self.num_frames - 1) * self.hop_length
		current_len = x.size(-1)
		pad = max(target_len - current_len, 0)
		if pad == 0:
			# extract random part of the audio file
			if self.shuffle_spec:
				start = int(np.random.uniform(0, current_len-target_len))
			else:
				start = int((current_len-target_len)/2)
			x = x[..., start:start+target_len]
			y = y[..., start:start+target_len]
		else:
			# pad audio if the length T is smaller than num_frames
			x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')
			y = F.pad(y, (pad//2, pad//2+(pad%2)), mode='constant')

		if self.normalize_audio:
			# normalize both based on noisy speech, to ensure same clean signal power in x and y.
			x = x / normfac
			y = y / normfac

		if self.return_time:
			return x, y

		X = torch.stft(x, **self.stft_kwargs)
		Y = torch.stft(y, **self.stft_kwargs)

		X, Y = self.spec_transform(X), self.spec_transform(Y)

		return X, Y

	def __len__(self):
		if self.dummy:
			# for debugging shrink the data set sizer
			return int(len(self.clean_files)/10)
		else:
			if self.format == "vctk":
				return len(self.clean_files)//2
			else:
				return len(self.clean_files)




class SpecsH5(Dataset):
	def __init__(
		self, data_dir, subset, dummy, shuffle_spec, num_frames, format,
		normalize_audio=True, spec_transform=None, stft_kwargs=None, spatial_channels=1, 
		return_time=False,
		**ignored_kwargs
	):
		self.data_dir = data_dir
		self.subset = subset
		self.format = format
		self.spatial_channels = spatial_channels
		self.return_time = return_time

		if self.data_dir.endswith(".h5"):
			assert os.path.exists(self.data_dir), f"File {self.data_dir} does not exist"
			h5_file = h5py.File(self.data_dir, 'r')
		else:
			if self.data_dir.endswith("/") or self.data_dir.endswith("\\"):
				self.data_dir = self.data_dir[:-1]
			assert os.path.exists(self.data_dir+".h5"), f"File {self.data_dir}.h5 does not exist"
			h5_file = h5py.File(self.data_dir+".h5", 'r')
		self.clean_data = h5_file[subset]['clean']
		self.noisy_data = h5_file[subset]['reverberant'] if "reverb" in format else h5_file[subset]['noisy']
		self.time_idxs = h5_file[subset]['time_idxs']

		self.dummy = dummy
		self.num_frames = num_frames
		self.shuffle_spec = shuffle_spec
		self.normalize_audio = normalize_audio
		self.spec_transform = spec_transform

		assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
		self.stft_kwargs = stft_kwargs
		self.hop_length = self.stft_kwargs["hop_length"]
		assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"

	def __getitem__(self, i, raw=False):
		idx_start, idx_end = self.time_idxs[i], self.time_idxs[i+1] #len is len(self.time_idxs) - 1 so no risk here
		x = torch.from_numpy(self.clean_data[idx_start: idx_end]).unsqueeze(0)
		y = torch.from_numpy(self.noisy_data[idx_start: idx_end]).unsqueeze(0)

		if raw:
			return x, y

		normfac = y.abs().max()

		# formula applies for center=True
		target_len = (self.num_frames - 1) * self.hop_length
		current_len = x.size(-1)
		pad = max(target_len - current_len, 0)
		if pad == 0:
			# extract random part of the audio file
			if self.shuffle_spec:
				start = int(np.random.uniform(0, current_len-target_len))
			else:
				start = int((current_len-target_len)/2)
			x = x[..., start:start+target_len]
			y = y[..., start:start+target_len]
		else:
			# pad audio if the length T is smaller than num_frames
			x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')
			y = F.pad(y, (pad//2, pad//2+(pad%2)), mode='constant')

		if self.normalize_audio:
			# normalize both based on noisy speech, to ensure same clean signal power in x and y.
			x = x / normfac
			y = y / normfac

		if self.return_time:
			return x, y

		X = torch.stft(x, **self.stft_kwargs)
		Y = torch.stft(y, **self.stft_kwargs)

		X, Y = self.spec_transform(X), self.spec_transform(Y)

		return X, Y

	def __len__(self):
		if self.dummy:
			# for debugging shrink the data set sizer
			return int((self.time_idxs.shape[0] - 1)/10)	
		else:
			return self.time_idxs.shape[0] - 1



class SpecsDataModule(pl.LightningDataModule):
	def __init__(
		self, base_dir="", format="wsj0", spatial_channels=1, batch_size=8,
		n_fft=510, hop_length=128, num_frames=256, window="hann",
		num_workers=8, dummy=False, spec_factor=0.15, spec_abs_exponent=0.5,
		gpu=True, return_time=False, **kwargs
	):
		super().__init__()
		self.base_dir = base_dir
		self.format = format
		self.spatial_channels = spatial_channels
		self.batch_size = batch_size
		self.n_fft = n_fft
		self.hop_length = hop_length
		self.num_frames = num_frames
		self.window = get_window(window, self.n_fft)
		self.windows = {}
		self.num_workers = num_workers
		self.dummy = dummy
		self.spec_factor = spec_factor
		self.spec_abs_exponent = spec_abs_exponent
		self.gpu = gpu
		self.return_time = return_time
		self.kwargs = kwargs

	def setup(self, stage=None):
		specs_kwargs = dict(
			stft_kwargs=self.stft_kwargs, num_frames=self.num_frames, spec_transform=self.spec_fwd,
			**self.stft_kwargs, **self.kwargs
		)
		if self.base_dir.endswith(".h5"):
			dataset_cls = SpecsH5
		else:
			dataset_cls = Specs

		if stage == 'fit' or stage is None:
			self.train_set = dataset_cls(self.base_dir, 'train', self.dummy, True,  
				format=self.format, spatial_channels=self.spatial_channels, 
				return_time=self.return_time, **specs_kwargs)
			self.valid_set = dataset_cls(self.base_dir, 'valid', self.dummy, False, 
				format=self.format, spatial_channels=self.spatial_channels, 
				return_time=self.return_time, **specs_kwargs)
			
		if stage == 'test' or stage is None:
			self.test_set = dataset_cls(self.base_dir, 'test', self.dummy, False, 
				format=self.format, spatial_channels=self.spatial_channels, 
				return_time=self.return_time, **specs_kwargs)

	def spec_fwd(self, spec):
		if self.spec_abs_exponent != 1:
			e = self.spec_abs_exponent
			spec = spec.abs()**e * torch.exp(1j * spec.angle())
		return spec * self.spec_factor

	def spec_back(self, spec):
		spec = spec / self.spec_factor
		if self.spec_abs_exponent != 1:
			e = self.spec_abs_exponent
			spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
		return spec

	@property
	def stft_kwargs(self):
		return {**self.istft_kwargs, "return_complex": True}

	@property
	def istft_kwargs(self):
		return dict(
			n_fft=self.n_fft, hop_length=self.hop_length,
			window=self.window, center=True
		)

	def _get_window(self, x):
		"""
		Retrieve an appropriate window for the given tensor x, matching the device.
		Caches the retrieved windows so that only one window tensor will be allocated per device.
		"""
		window = self.windows.get(x.device, None)
		if window is None:
			window = self.window.to(x.device)
			self.windows[x.device] = window
		return window

	def stft(self, sig):
		window = self._get_window(sig)
		return torch.stft(sig, **{**self.stft_kwargs, "window": window})

	def istft(self, spec, length=None):
		window = self._get_window(spec)
		return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})

	@staticmethod
	def add_argparse_args(parser):
		parser.add_argument("--format", type=str, default="wsj0", choices=["reverb_vctk", "ears_wham", "wsj0", "vctk", "dns", "reverb_wsj0", "timit", "voicebank"], help="File paths follow the DNS data description.")
		parser.add_argument("--base_dir", type=str, default="/data/lemercier/databases/wsj0+chime_julian/audio",
			help="The base directory of the dataset. Should contain `train`, `valid` and `test` subdirectories, "
				"each of which contain `clean` and `noisy` subdirectories.")
		parser.add_argument("--batch_size", type=int, default=8, help="The batch size. 32 by default.")
		parser.add_argument("--n_fft", type=int, default=510, help="Number of FFT bins. 510 by default.")   # to assure 256 freq bins
		parser.add_argument("--hop_length", type=int, default=128, help="Window hop length. 128 by default.")
		parser.add_argument("--num_frames", type=int, default=256, help="Number of frames for the dataset. 256 by default.")
		parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann", help="The window function to use for the STFT. 'sqrthann' by default.")
		parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to use for DataLoaders. 4 by default.")
		parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")
		parser.add_argument("--spec_factor", type=float, default=0.15, help="Factor to multiply complex STFT coefficients by.")
		parser.add_argument("--spec_abs_exponent", type=float, default=0.5,
			help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). "
				"1 by default; set to values < 1 to bring out quieter features.")
		parser.add_argument("--return_time", action="store_true", help="Return the waveform instead of the STFT")

		return parser

	def train_dataloader(self):
		# return DataLoader(
		return DataLoader(
			self.train_set, batch_size=self.batch_size,
			num_workers=self.num_workers, pin_memory=self.gpu, shuffle=True
		)

	def val_dataloader(self):
		# return DataLoader(
		return DataLoader(
			self.valid_set, batch_size=self.batch_size,
			num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
		)

	def test_dataloader(self):
		# return DataLoader(
		return DataLoader(
			self.test_set, batch_size=self.batch_size,
			num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
		)














class SpecsAndTranscriptions(Specs):

	def __init__(
		self, data_dir, subset, dummy, shuffle_spec, num_frames, format,
		**kwargs
	):
		super().__init__(data_dir, subset, dummy, shuffle_spec, num_frames, format, **kwargs)
		if format == "timit":
			dic_correspondence_subsets = {"train": "tr", "valid": "cv", "test": "tt"}
			self.clean_files = sorted(glob(join(data_dir, "audio", dic_correspondence_subsets[subset]) + '/clean/*.wav'))
			self.noisy_files = sorted(glob(join(data_dir, "audio", dic_correspondence_subsets[subset]) + '/noisy/*.wav'))
			self.transcriptions = sorted(glob(join(data_dir, "transcriptions", dic_correspondence_subsets[subset]) + '/*.txt'))
		else:
			raise NotImplementedError

	def __getitem__(self, i, raw=False):
		X, Y = super().__getitem__(i, raw=raw)
		transcription = open(self.transcriptions[i], "r").read()
		if self.format == "timit": #remove the number at the beginning
			transcription = " ".join(transcription.split(" ")[2: ])

		return X, Y, transcription

	def __len__(self):
		if self.dummy:
			return int(len(self.clean_files)/10)
		else:
			return len(self.clean_files)

class SpecsAndTranscriptionsDataModule(SpecsDataModule):

	def setup(self, stage=None):
		specs_kwargs = dict(
			stft_kwargs=self.stft_kwargs, num_frames=self.num_frames, spec_transform=self.spec_fwd,
			**self.stft_kwargs, **self.kwargs
		)
		if stage == 'fit' or stage is None:
			raise NotImplementedError
		if stage == 'test' or stage is None:
			self.test_set = SpecsAndTranscriptions(self.base_dir, 'test', self.dummy, False, 
			format=self.format, **specs_kwargs)


	@staticmethod
	def add_argparse_args(parser):
		parser.add_argument("--format", type=str, default="reverb_wsj0", choices=["wsj0", "vctk", "dns", "reverb_wsj0"], help="File paths follow the DNS data description.")
		parser.add_argument("--base-dir", type=str, default="/data/lemercier/databases/reverb_wsj0+chime/audio")
		parser.add_argument("--batch-size", type=int, default=8, help="The batch size.")
		parser.add_argument("--num-workers", type=int, default=8, help="Number of workers to use for DataLoaders.")
		parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")
		return parser