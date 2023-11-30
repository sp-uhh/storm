"""Example to generate a wind noise signal."""

import numpy as np
# Need access to the WindNoiseGenerator library (file: sc_wind_noise_generator.py) presented in D. Mirabilii et al. "Simulating wind noise with airflow speed-dependent characteristics,” in Int. Workshop on Acoustic Signal Enhancement, Sept. 2022"
# Please ask the authors as we are not responsible for the distribution of their code
from sc_wind_noise_generator import WindNoiseGenerator as wng
import argparse
import os
import shutil
import tqdm 

SEED = 100 # Seed for random sequence regeneration

# Parameters
wind_params = {
    "duration": 8,
    "fs": 16000,
    "gustiness_range": [1, 10],
    "wind_profile_magnitude_range": [200, 500],
    "wind_profile_acceptable_transition_threshold": 100
}

parser = argparse.ArgumentParser()

parser.add_argument('--dir', type=str)
parser.add_argument('--n', type=int, help="number of samples")
parser.add_argument('--sr', default=16000, type=int)

args = parser.parse_args()
params = vars(args)
params = {**wind_params, **params}

if os.path.exists(args.dir):
    shutil.rmtree(args.dir)
os.makedirs(args.dir, exist_ok=True)

for i in tqdm.tqdm(range(args.n)):

    # Generate wind profile
    gustiness = np.random.uniform(wind_params["gustiness_range"][0], wind_params["gustiness_range"][1]) # Number of speed points. One yields constant wind. High values yields gusty wind.
    number_points_wind_profile = int(1.5 * gustiness)
    wind_profile = [np.random.uniform(wind_params["wind_profile_magnitude_range"][0], wind_params["wind_profile_magnitude_range"][1])]

    while len(wind_profile) < number_points_wind_profile:
        is_valid = False
        while not is_valid:
            new_point = np.random.uniform(wind_params["wind_profile_magnitude_range"][0], wind_params["wind_profile_magnitude_range"][1])
            is_valid = new_point < wind_profile[-1] + wind_params["wind_profile_acceptable_transition_threshold"] and new_point > wind_profile[-1] - wind_params["wind_profile_acceptable_transition_threshold"]
        wind_profile.append(new_point)

    seed_sample = SEED + i
    # Generate wind noise
    wn = wng(fs=args.sr, duration=wind_params["duration"], generate=True,
                    wind_profile=wind_profile,
                    gustiness=gustiness, start_seed=seed_sample)
    wn_signal, wind_profile = wn.generate_wind_noise()

    # Save signal in .wav file
    wn.save_signal(wn_signal, filename=os.path.join(args.dir, f'simulated_{i}.wav'), num_ch=1, fs=args.sr)