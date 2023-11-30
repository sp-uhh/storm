import subprocess

ffmpeg = "/usr/local/bin/ffmpeg"


#!/usr/env/bin/python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from os.path import join
import numpy as np
import soundfile as sf
import glob
import argparse
import time
import json
from tqdm import tqdm
import shutil
import scipy.signal as ss
import io 
import scipy.io.wavfile 
import pyroomacoustics as pra

from utils import obtain_noise_file

SEED = 100
np.random.seed(SEED)


def buildFFmpegCommand(params):

    filter_commands = ""
    filter_commands += "[1:a]asplit=2[sc][mix];"
    filter_commands += "[0:a][sc]sidechaincompress=" + \
        f"threshold={params['threshold']}:" + \
        f"ratio={params['ratio']}:" + \
        f"level_sc={params['sc_gain']}" + \
        f":release={params['release']}" + \
        f":attack={params['attack']}" + \
        "[compr];"
    filter_commands += "[compr][mix]amix"

    commands_list = [
        "ffmpeg",
        "-y",
        "-i",
        params["speech_path"],
        "-i",
        params["noise_path"],
        "-filter_complex",
        filter_commands,
        params["output_path"]
        ]

    # return (" ").join(commands_list)
    return commands_list




params = {
    "snr_range": [-6, 14],
    "threshold_range": [0.1, 0.3],
    "ratio_range": [1, 20],
    "attack_range": [5, 100],
    "release_range": [5, 500],
    "sc_gain_range": [0.8, 1.2],
    "clipping_threshold_range": [0.85, 1.],
    "clipping_chance": 1.,
}

# ROOT = "" ## put your root directory here
ROOT = "/data/lemercier/databases"
assert ROOT != "", "You need to have a root databases directory"

parser = argparse.ArgumentParser()

parser.add_argument('--speech_dir', type=str, help='Clean speech', default="/data/lemercier/databases/wsj0+chime3/audio/{}/clean") #Put the correct regexp for your paths here
parser.add_argument('--noise_dir', type=str, help='Noise', default="/data/lemercier/databases/wind_noise_16k/{}/**/") #Put the correct regexp for your paths here
parser.add_argument('--sr', type=int, default=16000)
parser.add_argument('--dummy', action="store_true", help='Number of samples')

args = parser.parse_args()

output_dir = join(ROOT, "speech_in_noise_nonlinear")
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
log = open(join(output_dir, "log_stats.txt"), "w")
log.write("Parameters \n ========== \n")
for key, param in params.items():
    log.write(key + " : " + str(param) + "\n")

for i_split, split in enumerate(["cv", "tr", "tt"]):

    print("Processing split {}...".format(split))

    speech_split = sorted(glob.glob(join(args.speech_dir.format(split), "*.wav")))
    noise_split = sorted(glob.glob(join(args.noise_dir.format(split), "*.wav")))

    clean_output_dir = join(output_dir, split, "clean")
    noisy_output_dir = join(output_dir, split, "noisy")
    os.makedirs(clean_output_dir, exist_ok=True)
    os.makedirs(noisy_output_dir, exist_ok=True)

    real_nb_samples = 5 if args.dummy else len(speech_split)

    os.makedirs(join(output_dir, ".cache_noise"), exist_ok=True)
    os.makedirs(join(output_dir, ".cache_output"), exist_ok=True)

    for i in tqdm(range(real_nb_samples)):

        speech, sr = sf.read(speech_split[i])
        assert sr == args.sr, "Obtained an unexpected Sampling rate"
        i_noise = np.random.randint(len(noise_split))
        noise, sr = sf.read(noise_split[i_noise])
        assert sr == args.sr, "Obtained an unexpected Sampling rate"
    
        speech_scale = np.max(np.abs(speech))
        noise_scale = np.max(np.abs(noise))

        if noise.shape[0] < speech.shape[0]:
            noise = np.pad(noise, ((0, speech.shape[0] - noise.shape[0])))
        else:
            offset = np.random.randint(noise.shape[0] - speech.shape[0])
            noise = noise[offset: offset + speech.shape[0]]

        snr = np.random.uniform(params["snr_range"][0], params["snr_range"][1])
        noise_power = 1/noise.shape[0]*np.sum(noise**2)
        speech_power = 1/speech.shape[0]*np.sum(speech**2)
        noise_power_target = speech_power*np.power(10, -snr/10)
        noise_scaling = np.sqrt(noise_power_target / noise_power)
        
        noise_tmp = join(output_dir, ".cache_noise", "noise_tmp.wav")
        sf.write(noise_tmp, noise*noise_scaling, sr)

        # Compressor
        threshold = np.random.uniform(params["threshold_range"][0], params["threshold_range"][1])
        ratio = np.random.uniform(params["ratio_range"][0], params["ratio_range"][1])
        attack = np.random.uniform(params["attack_range"][0], params["attack_range"][1])
        release = np.random.uniform(params["release_range"][0], params["release_range"][1])
        sc_gain = np.random.uniform(params["sc_gain_range"][0], params["sc_gain_range"][1])

        output_tmp = join(output_dir, ".cache_output", "output_tmp.wav")

        commands = buildFFmpegCommand({
            "speech_path": speech_split[i],
            "noise_path": noise_tmp, 
            "output_path": output_tmp,
            "threshold": threshold,
            "ratio": ratio,
            "attack": attack,
            "release": release,
            "sc_gain": sc_gain
        })

        print(commands)
        if subprocess.run(commands).returncode != 0:
            print ("There was an error running your FFmpeg script")

        # Clipper
        mix, sr = sf.read(output_tmp)
        if np.random.random() < params["clipping_chance"]:
            clipping_threshold = np.random.uniform(params["clipping_threshold_range"][0], params["clipping_threshold_range"][1])
            mix = np.maximum(clipping_threshold * np.min(mix)*np.ones_like(mix), mix) 
            mix = np.minimum(clipping_threshold * np.max(mix)*np.ones_like(mix), mix)

        # Export
        output =  os.path.basename(speech_split[i])[: -4] + f"_{i}_snr={snr:.1f}.wav"
        sf.write(join(noisy_output_dir, output), mix, sr)
        sf.write(join(clean_output_dir, os.path.basename(speech_split[i])), speech, sr)

    shutil.rmtree(join(output_dir, ".cache_noise"))
    shutil.rmtree(join(output_dir, ".cache_output"))
    