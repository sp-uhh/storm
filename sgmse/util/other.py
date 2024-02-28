import numpy as np
import scipy.stats
import torch
import csv
import os
import glob
import tqdm
import torchaudio
import matplotlib.pyplot as plt
import time
import scipy.signal as ss

stft_kwargs = {"n_fft": 510, "hop_length": 128, "window": torch.hann_window(510), "return_complex": True}

def lsd(s_hat, s, eps=1e-10):
    S_hat, S = torch.stft(torch.from_numpy(s_hat), **stft_kwargs), torch.stft(torch.from_numpy(s), **stft_kwargs)
    logPowerS_hat, logPowerS = 2*torch.log(eps + torch.abs(S_hat)), 2*torch.log(eps + torch.abs(S))
    return torch.mean( torch.sqrt(torch.mean(torch.abs( logPowerS_hat - logPowerS ))) ).item()

def si_sdr_components(s_hat, s, n, eps=1e-10):
    # s_target
    alpha_s = np.dot(s_hat, s) / (eps + np.linalg.norm(s)**2)
    s_target = alpha_s * s

    # e_noise
    alpha_n = np.dot(s_hat, n) / (eps + np.linalg.norm(n)**2)
    e_noise = alpha_n * n

    # e_art
    e_art = s_hat - s_target - e_noise
    
    return s_target, e_noise, e_art

def energy_ratios(s_hat, s, n, eps=1e-10):
    """
    """
    s_target, e_noise, e_art = si_sdr_components(s_hat, s, n)

    si_sdr = 10*np.log10(eps + np.linalg.norm(s_target)**2 / (eps + np.linalg.norm(e_noise + e_art)**2))
    si_sir = 10*np.log10(eps + np.linalg.norm(s_target)**2 / (eps + np.linalg.norm(e_noise)**2))
    si_sar = 10*np.log10(eps + np.linalg.norm(s_target)**2 / (eps + np.linalg.norm(e_art)**2))

    return si_sdr, si_sir, si_sar

def mean_conf_int(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def mean_std(data):
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    return mean, std

class Method():
    def __init__(self, name, base_dir, metrics):
        self.name = name
        self.base_dir = base_dir
        self.metrics = {} 
        
        for i in range(len(metrics)):
            metric = metrics[i]
            value = []
            self.metrics[metric] = value 
            
    def append(self, matric, value):
        self.metrics[matric].append(value)

    def get_mean_ci(self, metric):
        return mean_conf_int(np.array(self.metrics[metric]))

def hp_filter(signal, cut_off=80, order=10, sr=16000):
    factor = cut_off /sr * 2
    sos = ss.butter(order, factor, 'hp', output='sos')
    filtered = ss.sosfilt(sos, signal)
    return filtered

def si_sdr(s, s_hat):
    alpha = np.dot(s_hat, s)/np.linalg.norm(s)**2   
    sdr = 10*np.log10(np.linalg.norm(alpha*s)**2/np.linalg.norm(
        alpha*s - s_hat)**2)
    return sdr

def si_sdr_torch(s, s_hat):
    min_len = min(s.size(-1), s_hat.size(-1))
    s, s_hat = s[..., : min_len], s_hat[..., : min_len]
    alpha = torch.dot(s_hat, s)/torch.norm(s)**2   
    sdr = 10*torch.log10(1e-10 + torch.norm(alpha*s)**2/(1e-10 + torch.norm(
        alpha*s - s_hat)**2))
    return sdr

def snr_dB(s,n):
    s_power = 1/len(s)*np.sum(s**2)
    n_power = 1/len(n)*np.sum(n**2)
    snr_dB = 10*np.log10(s_power/n_power)
    return snr_dB

def pad_spec(Y):
    T = Y.size(3)
    if T%64 !=0:
        num_pad = 64-T%64
    else:
        num_pad = 0
    pad2d = torch.nn.ZeroPad2d((0, num_pad, 0,0))
    return pad2d(Y)

# def pad_time(Y):
#     padding_target = 8320
#     T = Y.size(2)
#     if T%padding_target !=0:
#         num_pad = padding_target-T%padding_target
#     else:
#         num_pad = 0
#     pad2d = torch.nn.ZeroPad2d((0, num_pad, 0, 0))
#     return pad2d(Y)

def mean_std(data):
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    return mean, std



def init_exp_csv_samples(output_path, tag_metric):
    with open(output_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        fieldnames = ["Filename", "Length", "T60", "iSNR"] + tag_metric
        writer.writerow(fieldnames)
        csv_file.close()

def snr_scale_factor(speech, noise, snr):
    noise_var = np.var(noise)
    speech_var = np.var(speech)

    factor = np.sqrt(speech_var / (noise_var * 10. ** (snr / 10.)))

    return factor

def align(y, ref):
    l = np.argmax(ss.fftconvolve(ref.squeeze(), np.flip(y.squeeze()))) - (ref.shape[0] - 1)
    if l:
        y = torch.from_numpy(np.roll(y, l, axis=-1))
    return y

def wer(r, h):
    '''
    by zszyellow
    https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return float(d[len(r)][len(h)]) / len(r)