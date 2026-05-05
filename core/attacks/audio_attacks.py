"""
Audio attack functions for steganography robustness evaluation.
Each function takes (audio: np.ndarray, sr: int) and returns attacked audio.
"""

import numpy as np
from scipy import signal as sp_signal


def no_attack(audio, sr):
    return audio.copy()


def awgn_30db(audio, sr):
    sig_pow = np.mean(audio ** 2) + 1e-12
    noise_pow = sig_pow / (10 ** (30 / 10))
    noise = np.random.normal(0, np.sqrt(noise_pow), audio.shape)
    return np.clip(audio + noise, -1.0, 1.0)


def awgn_20db(audio, sr):
    sig_pow = np.mean(audio ** 2) + 1e-12
    noise_pow = sig_pow / (10 ** (20 / 10))
    noise = np.random.normal(0, np.sqrt(noise_pow), audio.shape)
    return np.clip(audio + noise, -1.0, 1.0)


def low_pass_4khz(audio, sr):
    nyq = sr / 2
    cutoff = min(4000, nyq * 0.9)
    b, a = sp_signal.butter(4, cutoff / nyq, btype="low")
    if audio.ndim == 1:
        return sp_signal.filtfilt(b, a, audio).astype(np.float32)
    return np.stack([sp_signal.filtfilt(b, a, audio[:, c]) for c in range(audio.shape[1])], axis=1).astype(np.float32)


def low_pass_8khz(audio, sr):
    nyq = sr / 2
    cutoff = min(8000, nyq * 0.9)
    b, a = sp_signal.butter(4, cutoff / nyq, btype="low")
    if audio.ndim == 1:
        return sp_signal.filtfilt(b, a, audio).astype(np.float32)
    return np.stack([sp_signal.filtfilt(b, a, audio[:, c]) for c in range(audio.shape[1])], axis=1).astype(np.float32)


def volume_scale_80(audio, sr):
    return np.clip(audio * 0.8, -1.0, 1.0)


def volume_scale_120(audio, sr):
    return np.clip(audio * 1.2, -1.0, 1.0)


def time_shift_100(audio, sr):
    return np.roll(audio, 100, axis=0)


def resample_attack(audio, sr):
    target = sr // 2  # downsample to half, then upsample back
    if audio.ndim == 1:
        down = sp_signal.resample_poly(audio, 1, 2)
        up = sp_signal.resample_poly(down, 2, 1)
        return up[:len(audio)].astype(np.float32)
    channels = []
    for c in range(audio.shape[1]):
        down = sp_signal.resample_poly(audio[:, c], 1, 2)
        up = sp_signal.resample_poly(down, 2, 1)
        channels.append(up[:len(audio)])
    return np.stack(channels, axis=1).astype(np.float32)


def echo_attack(audio, sr):
    delay = int(sr * 0.05)  # 50ms echo
    echo = np.zeros_like(audio)
    if audio.ndim == 1:
        echo[delay:] = audio[:-delay] * 0.3
    else:
        echo[delay:] = audio[:-delay] * 0.3
    return np.clip(audio + echo, -1.0, 1.0)


AUDIO_ATTACKS = [
    ("No Attack (Baseline)",        no_attack),
    ("AWGN SNR=30 dB",              awgn_30db),
    ("AWGN SNR=20 dB",              awgn_20db),
    ("Low-Pass Filter 4 kHz",       low_pass_4khz),
    ("Low-Pass Filter 8 kHz",       low_pass_8khz),
    ("Volume Scale ×0.8",           volume_scale_80),
    ("Volume Scale ×1.2",           volume_scale_120),
    ("Resampling (÷2 then ×2)",     resample_attack),
    ("Time Shift 100 samples",      time_shift_100),
    ("Echo (50 ms, 0.3×)",          echo_attack),
]
