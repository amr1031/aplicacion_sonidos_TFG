#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import wave
import os
import math
import numpy as np
import scipy.signal
import scipy.io.wavfile as wf

# --- Audio file reading and resampling ---
def read_wav_file(str_filename, target_rate):
    """
    Lee un archivo WAV (incluye 24 bits) y devuelve (target_rate, data_float32).
    """
    wav = wave.open(str_filename, mode='r')
    sample_rate, data = _extract2FloatArr(wav, str_filename)
    if sample_rate != target_rate:
        sample_rate, data = _resample(sample_rate, data, target_rate)
    wav.close()
    return sample_rate, data.astype(np.float32)

def _resample(current_rate, data, target_rate):
    x_orig = np.linspace(0, 100, len(data))
    x_new = np.linspace(0, 100, int(len(data) * (target_rate / current_rate)))
    y = np.interp(x_new, x_orig, data)
    return target_rate, y.astype(np.float32)

def _extract2FloatArr(lp_wave, str_filename):
    """
    Extrae datos de una wave en formato float32, maneja 8/16/24/32 bits.
    """
    bps, _ = _bitrate_channels(lp_wave)
    if bps in [1, 2, 4]:
        rate, data = wf.read(str_filename)
        if bps in [1, 2]:
            divisor = {1: 255, 2: 32768}[bps]
            data = data.astype(np.float32) / float(divisor)
        return rate, data
    elif bps == 3:
        return _read24bitwave(lp_wave)
    else:
        raise Exception(f'Formato de audio no reconocido: {bps} bytes por muestra')

def _read24bitwave(lp_wave):
    """
    Convierte 24-bit WAV a float32.
    """
    n_frames = lp_wave.getnframes()
    buf = lp_wave.readframes(n_frames)
    arr = np.frombuffer(buf, dtype=np.int8).reshape(n_frames, -1)
    short_out = np.empty((n_frames, 2), dtype=np.int8)
    short_out[:, :] = arr[:, -2:]
    short_out = short_out.view(np.int16)
    data = short_out.reshape(-1).astype(np.float32) / 32768.0
    return lp_wave.getframerate(), data

def _bitrate_channels(lp_wave):
    """
    Devuelve (bytes_por_muestra, canales).
    """
    bps = lp_wave.getsampwidth() / lp_wave.getnchannels()
    return bps, lp_wave.getnchannels()

def slice_data(start, end, raw_data, sample_rate):
    """
    Corta raw_data entre tiempos start y end (en segundos).
    """
    max_i = len(raw_data)
    i0 = min(int(start * sample_rate), max_i)
    i1 = min(int(end * sample_rate), max_i)
    return raw_data[i0:i1]

# --- Mel-spectrogramas y VTLP ---
def sample2MelSpectrum(cycle_info, sample_rate, n_filters, vtlp_params=None):
    """
    Convierte un fragmento de audio en Mel-espectrograma normalizado.
    cycle_info: (np.array_audio, crackles_flag, wheezes_flag)
    Devuelve: (mel_log_norm reshape (n_filters, time_steps,1), onehot_label)
    """
    audio, crackles, wheezes = cycle_info
    n_win = 512
    f, t, Sxx = scipy.signal.spectrogram(audio, fs=sample_rate, nfft=n_win, nperseg=n_win)
    Sxx = Sxx[:175, :].astype(np.float32)
    _, mel_log = FFT2MelSpectrogram(f[:175], Sxx, sample_rate, n_filters, vtlp_params)
    mn, mx = mel_log.min(), mel_log.max()
    diff = mx - mn
    norm = (mel_log - mn) / diff if diff > 0 else np.zeros_like(mel_log)
    label = label2onehot((crackles, wheezes))
    return norm.reshape(n_filters, mel_log.shape[1], 1), label

def Freq2Mel(freq):
    return 1125 * np.log(1 + freq / 700.0)

def Mel2Freq(mel):
    return 700 * (np.exp(mel / 1125.0) - 1)

def VTLP_shift(mel_freq, alpha, f_high, sample_rate):
    nyq = sample_rate / 2.0
    warp = min(alpha, 1.0)
    thresh = f_high * warp / alpha
    lower = mel_freq * alpha
    higher = nyq - (nyq - mel_freq) * ((nyq - f_high * warp) / (nyq - f_high * (warp / alpha)))
    return np.where(mel_freq <= thresh, lower, higher).astype(np.float32)

def GenerateMelFilterBanks(mel_space_freq, fft_bin_freqs):
    n_filt = len(mel_space_freq) - 2
    banks = []
    for m in range(1, n_filt + 1):
        f_m_minus, f_m, f_m_plus = mel_space_freq[m-1], mel_space_freq[m], mel_space_freq[m+1]
        bank = [(0 if f < f_m_minus else
                 (f - f_m_minus)/(f_m - f_m_minus) if f < f_m else
                 (f_m_plus - f)/(f_m_plus - f_m) if f < f_m_plus else 0)
                for f in fft_bin_freqs]
        banks.append(bank)
    return np.array(banks, dtype=np.float32)

def FFT2MelSpectrogram(f, Sxx, sample_rate, n_filterbanks, vtlp_params=None):
    mel_max, mel_min = Freq2Mel(f.max()), Freq2Mel(f.min())
    mel_bins = np.linspace(mel_min, mel_max, n_filterbanks+2)
    mel_freqs = Mel2Freq(mel_bins)
    if vtlp_params is None:
        fbanks = GenerateMelFilterBanks(mel_freqs, f)
    else:
        alpha, f_high = vtlp_params
        warped = VTLP_shift(mel_freqs, alpha, f_high, sample_rate)
        fbanks = GenerateMelFilterBanks(warped, f)
    mel = np.matmul(fbanks, Sxx)
    return mel_freqs[1:-1], np.log10(mel + 1e-11)

def label2onehot(flags):
    c, w = flags
    if not c and not w: return [1,0,0,0]
    if c and not w:     return [0,1,0,0]
    if not c and w:     return [0,0,1,0]
    return [0,0,0,1]

