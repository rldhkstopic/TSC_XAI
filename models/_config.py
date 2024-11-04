import torch
import numpy as np
from scipy.signal import stft
import torch.nn.utils.rnn as rnn_utils
from scipy.signal import stft, get_window



data_dir = '/data/kiwan/LPI_KIWAN/'
snr_values = [0, 2, 4, 6, 8, 10, 12, 14, 16]
datatypes = ['Signal', 'Noise', 'Noisy', 'pwnNoisy']
modeltypes = ['BiLSTM', 'UNet', 'U2Net']
waveforms = ['Barker', 'Costas', 'Frank', 'LFM', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4']
label_mapping = {signal: idx for idx, signal in enumerate(waveforms)}
        
typeSize = 12
fs = 100e6

def collate(batch):
    if len(batch[0])==5:  
        data, labels, lengths, snr, idx = zip(*batch)
        data = [torch.tensor(seq, dtype=torch.float32) for seq in data]
        data_pad = rnn_utils.pad_sequence(data, batch_first=True)

        label_to_index = {label: idx for idx, label in enumerate(waveforms)}
        labels = torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)
        lengths = torch.tensor(lengths, dtype=torch.int64)
        return data_pad, labels, lengths, snr, idx

    else:  # UNet 또는 U2Net 모델
        data, labels, snr, idx = zip(*batch)
        data = torch.stack(data, dim=0)
        label_to_index = {label: idx for idx, label in enumerate(waveforms)}
        labels = torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)
        return data, labels, None, snr, idx


def fft_transform(data_real, data_imag=None, fs=100e6):
    if data_imag is not None:
        complex_signal = data_real + 1j * data_imag
    else:
        complex_signal = data_real
    fft_data = np.fft.fft(complex_signal)
    fft_amp = np.abs(fft_data)                
    fft_freq = np.fft.fftfreq(len(fft_data), 1/fs)           
    return fft_freq[:len(fft_freq)//2] ,fft_amp[:len(fft_amp) // 2], fft_data

def stft_transform(data_real, data_imag, fs=100e6, nperseg=256):
    complex_signal = data_real + 1j * data_imag
    window = get_window('hann', 256)
    f, t, Zxx = stft(complex_signal, fs=fs, nperseg=nperseg, window=window)
    stft_amp = np.abs(Zxx)
    return f, t, stft_amp


def top_k(rel, k=10):
    return np.where(rel >= np.percentile(rel, 100-k), rel, 1e-6)

def lowpass(freq, amp, cutoff=1e6):
    low_pass = np.where(freq < cutoff)
    amp[low_pass] = 0
    return amp

def comp(real, imag):
    return real + 1j * imag












def snr_string(snr):
    return f'{snr}' if snr==0 else f'-{snr}'
    
def normalize(*data_arr):
    norms = []
    for data in data_arr:
        min_val = np.min(data)
        max_val = np.max(data)
        norm = (data - min_val) / (max_val - min_val)
        norms.append(norm)
    return norms
