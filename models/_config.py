import torch
import numpy as np
from scipy.signal import stft
import torch.nn.utils.rnn as rnn_utils



data_dir = '/data/kiwan/LPI_KIWAN/'
snr_values = [0, 2, 4, 6, 8, 10, 12, 14, 16]
datatypes = ['Signal', 'Noise', 'Noisy', 'pwnNoisy']
waveforms = ['Barker', 'Costas', 'Frank', 'LFM', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4']
label_mapping = {signal: idx for idx, signal in enumerate(waveforms)}
        
typeSize = 12
fs = 100e6

def collate(batch):
    if isinstance(batch[0][3], int):  
        data, labels, lengths, _ = zip(*batch)
        data = [torch.tensor(seq, dtype=torch.float32) for seq in data]
        data_pad = rnn_utils.pad_sequence(data, batch_first=True)

        label_to_index = {label: idx for idx, label in enumerate(waveforms)}
        labels = torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)
        lengths = torch.tensor(lengths, dtype=torch.int64)
        return data_pad, labels, lengths

    else:  # UNet 또는 U2Net 모델
        data, labels, _ = zip(*batch)
        data = torch.stack(data, dim=0)
        label_to_index = {label: idx for idx, label in enumerate(waveforms)}
        labels = torch.tensor([int(label) for label in labels], dtype=torch.long)
        return data, labels





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

def fft_transform(data_real, data_imag):
    fs = 100e6
    complex_signal = data_real + 1j * data_imag
    fft_data = np.fft.fft(complex_signal)
    fft_amp = np.abs(fft_data)                
    fft_freq = np.fft.fftfreq(len(fft_data), 1/fs)           
    return fft_freq[:len(fft_freq)//2] ,fft_amp[:len(fft_amp) // 2] 

def stft_transform(data_real, data_imag, fs=100e6, nperseg=256):
    complex_signal = data_real + 1j * data_imag
    f, t, Zxx = stft(complex_signal, fs=fs, nperseg=nperseg)
    stft_amp = np.abs(Zxx)
    return f, t, stft_amp
    
    
def get_entropy(rel):
    rel_flat = rel.flatten()
    rel_flat = rel_flat[rel_flat > 0]
    entropys = -np.sum(rel_flat * np.log(rel_flat+1e-6))
    return entropys
    

def test_dataset(dataset, test_type, snrs, target_label, target_snr, batch_size):
    snr_batch = []
    snr_value = snr_values[:snrs]
    
    for data, label, snr, length in dataset:
        
        if test_type == 0: # 동일 snr, 동일 label
            if snr == target_snr and label == target_label:
                snr_batch.append((data, label, snr, length))
                if len(snr_batch) == batch_size:
                    break
                    
        elif test_type == 1: # Label : Same, SNR : Same
            if label == target_label:
                if len([sb for _, _, sb, _ in snr_batch if sb == snr]) >= batch_size:
                    continue
                snr_batch.append((data, label, snr, length))
                
        elif test_type == 2: # Label : Diff, SNR : Same
            if snr == target_snr and length >= 1000 : 
                if all(existing_label != label for _, existing_label, _, _ in snr_batch):
                    snr_batch.append((data, label, snr, length))
                    if len(snr_batch) == batch_size:
                        break
        elif test_type == 3: # Label : Same, SNR : Diff
            if label == target_label and length >= 1000:
                if all(existing_snr != snr for _, _, existing_snr, _ in snr_batch):
                    snr_batch.append((data, label, snr, length))
                    if len(snr_batch) == batch_size:
                        break
                        
        else:
            print("Invalid test type")
    
    print(len(snr_batch))
    return snr_batch


def top_k(rel, k=10):
    return np.where(rel >= np.percentile(rel, 100-k), rel, 1e-6)

def under_k(rel, k=10):
    return np.where(rel <= np.percentile(rel, k), rel, 0)

def dynamic_k(rel, k=10): # 
    return np.mean(rel) + k * np.std(rel) 

def time_domain(data, rel_x, length, attn=None):
    Ts = length
    

    x_real, x_imag = data[:Ts, :].cpu().numpy().T
    R_x_real, R_x_imag = rel_x[:Ts, :].cpu().numpy().T

    x = x_real + x_imag
    R_x = R_x_real + R_x_imag

    if attn is None:
        return x, R_x, x_real, x_imag, R_x_real, R_x_imag
    else:
        W_att = attn[:Ts].cpu().numpy()
        return x, R_x, W_att, x_real, x_imag, R_x_real, R_x_imag

def fft_domain(x_real, x_imag, R_x_real, R_x_imag):
    x_fft = list(fft_transform(x_real, x_imag))
    R_fft = list(fft_transform(R_x_real, R_x_imag))
    R_fft[1] = R_fft[1][:len(x_fft[0])]
    
    return tuple(x_fft), tuple(R_fft)

def wRelevance(data, Relevance, weight=2):
    W = (data * Relevance**weight)
        
    return W