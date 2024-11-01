import os
import numpy as np
import random
from ._generator_utils import *

output_folder = '/data/kiwan/LPI_KIWAN/test'
os.makedirs(output_folder, exist_ok=True)
    
fs = 100e6
A = 1 
fps = 256
waveforms = ['LFM', 'Costas', 'Barker', 'Frank', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4']
for waveform in waveforms:
    os.makedirs(os.path.join(output_folder, waveform), exist_ok=True)

for waveform_idx, waveform in enumerate(waveforms):
    for snr_idx, snr in enumerate(snr_values := np.arange(-16, 2, 2)):

        print(f'Generating {waveform} waveform with SNR = {snr} dB....', end='\r')

        Nmin, Nmax = 1024, 1920
        
        if waveform == 'LFM':
            sweep = ['Up', 'Down']
            fc = np.linspace(fs/6, fs/5, fps)
            fc = fc[np.random.permutation(fps)]

            B = np.linspace(fs/20, fs/16, fps)
            B = B[np.random.permutation(fps)]

            N = np.linspace(Nmin, Nmax, fps)
            N = np.round(N[np.random.permutation(fps)])
            
            for fps_idx in range(fps):
                wav = type_lfm(N[fps_idx], fs, A, fc[fps_idx], B[fps_idx], random.choice(sweep))
                deployment(wav, snr, waveform, fps_idx)
                        
        elif waveform == 'Costas':
            fc = np.linspace(fs/30, fs/24, fps)
            fc = fc[np.random.permutation(fps)]
            N = np.linspace(Nmin, Nmax, fps)
            N = np.round(N[np.random.permutation(fps)]).astype(int) # [0] : 
            
            Lc = np.random.choice([3, 4, 5, 6])
            
            for fps_idx in range(fps):
                wav = type_costas(N[fps_idx], fs, A, fc[fps_idx], Lc)
                deployment(wav, snr, waveform, fps_idx)
            
        elif waveform == 'Barker':
            fc = np.linspace(fs/6, fs/5, fps)
            fc = fc[np.random.permutation(fps)]
            
            barker_codes = {
                7: np.array([1, 1, 1, -1, -1, 1, -1]),
                11: np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1]),
                13: np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])}
            
            Ncc = np.random.choice([20, 21, 22, 23, 24])
            Bar = np.random.choice([7, 11, 13])
            
            code = barker_codes[Bar]
            
            for fps_idx in range(fps):
                wav = type_barker(Ncc, fs, A, fc[fps_idx], code)
                deployment(wav, snr, waveform, fps_idx)
                
        elif waveform == 'Frank':
            fc = np.linspace(fs/6, fs/5, fps)
            fc = fc[np.random.permutation(fps)]
            
            cpp = np.random.choice([3, 4, 5])
            M = np.random.choice([6, 7, 8])            
            
            for fps_idx in range(fps):
                wav = type_frank(cpp, fs, A, fc[fps_idx], M)
                deployment(wav, snr, waveform, fps_idx)
        elif waveform == 'P1':
            fc = np.linspace(fs/6, fs/5, fps)
            fc = fc[np.random.permutation(fps)]
            
            cpp = np.random.choice([3, 4, 5])
            M = np.random.choice([6, 7, 8])
            
            for fps_idx in range(fps):            
                wav = type_P1(cpp, fs, A, fc[fps_idx], M)
                deployment(wav, snr, waveform, fps_idx)
        elif waveform == 'P2':
            fc = np.linspace(fs/6, fs/5, fps)
            fc = fc[np.random.permutation(fps)]
            
            cpp = np.random.choice([3, 4, 5])
            M = np.random.choice([6, 8])            
            
            for fps_idx in range(fps):
                wav = type_P2(cpp, fs, A, fc[fps_idx], M)
                deployment(wav, snr, waveform, fps_idx)
        elif waveform == 'P3':
            fc = np.linspace(fs/6, fs/5, fps)
            fc = fc[np.random.permutation(fps)]
            
            cpp = np.random.choice([3, 4, 5])
            p = np.random.choice([36, 49, 64])
            
            for fps_idx in range(fps):
                wav = type_P3(Ncc, fs, A, fc[fps_idx], p)
                deployment(wav, snr, waveform, fps_idx)

        elif waveform == 'P4':
            fc = np.linspace(fs/6, fs/5, fps)
            fc = fc[np.random.permutation(fps)]
            
            cpp = np.random.choice([3, 4, 5])
            p = np.random.choice([36, 49, 64])
            
            for fps_idx in range(fps):
                wav = type_P4(Ncc, fs, A, fc[fps_idx], p)
                deployment(wav, snr, waveform, fps_idx)
        elif waveform == 'T1':
            fc = np.linspace(fs/6, fs/5, fps)
            fc = fc[np.random.permutation(fps)]
            
            Nps = 2
            Ng = np.random.choice([4,5,6])
            
            for fps_idx in range(fps):
                wav = type_T1(fs, A, fc[fps_idx], Nps, Ng)
                deployment(wav, snr, waveform, fps_idx)
        elif waveform == 'T2':
            fc = np.linspace(fs/6, fs/5, fps)
            fc = fc[np.random.permutation(fps)]
            
            Nps = 2
            Ng = np.random.choice([4,5,6])
            
            for fps_idx in range(fps):
                wav = type_T2(fs, A, fc[fps_idx], Nps, Ng)
                deployment(wav, snr, waveform, fps_idx)
        elif waveform == 'T3':
            fc = np.linspace(fs/6, fs/5, fps)
            fc = fc[np.random.permutation(fps)]
            
            B = np.linspace(fs/20, fs/10, fps)
            B = B[np.random.permutation(fps)]
            
            Nps = 2
            Ng = np.random.choice([4,5,6])
            N = np.linspace(Nmin, Nmax, fps)
            N = np.round(N[np.random.permutation(fps)]).astype(int)
            
            for fps_idx in range(fps):
                wav = type_T3(N[fps_idx], fs, A, fc[fps_idx], Nps, B[fps_idx])
                deployment(wav, snr, waveform, fps_idx)
        elif waveform == 'T4':
            fc = np.linspace(fs/6, fs/5, fps)
            fc = fc[np.random.permutation(fps)]
            B = np.linspace(fs/20, fs/10, fps)
            B = B[np.random.permutation(fps)]
            
            Nps = 2
            Ng = np.random.choice([4,5,6])
            N = np.linspace(Nmin, Nmax, fps)
            N = np.round(N[np.random.permutation(fps)]).astype(int)
            
            for fps_idx in range(fps):
                wav = type_T4(N[fps_idx], fs, A, fc[fps_idx], Nps, B[fps_idx])
                deployment(wav, snr, waveform, fps_idx)
            
            
        
       

        

