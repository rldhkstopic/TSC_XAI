import torch
from torch.utils.data import Dataset

class RadarSignalDataset(Dataset):
    def __init__(self, signals_data, signal_types, snr_max=17, fft=False):
        self.data, self.labels, self.snrs, self.lengths = ([] for _ in range(4))
        self.fft = fft
        self.device = 'cuda'
        
        for signal_type in signal_types:
            print(f"Data loading for '{signal_type}'", end='')
            for snr_idx, snr in enumerate(range(0, snr_max, 2)): 
                print(".", end='') if snr_idx % 2 == 0 else None
                ssnr = str(snr)
                if ssnr in signals_data[signal_type]: 
                    signal_snr_data = signals_data[signal_type][ssnr]
                    for signal in signal_snr_data:
                        complex_signal = [self.convIQ(x) for x in signal]
                        if self.fft:
                            complex_signal = self.applyFFT(complex_signal)
                        self.data.append(complex_signal)
                        self.labels.append(signal_type)
                        self.snrs.append(snr)
                        self.lengths.append(len(complex_signal))  # 시퀀스 길이 저장
            print("Done!")
    
    def convIQ(self, datastring):
        comp = complex(datastring.replace('i', 'j'))
        return comp.real, comp.imag    
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.snrs[idx], self.lengths[idx]  
