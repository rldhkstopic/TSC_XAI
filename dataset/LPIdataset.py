import os
import numpy as np
import torch
import models._config as c
import torch.nn.utils.rnn as rnn_utils


class LPIDataset:
    def __init__(self, data_dir, waveforms, data_type='Signal'):
        self.data_dir = data_dir
        self.waveform = waveforms
        self.data_type = data_type
        self.file_list = self._collect()

    def _collect(self):
        files = []
        for waveform in self.waveform:
            waveform_folder = os.path.join(self.data_dir, waveform)
            files.extend([f for f in os.listdir(waveform_folder) if self.data_type in f])
        
        return files

    def _parse(self, file):
        parts = file.replace('.npy', '').split('_')
        label, snr, type, fps_idx = parts[0], int(parts[1].replace('snr', '')), parts[2], int(parts[3])
        
        return type, label, snr, fps_idx

    def _convIQ(self, complex_data):
        return complex_data.real, complex_data.imag
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        type, label, snr, fps_idx= self._parse(file)
        file_path = os.path.join(self.data_dir, label, file)
        
        complex_data = np.load(file_path)
        IQ_data = [self._convIQ(c) for c in complex_data]
        
        return IQ_data, label, len(complex_data), snr, type, fps_idx 
    
    @staticmethod
    def collate(batch):
        data, labels, lengths, _, _, _ = zip(*batch)
        data = [torch.tensor(seq, dtype=torch.float32) for seq in data]
        data_pad = rnn_utils.pad_sequence(data, batch_first=True)
        labels = torch.tensor([c.label_mapping[label] for label in labels], dtype=torch.long)
        lengths = torch.tensor(lengths, dtype=torch.int64)
        
        return data_pad, labels, lengths