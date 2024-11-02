import os
import numpy as np
import torch
from torchvision import transforms
from scipy.signal import stft

class LPIDataset:
    def __init__(self, data_dir, waveforms, data_type='Signal', nperseg=256, model_type='BiLSTM'):
        self.data_dir = data_dir
        self.waveform = waveforms
        self.data_type = data_type
        self.file_list = self._collect()
        self.nperseg = nperseg  # STFT 윈도우 크기
        self.model_type = model_type
        self.resize_shape = (128, 128)  
        self.resize_transform = transforms.Resize(self.resize_shape)

    def _collect(self):
        """ 데이터 파일 목록을 수집 """
        files = []
        for waveform in self.waveform:
            waveform_folder = os.path.join(self.data_dir, waveform)
            files.extend([f for f in os.listdir(waveform_folder) if self.data_type in f])
        return files

    def _parse(self, file):
        """ 파일명에서 메타데이터 파싱 """
        parts = file.replace('.npy', '').split('_')
        label, snr, type, fps_idx = parts[0], int(parts[1].replace('snr', '')), parts[2], int(parts[3])
        return type, label, snr, fps_idx

    def _convIQ(self, complex_data):
        """ I/Q 데이터를 실수부와 허수부로 분리 """
        return complex_data.real, complex_data.imag

    def _stft_transform(self, data):
        f, t, Zxx_real = stft(data.real, nperseg=self.nperseg)
        _, _, Zxx_imag = stft(data.imag, nperseg=self.nperseg)
        
        TFI_image = np.stack([np.abs(Zxx_real), np.abs(Zxx_imag)], axis=0)  # (2, freq, time)
        
        TFI_tensor = torch.tensor(TFI_image, dtype=torch.float32)
        TFI_tensor = self.resize_transform(TFI_tensor)
        
        return TFI_tensor

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        type, label, snr, fps_idx = self._parse(file)
        file_path = os.path.join(self.data_dir, label, file)

        complex_data = np.load(file_path)

        if self.model_type == 'BiLSTM':
            # BiLSTM 모델의 경우 원본 시계열 데이터를 I/Q 분리하여 사용
            IQ_data = [self._convIQ(c) for c in complex_data]
            data_tensor = torch.tensor(IQ_data, dtype=torch.float32)
            length = len(data_tensor)  # 시퀀스 길이 반환
            return data_tensor, label, length, snr

        else:
            # UNet 및 U2Net 모델의 경우 STFT 변환된 TFI 이미지 사용
            TFI_image = self._stft_transform(complex_data)
            data_tensor = torch.tensor(TFI_image, dtype=torch.float32)
            return data_tensor, label, snr


