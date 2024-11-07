import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class LPIDataset:
    def __init__(self, data_dir, waveforms, data_type='Signal', nperseg=256, model_type='BiLSTM'):
        self.data_dir = data_dir
        self.waveform = waveforms
        self.data_type = data_type
        self.file_list = self._collect()
        
        self.nperseg = nperseg  
        self.model_type = model_type
        self.resize_shape = (128, 128) 
        self.resize_transform = transforms.Compose([
            transforms.Resize(self.resize_shape),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

    def _collect(self):
        files = []
        for waveform in self.waveform:
            waveform_folder = os.path.join(self.data_dir, waveform)
            files.extend([f for f in os.listdir(waveform_folder) if self.data_type in f or 'STFT' in f])
            
            
        return files

    def _parse(self, file):
        parts = os.path.splitext(file)[0].split('_') # 파일명에서 확장자 제거 후 '_'로 분리
        label = parts[0]
        snr = int(parts[1].replace('snr', ''))
        file_type = parts[2]
        fps_idx = int(parts[3])

        return file_type, label, snr, fps_idx

    def _convIQ(self, complex_data):
        return complex_data.real, complex_data.imag

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        file_type, label, snr, fps_idx = self._parse(file)
        
        if self.model_type == 'BiLSTM':
            # BiLSTM 모델의 경우 원본 시계열 데이터를 I/Q 분리하여 사용
            file_path = os.path.join(self.data_dir, label, file)
            complex_data = np.load(file_path)
            IQ_data = [self._convIQ(c) for c in complex_data]
            data_tensor = torch.tensor(IQ_data, dtype=torch.float32)
            length = len(data_tensor)  # 시퀀스 길이 반환
            return data_tensor, label, length, snr, fps_idx

        elif self.model_type in ['UNet', 'U2Net']:
            # UNet 및 U2Net 모델의 경우 STFT 이미지 파일 사용
            img_path = os.path.join(self.data_dir, label, file)

            image = Image.open(img_path)
            image = self.resize_transform(image)  # 리사이즈 및 텐서 변환
            
            
            return image, label, snr, fps_idx