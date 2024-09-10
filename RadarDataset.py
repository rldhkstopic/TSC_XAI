import os
import re
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class RadarDataset(Dataset):
    def __init__(self, directory, signal_types):
        self.directory = directory
        self.signal_types = signal_types
        self.data, self.labels, self.snrs, self.numb = ([] for _ in range(4))
        self.label_map = {signal: idx for idx, signal in enumerate(signal_types)}
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((100, 100)),
            transforms.ToTensor()
        ])
        self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label_map[self.labels[idx]], self.snrs[idx], self.numb[idx]
    
    def snr_sorted(self, filename):
        match = re.search(r'snr-(\d+)', filename)
        if match:
            return int(match.group(1))
        else:
            return None
        
    def _load_data(self):
        for signal in self.signal_types:
            signal_path = os.path.join(self.directory, signal)
            if os.path.exists(signal_path):
                png_files = sorted(glob.glob(os.path.join(signal_path, "*.png")), key=self.snr_sorted)
                for file_path in png_files:
                    image = Image.open(file_path).convert('RGB')
                    image = self.transform(image)
                    snr = self.snr_sorted(os.path.basename(file_path))
                    num = re.search(r'no(\d{5})', os.path.basename(file_path)).group(1)

                    self.data.append(image)
                    self.labels.append(signal)
                    self.snrs.append(snr)
                    self.numb.append(num)  
    