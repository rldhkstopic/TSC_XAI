import torch
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

signalTypes = ['Barker', 'Costas', 'Frank', 'LFM', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4']
label_mapping = {signal: idx for idx, signal in enumerate(signalTypes)}

RawType = "/data/kiwan/dataset-CWD-50/"
TransformedTypes = {'DWT' : "/data/kiwan/Unknown_radar_detection/Adaptive_wavelet_transform/dataset-SPWVD-denoised-Adaptive_DWT",
                    'CWD' : "/data/kiwan/Unknown_radar_detection/Adaptive_wavelet_transform/240523_CWD-v1/",
                    'SAFI' : "/data/kiwan/Unknown_radar_detection/Adaptive_wavelet_transform/240523_SAFI-v1/",}

json_file = 'dataset/CWD_signal.json'

with open(json_file, 'r') as f:
    SignalData = json.load(f)
    
    
    