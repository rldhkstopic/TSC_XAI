import torch
import json

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

signalTypes = ['Barker', 'Costas', 'Frank', 'LFM', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4']
label_mapping = {signal: idx for idx, signal in enumerate(signalTypes)}

RawType = "/data/kiwan/dataset-CWD-50/"
sameType = '/data/kiwan/LPI signals_train/'

TransformedTypes = {'DWT' : "/data/kiwan/Unknown_radar_detection/Adaptive_wavelet_transform/dataset-SPWVD-denoised-Adaptive_DWT",
                    'CWD' : "/data/kiwan/Unknown_radar_detection/Adaptive_wavelet_transform/240523_CWD-v1/",
                    'SAFI' : "/data/kiwan/Unknown_radar_detection/Adaptive_wavelet_transform/240523_SAFI-v1/",}

orig = False
if orig:
    with open('/data/kiwan/LPI signals/LPI signals_train.json', 'r') as f:
        TrainData = json.load(f)

    with open('/data/kiwan/LPI signals/LPI signals_train.json', 'r') as f:
        TestData = json.load(f)

else :
    with open('dataset/LPI12 dataset.json', 'r') as f:
        TrainData = json.load(f)

    with open('dataset/LPI12 dataset test.json', 'r') as f:
        TestData = json.load(f)

    with open('/data/kiwan/LPI signals/LPI signals_train.json', 'r') as f:
        TestData2 = json.load(f)
        
typeSize = 12

