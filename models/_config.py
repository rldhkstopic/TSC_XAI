import torch
import json

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(device)

signalTypes = ['Barker', 'Costas', 'Frank', 'LFM', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4']
label_mapping = {signal: idx for idx, signal in enumerate(signalTypes)}
        
typeSize = 12

class C:
    def __init__(self, typeSize=12):
        self.typeSize = typeSize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.signalTypes = ['Barker', 'Costas', 'Frank', 'LFM', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4']
        self.label_mapping = {signal: idx for idx, signal in enumerate(self.signalTypes)}
        
    def dataload(self, mode, csv=True):
        if mode == 'train' or mode == 'explain':
            print(f"<<Loading Train Data [{csv}]>>")
            ckpt = './dataset/LPI12_CSV.json' if csv == True else './dataset/LPI12_NMP.json'
        elif mode == 'test':
            print(f"<<Loading Test Data [{csv}]>>")
            ckpt = './dataset/LPI12_CSV_test.json' if csv == True else './dataset/LPI12_NMP_test.json'
            
        with open(ckpt, 'r') as f:
            self.dataset = json.load(f)
        return self.dataset
    
    def signals(self):
        return self.signalTypes, self.label_mapping