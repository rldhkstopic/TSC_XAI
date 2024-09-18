import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import _config as c
from RadarDataset import RadarSignalDataset
from model import BiLSTM, collate

typeSize = 4
dataset = RadarSignalDataset(c.SignalData, c.signalTypes[0:typeSize], snr_max=17, fft=False)

# python -m visdom.server
# ssh -L 8097:localhost:8097 kiwan@166.104.232.239
# Argument parser
parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('-m', '--mode', type=str, default='train', help='Mode of operation (train/eval)')

parser.add_argument('-smin','--snr_min', type=int, default=0, help='Minimum SNR value')
parser.add_argument('-smax','--snr_max', type=int, default=16, help='Maximum SNR value')
parser.add_argument('--split_size', type=float, default=0.8, help='Train/Test split size')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
parser.add_argument('--input_size', type=int, default=2, help='Input size for the model')
parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size for the model')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the model')
parser.add_argument('--num_classes', type=int, default=len(c.signalTypes[:typeSize]), help='Number of output classes')

args = parser.parse_args()

params = {
    'snr_min': args.snr_min,
    'snr_max': args.snr_max,
    'split_size': args.split_size,
    'batch_size': args.batch_size,
    'num_epochs': args.num_epochs,
    'learning_rate': args.learning_rate,
    'weight_decay': args.weight_decay,
    'input_size': args.input_size,
    'hidden_size': args.hidden_size,
    'num_layers': args.num_layers,
    'num_classes': args.num_classes
}
    
criterion = nn.CrossEntropyLoss()

train_size = int(params['split_size'] * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

def train_set():
    for snr in range(params['snr_min'], params['snr_max']+1, 2):
        print(f"\nTraining model for SNR: {snr}...")
        ckpt = os.path.join("./ckpts/", snr_str:=f"-{snr}dB" if snr != 0 else "0dB")
        os.makedirs(ckpt, exist_ok=True)
        
        LSTMmodel = BiLSTM(params['input_size'], params['hidden_size'], 
                    params['num_layers'], params['num_classes']).to(c.device)
        optimizer = optim.Adam(LSTMmodel.parameters(), lr=params['learning_rate'], 
                                                    weight_decay=params['weight_decay'])
        
        snr_dataset = [(data, label, data_snr) for data, label, data_snr in dataset if data_snr == snr]
        snr_loader = DataLoader(snr_dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=collate)
            
        best_state, best_loss = LSTMmodel.train_model(snr_loader, criterion, optimizer, params['num_epochs'], c.device, snr_str, ckpt)

        save_point = f'{ckpt}/LSTM_{snr_str}_{best_loss:.4f}.pt'
        
        torch.save(best_state, save_point)
        print(f"Model checkpoint saved at {save_point}")

def eval_set():
    for snr in range(0, params['snr_max'], 2):
        print(f"Evaluating model for SNR: -{snr}dB...")
        ckpt = os.path.join("./ckpts/", snr_str:=f"-{snr}dB" if snr != 0 else "0dB")
        ckpts = [f for f in os.listdir(ckpt) if f.endswith(".pt")]
        
        LSTMmodel = BiLSTM(params['input_size'], params['hidden_size'], 
                    params['num_layers'], params['num_classes']).to(c.device)
        LSTMmodel.load_state_dict(torch.load(f"{ckpt}/{ckpts[-1]}"))
        LSTMmodel.eval()
        
        snr_dataset = [(data, label, data_snr) for data, label, data_snr in dataset if data_snr == snr]
        snr_loader = DataLoader(snr_dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=collate)
        
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in snr_loader:
                data, labels = batch[0].to(c.device), batch[1].to(c.device)
                outputs = LSTMmodel(data)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
        acc = correct / total * 100
        avg_loss = total_loss / len(snr_loader)
        
        print(f"SNR -{snr}dB | Accuracy: {acc:.2f}% | Average Loss: {avg_loss:.4f}")
        
if __name__ == "__main__":
    if args.mode == 'train':
        train_set()
    elif args.mode == 'eval':
        eval_set()