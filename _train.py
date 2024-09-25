#%%
import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import _config as c
from models.LSTM import BiLSTM
from models.Attention import Transformer
from explainer import LRP


# python -m visdom.server
# ssh -L 8097:localhost:8097 kiwan@166.104.232.239

parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('-m', '--mode', type=str, default='train', help='Mode of operation (train/eval)')
parser.add_argument('-t', '--mtype', type=str, default='LSTM', help='Type of model to train (LSTM/Attention)')

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
parser.add_argument('--num_classes', type=int, default=len(c.signalTypes[:c.typeSize]), help='Number of output classes')

args = parser.parse_args()

params = {
    'mode': args.mode,
    'model_type': args.mtype,
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

def train_set(dataset, mtype='LSTM'):
    print(f"Size of train dataset: {len(dataset)}")
    for snr in range(params['snr_min'], params['snr_max']+1, 2):
        ckpt = os.path.join("./ckpts/", snr_str:=f"-{snr}dB" if snr != 0 else "0dB")
        print(f"\nTraining model for SNR: {snr_str}...")
        os.makedirs(ckpt, exist_ok=True)
        
        if mtype == 'LSTM':
            model = BiLSTM(params['input_size'], params['hidden_size'], 
                        params['num_layers'], params['num_classes']).to(c.device)
        elif mtype == 'Attention':
            model = Transformer(params['input_size'], params['hidden_size'], 
                    params['num_layers'], params['num_classes'], nhead=8, dropout=0.3).to(c.device)
           
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], 
                                                    weight_decay=params['weight_decay'])
        
        snr_dataset = [(data, label, data_snr) for data, label, data_snr in dataset if data_snr == snr]
        snr_loader = DataLoader(snr_dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=model.collate)
            
        best_state, best_loss = model.train_model(snr_loader, criterion, optimizer, params['num_epochs'], c.device, snr_str, ckpt)
        
        save_point = f'{ckpt}/{mtype}_{snr_str}_{best_loss:.4f}.pt'
        
        torch.save(best_state, save_point)
        print(f"Model checkpoint saved at {save_point}")

def eval_set(dataset, mtype='LSTM'):
    print(f"Size of test dataset: {len(dataset)}")
    for snr in range(params['snr_min'], params['snr_max']+1, 2):
        ckpt = os.path.join("./ckpts/", snr_str:=f"-{snr}dB" if snr != 0 else "0dB")
        ckpts = [f for f in os.listdir(ckpt) if f.endswith(".pt")]
        
        if mtype == 'LSTM':
            model = BiLSTM(params['input_size'], params['hidden_size'], 
                        params['num_layers'], params['num_classes']).to(c.device)
        elif mtype == 'Attention':
            model = Transformer(params['input_size'], params['hidden_size'], 
                    params['num_layers'], params['num_classes'], nhead=8, dropout=0.3).to(c.device)
            
        model.load_state_dict(torch.load(f"{ckpt}/{ckpts[0]}"))
        model.eval()
        
        snr_dataset = [(data, label, data_snr) for data, label, data_snr in dataset if data_snr == snr]
        snr_loader = DataLoader(snr_dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=model.collate)
        
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in snr_loader:
                data, labels = batch[0].to(c.device), batch[1].to(c.device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
        acc = correct / total * 100
        avg_loss = total_loss / len(snr_loader)
        
        print(f"SNR -{snr}dB | Accuracy: {acc:.2f}% | Average Loss: {avg_loss:.4f}")


def explain_set(dataset, mtype='LSTM'):
    """
    설명 및 기여도 분석을 위한 함수.
    주어진 데이터셋과 모델을 사용하여 각 데이터에 대한 LRP 기여도 계산 및 시각화.
    
    Parameters:
        - dataset: 입력 데이터셋
        - mtype: 모델 타입 ('LSTM', 'Attention' 등)
    """
    print(f"Size of explain dataset: {len(dataset)}")
    for snr in range(params['snr_min'], params['snr_max']+1, 2):
        print(f"\nExplaining model for SNR: {snr}...")
        ckpt = os.path.join("./ckpts/", snr_str := f"-{snr}dB" if snr != 0 else "0dB")
        ckpts = [f for f in os.listdir(ckpt) if f.endswith(".pt")]

        if mtype == 'LSTM':
            model = BiLSTM(params['input_size'], params['hidden_size'],
                        params['num_layers'], params['num_classes']).to(c.device)
        elif mtype == 'Attention':
            model = Transformer(params['input_size'], params['hidden_size'],
                    params['num_layers'], params['num_classes'], nhead=8, dropout=0.3).to(c.device)

        model.load_state_dict(torch.load(f"{ckpt}/{ckpts[0]}"))
        model.eval()
        
        lrp = LRP(model)

        snr_dataset = [(data, label, data_snr) for data, label, data_snr in dataset if data_snr == snr]
        snr_loader = DataLoader(snr_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=model.collate)

        results_dir = os.path.join("./explanations/", f"{mtype}_{snr_str}")
        os.makedirs(results_dir, exist_ok=True)

        explanation_results = []

        with torch.no_grad():
            for i, batch in enumerate(snr_loader):
                data, labels = batch[0].to(c.device), batch[1].to(c.device)

                relevance = lrp.get_relevance(data)
                
                for j in range(data.size(0)):
                    sample_data = data[j].cpu().numpy()
                    sample_relevance = relevance[j].cpu().numpy()
                    label = labels[j].item()
                    
                    explanation_results.append((sample_relevance, label))

                    # 히트맵으로 시각화
                    fig, ax = plt.subplots(figsize=(12, 6))
                    time_steps = np.arange(sample_data.shape[0])

                    # 원본 신호 플롯
                    ax.plot(time_steps, sample_data[:, 0], label='Signal', color='blue')
                    ax.set_title(f'Sample {i * params["batch_size"] + j} - SNR: {snr}dB, Label: {label}')
                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Amplitude')

                    # 히트맵 오버레이
                    im = ax.imshow(sample_relevance.T, aspect='auto', cmap='hot', alpha=0.6,
                                   extent=[0, sample_data.shape[0], 
                                           np.min(sample_data), np.max(sample_data)])

                    # 컬러바 추가
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Relevance Score')

                    # 시각화 저장
                    plot_path = os.path.join(results_dir, f"sample_{i * params['batch_size'] + j}_heatmap.png")
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"Heatmap for sample {i * params['batch_size'] + j} saved to {plot_path}")

        print(f"LRP explanations and visualizations for SNR {snr} saved in {results_dir}")

    return explanation_results
        

from RadarDataset import RadarSignalDataset  

if __name__ == "__main__":
    if args.mode == 'train':
        train_dataset = RadarSignalDataset(c.TrainData, c.signalTypes[0:c.typeSize], snr_max=17)
        train_set(train_dataset, mtype=params['model_type'])
    elif args.mode == 'eval':
        test_dataset = RadarSignalDataset(c.TestData, c.signalTypes[0:c.typeSize], snr_max=17)
        eval_set(test_dataset, mtype=params['model_type'])
    elif args.mode == 'explain':
        explain_dataset = RadarSignalDataset(c.TestData, c.signalTypes[0:c.typeSize], snr_max=17)
        explain_set(explain_dataset)
        
# %%


# %%
