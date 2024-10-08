#%%
import os
import argparse
import numpy as np

import visdom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

import models._config as c
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
    

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        true_dist = torch.zeros_like(pred).scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist += self.smoothing / self.cls
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

def train_set(dataset):
    vis = visdom.Visdom()
    assert vis.check_connection(), "Visdom 서버 실행 : python -m visdom.server"
    vis_window = vis.line(X=torch.zeros((1,)).cpu(), Y=torch.zeros((1,)).cpu(), opts=dict(xlabel='Epoch', ylabel='Loss', title=f'Training Loss', legend=['Loss'])    )
    print(f"Size of train dataset: {len(dataset)}")

    losses = []  
    best_loss = float('inf')
    best_state = None
    

    print("Using", torch.cuda.device_count(), "GPUs with DataParallel.")
    model = BiLSTM(params['input_size'], params['hidden_size'], params['num_layers'], params['num_classes'])
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.to('cuda')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=model.module.collate, num_workers=16)
    
    epochs = params['num_epochs']
    for epoch in range(epochs):
        model.train()  
        rl = 0.0  
        
        progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        
        for batch_idx, (data_batch, labels_batch, _, lengths_batch) in progress_bar:
            optimizer.zero_grad()
            
            outputs = model(data_batch, lengths_batch)
            labels_batch = labels_batch.to(outputs.device)
            
            loss = criterion(outputs, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            rl += loss.item()

            progress_bar.set_postfix({'Batch': f"{batch_idx + 1}/{len(loader)}", 'Loss': f"{loss.item():.4f}"})

            loss.detach()

        avg_loss = rl / len(loader)
        losses.append(avg_loss)

        vis.line(X=torch.tensor([epoch + 1]).cpu(), Y=torch.tensor([avg_loss]).cpu(), win=vis_window, update='append')

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict()
            torch.save(best_state, f'./ckpts/result_loss.pt')

            if best_loss < 0.01:
                break
        
        torch.cuda.empty_cache()

    torch.save(best_state, f'./ckpts/result_loss_{best_loss:.4f}.pt')
    print("Train is done.")

def eval_set(dataset, mtype='LSTM'):
    print(f"Size of test dataset: {len(dataset)}")
    for snr in range(params['snr_min'], params['snr_max']+1, 2):
        ckpt = os.path.join("./ckpts/", snr_str:=f"SNR-{snr}dB" if snr != 0 else " 0dB")
        ckpts = [f for f in os.listdir(ckpt) if f.endswith(".pt") and f.startswith("BiLSTM")]
        
        model = BiLSTM(params['input_size'], params['hidden_size'], params['num_layers'], params['num_classes']).to(c.device)
        model.load_state_dict(torch.load(f"{ckpt}/{ckpts[0]}"))
        model.eval()
        criterion = nn.CrossEntropyLoss()
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

                    fig, ax = plt.subplots(figsize=(12, 6))
                    time_steps = np.arange(sample_data.shape[0])

                    ax.plot(time_steps, sample_data[:, 0], label='Signal', color='blue')
                    ax.set_title(f'Sample {i * params["batch_size"] + j} - SNR: {snr}dB, Label: {label}')
                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Amplitude')

                    im = ax.imshow(sample_relevance.T, aspect='auto', cmap='hot', alpha=0.6,
                                   extent=[0, sample_data.shape[0], 
                                           np.min(sample_data), np.max(sample_data)])

                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Relevance Score')

                    plot_path = os.path.join(results_dir, f"sample_{i * params['batch_size'] + j}_heatmap.png")
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"Heatmap for sample {i * params['batch_size'] + j} saved to {plot_path}")

        print(f"LRP explanations and visualizations for SNR {snr} saved in {results_dir}")

    return explanation_results
        

from dataset.RadarDataset import RadarSignalDataset  

from models._config import C
c = C()

if __name__ == "__main__":
    datajson = c.dataload(csv=True, mode=args.mode)
    dataset = RadarSignalDataset(datajson, c.signalTypes[0:c.typeSize], snr_max=17)
    
    if args.mode == 'train':
        train_set(dataset)
    elif args.mode == 'eval':
        eval_set(dataset, mtype=params['model_type'])
    elif args.mode == 'explain':
        explain_set(dataset)
        
# %%


# %%
