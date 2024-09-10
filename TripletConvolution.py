import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import os

class TCN(nn.Module): # Model2 (TCN)
    def __init__(self, input_channel=1):
        super(TCN, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv2d(input_channel, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1), # (64, 24, 24)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (64, 24, 24)
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1), # (32, 24, 24)
            nn.BatchNorm2d(32),                     
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (32, 12, 12)
            
            nn.Flatten(),  
            nn.Linear(32 * 12 * 12, 128) 
        )
        
    def forward(self, x):
        x = self.tcn(x)
        return x


def getTriplets(data, labels, num_triplets:int):
    triplets = []    
    for _ in range(num_triplets):
        anchor_idx = random.choice(range(len(data)))
        anchor_label = labels[anchor_idx]
        
        positive_indices = [i for i, label in enumerate(labels) if label == anchor_label and i != anchor_idx]
        negative_indices = [i for i, label in enumerate(labels) if label != anchor_label]
        
        if not positive_indices or not negative_indices:
            continue
        
        positive_idx = random.choice(positive_indices)
        negative_idx = random.choice(negative_indices)
        
        triplets.append((data[anchor_idx], data[positive_idx], data[negative_idx]))
        
    return triplets

def trainTCN(model, optim, dataset, snr, data_type, epochs=20, size=0.1, batch_size=128):
    os.makedirs(ckpt_dir := f'ckpts/{data_type}_SNR_-{snr}dB', exist_ok=True)
    print(save_point := f'Save Point: {ckpt_dir}/TCN_{data_type}_SNR_-{snr}dB.pth')
            
    train_data, train_labels = [d[0] for d in dataset], [d[1] for d in dataset]
    total_batches = len(train_data) // batch_size
    triplets = getTriplets(train_data, train_labels, int(len(train_data)*size)) # Triplet size = 10% of the dataset
    
    with open(f'{ckpt_dir}/tcn_results_SNR_-{snr}dB.txt', 'w') as f:
        start = time.time()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx in range(total_batches):
                batch_triplets = getTriplets(train_data, train_labels, batch_size)
                with tqdm(batch_triplets, desc=f'|Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{total_batches}]', dynamic_ncols=True) as progress_bar:
                    for a, p, n in progress_bar:
                        a, p, n = a.cuda(), p.cuda(), n.cuda()
                        optim.zero_grad()       
                        
                        f_a = model(a.unsqueeze(0))
                        f_p = model(p.unsqueeze(0))
                        f_n = model(n.unsqueeze(0))
                        
                        pos_dist = torch.norm(f_a - f_p, p=2)
                        neg_dist = torch.norm(f_a - f_n, p=2)
                        loss = torch.relu(pos_dist - neg_dist + 1.0).mean()
                        loss.backward()
                        
                        optim.step()
                        total_loss += loss.item()
                        progress_bar.set_postfix({'Loss': f'{total_loss/len(triplets):.4f}'})
        
        torch.save(model.state_dict(), save_point)
        print(f'Total Loss: {total_loss/len(triplets)} / time : {time.time()-start}', file=f)
    

