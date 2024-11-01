import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

import models._config as c
from models.LSTM import BiLSTM
from dataset.LPIdataset import LPIDataset
from __params import getParams

def Train(batch_size=256, epochs=500):
    datatype = c.datatypes[-1]
    # torch.cuda.set_device(1)
    CEloss = nn.CrossEntropyLoss()
    model = BiLSTM(input_size=2, hidden_size=128, num_layers=2, num_classes=len(c.waveforms))
    model = nn.DataParallel(model, device_ids=[0, 1]).cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    dataset = LPIDataset(c.data_dir, c.waveforms, data_type=datatype)              
    dataload = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate)

    best_loss = float('inf')
    best_state = None

    losses = []
    epochs = 500
    for epoch in range(epochs):
        total_loss = 0.0
        progress = tqdm(enumerate(dataload), total=len(dataload), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            
        for batch_idx, (data, labels, lengths) in progress:
            optimizer.zero_grad()
            data, labels = data.cuda(), labels.cuda()
            outputs = model(data, lengths)
            loss = CEloss(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            progress.set_postfix({'Batch': f'{batch_idx+1}/{len(dataload)}', 'Loss': f'{loss.item():.4f}'})

            loss.detach()
        
        avg_loss = total_loss / len(dataload)
        losses.append(avg_loss)
        
        print(f'Epoch : {epoch+1}/{epochs}, Loss : {avg_loss:.4f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict()
            torch.save(best_state, f'./ckpts/{datatype}/_best_{best_loss}.pth')
            if best_loss < 0.0002:
                break
        torch.cuda.empty_cache()
    torch.save(model.state_dict(), f'./ckpts/{datatype}/_last_{best_loss}.pth')
