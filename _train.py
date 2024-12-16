import torch
import torch.nn as nn
import torch.optim as optim
import models._config as c

from tqdm import tqdm
from torch.utils.data import DataLoader
from models.LSTM import BiLSTM
from models.CNN import UNet, U2Net
from dataset.LPIdataset import LPIDataset


def Train(model_type, data_dir, waveforms, datatype='pwnNoisy', batch_size=256, epochs=500, learning_rate=0.001, weight_decay=1e-5, device_ids=[0, 1]):
    CEloss = nn.CrossEntropyLoss()
    
    if model_type == 'BiLSTM':
        model = BiLSTM(input_size=2, hidden_size=128, num_layers=2, num_classes=len(waveforms))
    elif model_type == 'UNet':
        model = UNet(in_channels=1, out_channels=len(waveforms))
    elif model_type == 'U2Net':
        model = U2Net(in_channels=1, out_channels=len(waveforms))
    else:
        raise ValueError("Invalid model_type. Choose from 'BiLSTM', 'UNet', 'U2Net'.")
    
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    dataset = LPIDataset(data_dir, waveforms, data_type=datatype, model_type=model_type)
    dataload = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=c.collate)
    
    best_loss = float('inf')
    best_state = None
    losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        progress = tqdm(enumerate(dataload), total=len(dataload), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for batch_idx, (data, labels, lengths, snrs) in progress:
            if model_type == 'BiLSTM':
                # BiLSTM은 시계열 입력
                data, labels = data.cuda(), labels.cuda()
                outputs = model(data, lengths)
            else:
                # UNet, U2Net은 TFI 이미지 입력
                data, labels = data.cuda(), labels.cuda()
                outputs = model(data)



            loss = CEloss(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            progress.set_postfix({'Batch': f'{batch_idx+1}/{len(dataload)}', 'Loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataload)
        losses.append(avg_loss)
        
        print(f'Epoch : {epoch+1}/{epochs}, Loss : {avg_loss:.4f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict()
            torch.save(best_state, f'./ckpts/{datatype}/{model_type}_best_{best_loss:.4f}.pth')
            if best_loss < 0.002:
                break

        torch.cuda.empty_cache()
    
    torch.save(model.state_dict(), f'./ckpts/{datatype}/{model_type}_last_{best_loss:.4f}.pth')
    