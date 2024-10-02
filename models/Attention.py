import torch
import torch.nn as nn
import torch.nn.functional as F
import models._config as c
import torch.nn.utils.rnn as rnn_utils
from matplotlib import pyplot as plt
import visdom

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, nhead=8, dropout=0.3):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size*2, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        
        # attention_mask는 [batch_size, seq_len] 형태로 들어와야 함
        # src_key_padding_mask에 맞게 조정 [batch_size, seq_len]
        attention_mask = ~attention_mask.bool()  # 패딩 부분을 True로, 패딩이 아닌 부분을 False로 변경

        # Transformer 인코더에 전달
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        x = x.mean(dim=0)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def train_model(self, train_loader, criterion, optimizer, num_epochs, device, snr_str, ckpt):
        vis = visdom.Visdom()
        assert vis.check_connection(), "Visdom server not running, start with: python -m visdom.server"
        
        losses = []  
        vis_window = vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1,)).cpu(),
            opts=dict(xlabel='Epoch', ylabel='Loss', title=f'Training Loss for SNR {snr_str}', legend=['Loss'])
        )

        best_loss = float('inf')
        best_state = None
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            
            for data_batch, labels_batch, _, attention_mask in train_loader:
                data_batch = data_batch.to(device)
                labels_batch = labels_batch.to(device)
                attention_mask = attention_mask.to(device)

                optimizer.zero_grad() 
                outputs = self(data_batch, attention_mask)  # Transformer의 forward 함수에 attention_mask 추가
                
                loss = criterion(outputs, labels_batch)
                loss.backward()  
                optimizer.step() 

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            losses.append(avg_loss)
            
            vis.line(
                X=torch.tensor([epoch+1]).cpu(),
                Y=torch.tensor([avg_loss]).cpu(),
                win=vis_window,
                update='append'
            )

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
            
            # Best loss 갱신
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = self.state_dict()

            if best_loss < 0.01:
                break

        plt.figure()
        plt.plot(losses, label='Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training Loss for SNR {snr_str}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{ckpt}/loss_plot_{snr_str}.png")
        plt.close()
                    
        return best_state, best_loss

    @staticmethod
    def collate(batch):
        data, labels, snrs = zip(*batch)
        data_pad = rnn_utils.pad_sequence([torch.tensor(seq, dtype=torch.float32) for seq in data], batch_first=True)
        
        labels = torch.tensor([c.label_mapping[label] for label in labels], dtype=torch.long)
        snrs = torch.tensor(snrs, dtype=torch.float32)
        
        # attention_mask를 2차원으로 생성 (패딩이 아닌 부분: 1, 패딩인 부분: 0)
        attention_mask = (data_pad != 0).any(dim=-1).float()
        
        return data_pad, labels, snrs, attention_mask
