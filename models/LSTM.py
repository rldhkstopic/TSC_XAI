import visdom
import torch
import torch.nn as nn
import models._config as c
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from matplotlib import pyplot as plt
from tqdm import tqdm

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size)) 
        
    def forward(self, hidden_states):
        attn_weights = torch.tanh(self.attn(hidden_states))                           # [batch_size, seq_len, hidden_size]
        attn_weights = attn_weights.matmul(self.v)
        attn_weights = torch.softmax(attn_weights, dim=1)                             # [batch_size, seq_len]
        
        context = torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1) # [batch_size, hidden_size * 2]
        return context, attn_weights
    
# Temporal Attention Layer (Zero-padding에 대한 가중치 조정 포함)
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)  # 
        self.v = nn.Parameter(torch.rand(hidden_size)) # 
        
    def forward(self, hidden_states, mask=None):
        """
        hidden_states: [batch_size, seq_len, hidden_size * 2]
        mask: [batch_size, seq_len] - zero-padding mask
        """
        attn_weights = torch.tanh(self.attn(hidden_states))  # [batch_size, seq_len, hidden_size]
        attn_weights = attn_weights.matmul(self.v)           # [batch_size, seq_len]
        attn_weights = F.softmax(attn_weights, dim=1)  # [batch_size, seq_len]에서 softmax로 중요도 결정
        
        # 가중치를 반영하여 각 타임 스텝의 hidden state를 곱해줌
        context = torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_size * 2]
        return context, attn_weights

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        
        self.attention = TemporalAttention(hidden_size)
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)  
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, lengths, lstm_outputs=False):
        device = next(self.parameters()).device  # 모델의 디바이스를 확인
        if x.device != device:
            x = x.to(device)
        batch_size, seq_len, _ = x.size()

        # Initial hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)

        # PackedSequence로 변환하여 RNN/LSTM에서 패딩 무시
        packed_x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False).to(x.device)
        
        
        packed_out, (hn, cn) = self.lstm(packed_x, (h0, c0))  # LSTM 통과
        
        # 다시 패딩된 시퀀스로 변환
        out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)

        # Zero-padding Mask 생성
        mask = torch.arange(seq_len).expand(batch_size, seq_len) < lengths.unsqueeze(1)
        
        # Attention with Zero-padding Mask 적용
        context, attn_weights = self.attention(out, mask)  # Self-Attention 통과
        
        out_last = self.dropout(context)  # Dropout
        out_fc = self.fc(out_last)  # Fully connected layer
                
        if lstm_outputs:
            return out_fc, out, attn_weights
        else:
            return out_fc


    def train_model(self, train_loader, criterion, optimizer, num_epochs):
        vis = visdom.Visdom()
        assert vis.check_connection(), "Visdom 서버를 실행 필수 : python -m visdom.server"

        losses = []  
        vis_window = vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1,)).cpu(),
            opts=dict(xlabel='Epoch', ylabel='Loss', title=f'Training Loss', legend=['Loss'])
        )
        
        best_loss = float('inf')  # Best loss 초기화
        best_state = None

        for epoch in range(num_epochs):
            self.train()  
            running_loss = 0.0  
            
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
            
            for batch_idx, (data_batch, labels_batch, _, lengths_batch) in progress_bar:
                
                optimizer.zero_grad()
                

                outputs = self(data_batch, lengths_batch)  # self()는 forward()를 호출함
                labels_batch = labels_batch.to(outputs.device)
                
                # Loss 계산
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # 현재 배치 번호와 평균 손실을 tqdm에 표시
                progress_bar.set_postfix({
                    'Batch': f"{batch_idx + 1}/{len(train_loader)}",
                    'Loss': f"{loss.item():.4f}"
                })

                loss.detach()

            avg_loss = running_loss / len(train_loader)
            losses.append(avg_loss)

            vis.line(
                X=torch.tensor([epoch + 1]).cpu(),
                Y=torch.tensor([avg_loss]).cpu(),
                win=vis_window,
                update='append'
            )

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
            
            # Best loss 갱신
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = self.state_dict()

            if best_loss < 0.01:
                break
            
            torch.cuda.empty_cache()

        return best_state, best_loss

    @staticmethod        
    def collate(batch):
        data, labels, snrs, lengths = zip(*batch)
        data_pad = rnn_utils.pad_sequence([torch.tensor(seq, dtype=torch.float32) for seq in data], batch_first=True)
        
        labels = torch.tensor([c.label_mapping[label] for label in labels], dtype=torch.long)
        snrs = torch.tensor(snrs, dtype=torch.float32)
        lengths = torch.tensor(lengths, dtype=torch.float32)  # 시퀀스 길이를 함께 전달

        return data_pad, labels, snrs, lengths
    

