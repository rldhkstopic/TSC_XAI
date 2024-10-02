import visdom
import torch
import torch.nn as nn
import models._config as c
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from matplotlib import pyplot as plt

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
        self.attn = nn.Linear(hidden_size * 2, hidden_size)  # BiLSTM이므로 hidden_size * 2
        self.v = nn.Parameter(torch.rand(hidden_size)) 
        
    def forward(self, hidden_states, mask=None):
        """
        hidden_states: [batch_size, seq_len, hidden_size * 2]
        mask: [batch_size, seq_len] - zero-padding mask
        """
        attn_weights = torch.tanh(self.attn(hidden_states))  # [batch_size, seq_len, hidden_size]
        attn_weights = attn_weights.matmul(self.v)  # [batch_size, seq_len]

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)  # Zero-padding에 대한 large negative

        attn_weights = F.softmax(attn_weights, dim=1)  # [batch_size, seq_len]에서 softmax로 중요도 결정
        
        # 가중치를 반영하여 각 타임 스텝의 hidden state를 곱해줌
        context = torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_size * 2]
        return context, attn_weights

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = TemporalAttention(hidden_size)
        # self.multihead_attention = MultiHeadAttention(hidden_size * 2, num_heads)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Bidirectional이므로 hidden_size * 2
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, lengths, lstm_outputs=False):
        batch_size, seq_len, _ = x.size()

        # Initial hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)

        # PackedSequence로 변환하여 RNN/LSTM에서 패딩 무시
        packed_x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM output
        packed_out, (hn, cn) = self.lstm(packed_x, (h0, c0))  # LSTM 통과
        
        # 다시 패딩된 시퀀스로 변환
        out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)

        # Zero-padding Mask 생성
        mask = torch.arange(seq_len).expand(batch_size, seq_len).to(x.device) < lengths.to(x.device).unsqueeze(1)
        
        # Attention with Zero-padding Mask 적용
        context, attn_weights = self.attention(out, mask)  # Self-Attention 통과
        
        out_last = self.dropout(context)  # Dropout
        out_fc = self.fc(out_last)  # Fully connected layer

        if lstm_outputs:
            return out_fc, out, attn_weights
        else:
            return out_fc


    def train_model(self, train_loader, criterion, optimizer, num_epochs, device, snr_str, ckpt):
        vis = visdom.Visdom()
        assert vis.check_connection(), "Visdom 서버를 실행 필수 : python -m visdom.server"

        losses = []  
        vis_window = vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1,)).cpu(),
            opts=dict(xlabel='Epoch', ylabel='Loss', title=f'Training Loss for SNR {snr_str}', legend=['Loss'])
        )
        
        best_loss = float('inf')  # Best loss 초기화
        best_state = None

        for epoch in range(num_epochs):
            self.train()  
            running_loss = 0.0  
            
            for data_batch, labels_batch, _, lengths_batch in train_loader:  # lengths_batch 추가
                data_batch = data_batch.to(device)
                labels_batch = labels_batch.to(device)
                lengths_batch = lengths_batch.cpu()  # 시퀀스 길이를 CPU로 이동

                optimizer.zero_grad() 
                
                # forward에 lengths_batch도 함께 전달
                outputs = self(data_batch, lengths_batch)  # self()는 forward()를 호출함
                
                # Loss 계산
                loss = criterion(outputs, labels_batch)
                loss.backward()  
                optimizer.step() 

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            losses.append(avg_loss)

            # Visdom으로 Loss 그래프 업데이트
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

        return best_state, best_loss

    @staticmethod        
    def collate(batch):
        data, labels, snrs, lengths = zip(*batch)
        data_pad = rnn_utils.pad_sequence([torch.tensor(seq, dtype=torch.float32) for seq in data], batch_first=True)
        
        labels = torch.tensor([c.label_mapping[label] for label in labels], dtype=torch.long)
        snrs = torch.tensor(snrs, dtype=torch.float32)
        lengths = torch.tensor(lengths, dtype=torch.long)  # 시퀀스 길이를 함께 전달

        return data_pad, labels, snrs, lengths