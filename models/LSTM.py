
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
        self.attn = nn.Linear(hidden_size * 2, hidden_size)  # 
        self.v = nn.Parameter(torch.rand(hidden_size)) # 
        
    def forward(self, hidden_states, mask=None):
        """
        hidden_states: [batch_size, seq_len, hidden_size * 2]
        mask: [batch_size, seq_len] - zero-padding mask
        """
        attn_weights = torch.tanh(self.attn(hidden_states))  # [batch_size, seq_len, hidden_size]
        attn_weights = attn_weights.matmul(self.v)           # [batch_size, seq_len]
        attn_weights = F.softmax(attn_weights, dim=1)        # [batch_size, seq_len]에서 softmax로 중요도 결정
        
        context = torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_size * 2]
        return context, attn_weights

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = SelfAttention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, lengths, lstm_outputs=False):
        batch_size, seq_len, _ = x.size()

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        
        packed_x = rnn_utils.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        packed_out, (hn, cn) = self.lstm(packed_x, (h0, c0))

        out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)
        mask = torch.arange(seq_len).expand(batch_size, seq_len).to(x.device) < lengths.unsqueeze(1).to(x.device)
        
        context, attn_weights = self.attention(out, mask) 
        
        out_last = self.dropout(context) 
        out_fc = self.fc(out_last) 
                
        if lstm_outputs:
            return out_fc, out, attn_weights
        else:
            return out_fc
        