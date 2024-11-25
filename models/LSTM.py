
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, reduction_ratio=2, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.hidden_size = hidden_size
        self.reduced_size = hidden_size // reduction_ratio  # Low-rank Approximation

        self.query = nn.Linear(hidden_size, self.reduced_size)
        self.key = nn.Linear(hidden_size * 2, self.reduced_size)
        self.value = nn.Linear(hidden_size * 2, self.reduced_size)
        self.out = nn.Linear(self.reduced_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask=None):
        """
        queries: [batch_size, target_len, hidden_size]
        keys: [batch_size, source_len, hidden_size * 2]
        values: [batch_size, source_len, hidden_size * 2]
        mask: [batch_size, source_len]
        """
        # Transform Query, Key, Value
        query_proj = self.query(queries)  # [batch_size, target_len, reduced_size]
        key_proj = self.key(keys)  # [batch_size, source_len, reduced_size]
        value_proj = self.value(values)  # [batch_size, source_len, reduced_size]

        # Compute attention scores
        attn_scores = torch.bmm(query_proj, key_proj.transpose(1, 2))  # [batch_size, target_len, source_len]

        # Expand mask to match attn_scores
        if mask is not None:
            # Ensure mask matches Key (source_len)
            source_len = keys.size(1)  # Dynamically fetch Key length
            mask = mask[:, :source_len]  # Truncate mask to match source_len
            mask = mask.unsqueeze(1).expand(-1, attn_scores.size(1), -1)  # [batch_size, target_len, source_len]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Normalize attention scores
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, target_len, source_len]

        # Compute context vectors
        context = torch.bmm(attn_weights, value_proj)  # [batch_size, target_len, reduced_size]
        context = self.out(context)  # Project back to hidden_size

        return context, attn_weights



class BiLSTM_CA(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, query_len, dropout=0.1):
        super(BiLSTM_CA, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.query_len = query_len

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = CrossAttention(hidden_size, reduction_ratio=2, dropout=dropout)
        self.query = nn.Parameter(torch.randn(1, query_len, hidden_size))  # Learnable Query
        self.key_transform = nn.Linear(hidden_size * 2, hidden_size)  # Reduce from hidden_size * 2 to hidden_size
        self.value_transform = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)  # Classification layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths, lstm_outputs=False):
        batch_size, seq_len, _ = x.size()

        # Initialize LSTM hidden states
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)

        # Pack and process LSTM output
        packed_x = rnn_utils.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        packed_out, _ = self.lstm(packed_x, (h0, c0))
        out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)  # [batch_size, seq_len, hidden_size * 2]

        # Create mask for padding
        mask = torch.arange(seq_len).expand(batch_size, seq_len).to(x.device) < lengths.unsqueeze(1).to(x.device)

        # Transform BiLSTM output to match Cross-Attention
        keys = out  # [batch_size, seq_len, hidden_size * 2]
        values = out  # [batch_size, seq_len, hidden_size * 2]
        queries = self.query.expand(batch_size, -1, -1)  # [batch_size, query_len, hidden_size]

        # Cross-Attention
        context, attn_weights = self.attention(queries, keys, values, mask)

        # Final classification
        out_last = self.dropout(context[:, -1, :])  # Use the last query result
        out_fc = self.fc(out_last)  # [batch_size, num_classes]

        if lstm_outputs:
            return out_fc, out, attn_weights
        else:
            return out_fc


