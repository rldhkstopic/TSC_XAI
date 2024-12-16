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
        attn_weights = torch.tanh(self.attn(hidden_states)) 
        attn_weights = attn_weights.matmul(self.v)          
        attn_weights = F.softmax(attn_weights, dim=1)       
        
        context = torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1)  
        return context, attn_weights

# class SelfAttention(nn.Module):
#     def __init__(self, hidden_size):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Linear(hidden_size, hidden_size)
#         self.key = nn.Linear(hidden_size, hidden_size)
#         self.value = nn.Linear(hidden_size, hidden_size)
#         self.scale = hidden_size ** 0.5  

#     def forward(self, hidden_states, mask=None):
#         Q = self.query(hidden_states)  
#         K = self.key(hidden_states)   #
#         V = self.value(hidden_states) #

#         scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale 
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float('-inf')) 

#         attn_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]
#         context = torch.matmul(attn_weights, V)  # [batch_size, seq_len, hidden_size]

#         return context, attn_weights


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




class CrossAttention(nn.Module):
    def __init__(self, hidden_size, reduction=2, dropout=0.1):
        super(CrossAttention, self).__init__()
        
        # Low-rank Approximation
        self.hidden_apx = hidden_size // reduction  

        self.query = nn.Linear(hidden_size, self.hidden_apx)
        self.key = nn.Linear(hidden_size * 2, self.hidden_apx)
        self.value = nn.Linear(hidden_size * 2, self.hidden_apx)
        self.out = nn.Linear(self.hidden_apx, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask=None):
        query_proj = torch.tanh(self.query(queries))  
        key_proj = torch.tanh(self.key(keys))        
        value_proj = torch.tanh(self.value(values)) 

        attn_scores = torch.bmm(query_proj, key_proj.transpose(1, 2)) 

        if mask is not None:
            source_len = keys.size(1)  
            mask = mask[:, :source_len]
            mask = mask.unsqueeze(1).expand(-1, attn_scores.size(1), -1) 
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1) 

        context = torch.bmm(attn_weights, value_proj) 
        context = self.out(context)  

        return context, attn_weights
    

class BiLSTM_CA(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, factor=10, dropout=0.1):
        super(BiLSTM_CA, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.factor = factor 

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = CrossAttention(hidden_size, reduction=2, dropout=dropout)
        
        self.key_transform = nn.Linear(hidden_size, hidden_size * 2)  # Reduce from hidden_size * 2 to hidden_size
        self.value_transform = nn.Linear(hidden_size * 2, hidden_size)
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)  
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths, lstm_outputs=False):
        batch_size, seq_len, _ = x.size()

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)

        packed_x = rnn_utils.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        packed_out, _ = self.lstm(packed_x, (h0, c0))
        out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)  # [batch_size, seq_len, hidden_size * 2]

        mask = torch.arange(seq_len).expand(batch_size, seq_len).to(x.device) < lengths.unsqueeze(1).to(x.device)

        keys = out  
        values = out 

        reduced_query_len = max(1, seq_len // self.factor) 
        queries = torch.randn(batch_size, reduced_query_len, self.hidden_size, device=x.device)

        context, attn_weights = self.attention(queries, keys, values, mask)
        context = self.key_transform(context)

        out_last = self.dropout(context[:, -1, :]) 
        out_fc = self.fc(out_last) 

        if lstm_outputs:
            return out_fc, out, attn_weights, queries
        else:
            return out_fc







# class CrossAttention(nn.Module):
#     def __init__(self, hidden_size, dropout=0.1):
#         super(CrossAttention, self).__init__()
#         # low-rank attention
#         self.query = nn.Linear(hidden_size, hidden_size)
#         self.key = nn.Linear(hidden_size * 2, hidden_size)
#         self.value = nn.Linear(hidden_size * 2, hidden_size)
#         self.out = nn.Linear(hidden_size, hidden_size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, queries, keys, values, mask=None):
#         query_proj = self.query(queries)  # Q 변환
#         key_proj = self.key(keys)        # K 변환
#         value_proj = self.value(values)  # V 변환

#         attn_scores = torch.bmm(query_proj, key_proj.transpose(1, 2))  # Q * K^T
#         if mask is not None:
#             mask = mask.unsqueeze(1).expand(-1, attn_scores.size(1), -1)
#             attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
#         attn_weights = F.softmax(attn_scores, dim=-1)

#         context = torch.bmm(attn_weights, value_proj)  # Attention 적용
#         context = self.out(context)  # Context 변환
#         return context, attn_weights


# class BiLSTM_CA(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes, factor=2, dropout=0.1):
#         super(BiLSTM_CA, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.factor = factor
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
#         self.attention = CrossAttention(hidden_size, dropout=dropout)
#         self.fc = nn.Linear(hidden_size, num_classes)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, lengths, lstm_outputs=False):
#         batch_size, seq_len, _ = x.size()
#         h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
#         c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
#         packed_x = rnn_utils.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
#         self.lstm.flatten_parameters()
#         packed_out, _ = self.lstm(packed_x, (h0, c0))
#         out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)
        
#         mask = torch.arange(seq_len).expand(batch_size, seq_len).to(x.device) < lengths.unsqueeze(1).to(x.device)
#         mask = mask[:, :out.size(1)]
        
#         keys = out
#         values = out
#         query_len = max(1, seq_len // self.factor)
#         queries = torch.randn(batch_size, query_len, self.hidden_size, device=x.device)
#         context, attn_weights = self.attention(queries, keys, values, mask)
#         out_last = self.dropout(context[:, -1, :])
#         out_fc = self.fc(out_last)
#         if lstm_outputs:
#             return out_fc, out, attn_weights
#         else:
#             return out_fc




