import torch

class LRP:
    def __init__(self, model, device, epsilon=1e-6):
        self.model = model
        self.epsilon = epsilon
        self.device = device
    
    def relevance(self, x, lengths, target=None):
        x = x.to(self.device)
        lengths = lengths.to(self.device)
        output, h, a = self.model(x, lengths, lstm_outputs=True)
        target = torch.argmax(output, dim=1) if target is None else target
        # Output shape: torch.Size([B, cls])
        # Hidden states shape: torch.Size([B, T, H*2])
        # Attention weights shape: torch.Size([B, T])
        # Context shape: torch.Size([B, H*2])
        
        r = torch.zeros_like(output)
        for i in range(output.size(0)):
            r[i, target[i]] = output[i, target[i]] # Target class에 대한 relevance만 생존
        
        r_c = self.bpp_fc(h, r)
        r_h = self.bpp_att(r_c, h, a)
        r_x = self.bpp_bilstm(r_h, h, x, lengths)
        
        # ground truth
        gt_label = torch.argmax(output, dim=1)
        return r_x, gt_label
    
    def lstm_gates(self, x_t, h_prev, W_ih, W_hh, b_ih, b_hh, cl_prev):
        x_t = x_t.to(self.device)
        h_prev = h_prev.to(self.device)
        cl_prev = cl_prev.to(self.device)
        
        gates = torch.matmul(W_ih, x_t) + torch.matmul(W_hh, h_prev) + b_ih + b_hh
        i_t, f_t, o_t, g_t = torch.chunk(gates, 4, dim=0)
        
        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        o_t = torch.sigmoid(o_t)
        cl_t = f_t * cl_prev + i_t * torch.tanh(g_t)
        
        return (f_t, i_t, o_t, cl_t)
        
    def bpp_bilstm(self, rel_h, h, x, lengths):
        B, T, H = h.size()
        H = H // 2
        D = x.size(-1)
        
        rel_x_fw = torch.zeros(B, T, D)                     # [B, T, D]
        rel_cl_fw_t1 = torch.zeros(B, H)                    # [B, H]
        W_ih = self.model.lstm.weight_ih_l0                             # [4*h, input_size]
        W_hh = self.model.lstm.weight_hh_l0                             # [4*h, hidden_size]
        b_ih = self.model.lstm.bias_ih_l0
        b_hh = self.model.lstm.bias_hh_l0
        
        rel_x_bw = torch.zeros(B, T, 2)
        rel_cl_bw_t1 = torch.zeros(B, H)
        W_ih_bw = self.model.lstm.weight_ih_l0_reverse
        W_hh_bw = self.model.lstm.weight_hh_l0_reverse
        b_ih_bw = self.model.lstm.bias_ih_l0_reverse
        b_hh_bw = self.model.lstm.bias_hh_l0_reverse
        
        # relevance는 cell state와 gate 값들을 통해 전파됨
        # rel_h_t : [B, T, sigma]
        # rel_x_fw[i, t] : [d] / rel_cl_fw_t1[i] : [h]
        for i in range(B):                                    # B
            h_prev_fw = torch.zeros(H)
            cl_prev_fw = torch.zeros(H)
            for t in reversed(range(T)):                      # T
                if t < lengths[i]:
                    x_t = x[i, t]                             # [H]
                    rel_h_fw_t = rel_h[i, t, :H]              # [H]
                    
                    f_t, i_t, o_t, cl_t = self.lstm_gates(x_t, h_prev_fw, W_ih, W_hh, b_ih, b_hh, cl_prev_fw) # [H]
                    rel_x_fw[i, t], rel_cl_fw_t1[i] = self.bpp_lstm_cell(rel_h_fw_t, rel_cl_fw_t1[i], 
                                                                            f_t, o_t, cl_t, W_ih)              
                    cl_prev_fw = cl_t
            
            h_prev_bw = torch.zeros(H)
            cl_prev_bw = torch.zeros(H)
            for t in range(T):
                if t < lengths[i]:
                    x_t = x[i, t]
                    rel_h_bw_t = rel_h[i, t, H:]
                    bf_t, bi_t, bo_t, bcl_t = self.lstm_gates(x_t, h_prev_bw, W_ih_bw, W_hh_bw, b_ih_bw, b_hh_bw, cl_prev_bw)
                    rel_x_bw[i, t], rel_cl_bw_t1[i] = self.bpp_lstm_cell(rel_h_bw_t, rel_cl_bw_t1[i], 
                                                                            bf_t, bo_t, bcl_t, W_ih_bw)
                    cl_prev_bw = bcl_t
                    
        rel_x = rel_x_fw + rel_x_bw
        return rel_x
        
    def bpp_lstm_cell(self, rel_h_t, rel_cl_t1, f_t, o_t, cl_t, W_ih):
        rel_cl_t = rel_cl_t1.to(self.device) + rel_h_t * o_t * (1 - torch.tanh(cl_t) ** 2)  # 전체 흐름 고려
    
        # 현재 시점에서 forget gate를 통해 이전 시점으로 relevance 전파
        rel_cl_t1 = rel_cl_t * f_t
        
        # Input gate를 통해 입력으로 relevance 전파 (i_t 반영)
        _, _, W_ih_o, W_ih_cl = torch.chunk(W_ih, 4, dim=0)  # input gate 가중치 추출
        
        rel_x_o = torch.matmul(W_ih_o.T, rel_cl_t * o_t)  # output gate 사용
        rel_x_cl = torch.matmul(W_ih_cl.T, rel_cl_t * torch.tanh(cl_t))  # cell state 사용
        
        # input gate를 통해 relevance 전파 (i_t 반영)
        rel_x_t = rel_x_o + rel_x_cl  # 최종 relevance 계산
        
        return rel_x_t, rel_cl_t1
        
    def bpp_att(self, r_c, h, w):
        rel_h = torch.zeros_like(h)
        for i in range(h.size(0)):
            rel_h_t = r_c[i].unsqueeze(0) * w[i].unsqueeze(-1)
            rel_h[i] = rel_h_t

        return rel_h
    
    def bpp_fc(self, c, r):
        fc_W = self.model.fc.weight
        rel_c = torch.zeros_like(c)
        for i in range(r.size(0)):
            for j in range(r.size(1)):
                rel_c[i] += (c[i]*fc_W[j]) * r[i, j] / (fc_W[j].abs().sum() + self.epsilon)
        return rel_c