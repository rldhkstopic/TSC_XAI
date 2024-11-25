import torch

class LRP:
    def __init__(self, model, device, epsilon=1e-6):
        self.model = model
        self.epsilon = epsilon
        self.device = device
    
    def relevance(self, x, lengths, target=None, attributions=None):
        x = x.to(self.device)
        lengths = lengths.cpu()
        output, h, a = self.model(x, lengths, lstm_outputs=True)
        target = torch.argmax(output, dim=1) if target is None else target
        
        r = torch.zeros_like(output)
        for i in range(output.size(0)):
            r[i, target[i]] = output[i, target[i]]  # Target class에 대한 relevance만 생존

        # FC layer에 대한 relevance 계산 (LRP-all rule)
        r_c = self.bpp_fc(h, r)
        
        # Attention layer에 대한 relevance 계산 (Ω-LRP rule)
        # r_h = self.bpp_att(r_c, h, a)
        
        # LSTM에 대한 relevance 계산 (lstm cell 내 각 게이트에 ε-LRP 및 Copy LRP 적용)
        r_x = self.bpp_bilstm(r_c, h, x, lengths)
        
        gt_label = torch.argmax(output, dim=1)
        
        if attributions is None:
            return r_x, gt_label
        else:
            return r_c, r_x, a, h, gt_label
    
    def lstm_gates(self, x_t, h_prev, W_ih, W_hh, b_ih, b_hh, cl_prev):
        x_t = x_t.to(self.device)
        h_prev = h_prev.to(self.device)
        cl_prev = cl_prev.to(self.device)
        
        gates = torch.matmul(W_ih, x_t) + torch.matmul(W_hh, h_prev) + b_ih + b_hh
        i_t, f_t, o_t, g_t = torch.chunk(gates, 4, dim=0)
        
        # Gate activations
        i_t = torch.sigmoid(i_t)  # Input gate
        f_t = torch.sigmoid(f_t)  # Forget gate
        o_t = torch.sigmoid(o_t)  # Output gate
        cl_t = f_t * cl_prev + i_t * torch.tanh(g_t)  # Cell state update
        
        return (f_t, i_t, o_t, cl_t)
        
    def bpp_bilstm(self, rel_h, h, x, lengths):
        B, T, H = h.size()
        H = H // 2
        D = x.size(-1)
        
        rel_x_fw = torch.zeros(B, T, D).to(self.device)
        rel_cl_fw_t1 = torch.zeros(B, H).to(self.device)
        W_ih = self.model.lstm.weight_ih_l0
        W_hh = self.model.lstm.weight_hh_l0
        b_ih = self.model.lstm.bias_ih_l0
        b_hh = self.model.lstm.bias_hh_l0
        
        rel_x_bw = torch.zeros(B, T, D).to(self.device)
        rel_cl_bw_t1 = torch.zeros(B, H).to(self.device)
        W_ih_bw = self.model.lstm.weight_ih_l0_reverse
        W_hh_bw = self.model.lstm.weight_hh_l0_reverse
        b_ih_bw = self.model.lstm.bias_ih_l0_reverse
        b_hh_bw = self.model.lstm.bias_hh_l0_reverse
        
        for i in range(B): 
            h_prev_fw = torch.zeros(H).to(self.device)
            cl_prev_fw = torch.zeros(H).to(self.device)
            for t in reversed(range(T)): 
                if t < lengths[i]:
                    x_t = x[i, t]
                    rel_h_fw_t = rel_h[i, t, :H]
                    
                    # LRP 규칙 적용
                    f_t, i_t, o_t, cl_t = self.lstm_gates(x_t, h_prev_fw, W_ih, W_hh, b_ih, b_hh, cl_prev_fw)
                    rel_x_fw[i, t], rel_cl_fw_t1[i] = self.bpp_lstm_cell(rel_h_fw_t, rel_cl_fw_t1[i], f_t, o_t, cl_t, W_ih)
                    cl_prev_fw = cl_t
            
            h_prev_bw = torch.zeros(H).to(self.device)
            cl_prev_bw = torch.zeros(H).to(self.device)
            for t in range(T):
                if t < lengths[i]:
                    x_t = x[i, t]
                    rel_h_bw_t = rel_h[i, t, H:]
                    bf_t, bi_t, bo_t, bcl_t = self.lstm_gates(x_t, h_prev_bw, W_ih_bw, W_hh_bw, b_ih_bw, b_hh_bw, cl_prev_bw)
                    rel_x_bw[i, t], rel_cl_bw_t1[i] = self.bpp_lstm_cell(rel_h_bw_t, rel_cl_bw_t1[i], bf_t, bo_t, bcl_t, W_ih_bw)
                    cl_prev_bw = bcl_t
                    
        rel_x = rel_x_fw + rel_x_bw
        return rel_x
        
    def bpp_lstm_cell(self, rel_h_t, rel_cl_t1, f_t, o_t, cl_t, W_ih):
        # ε-LRP rule 적용
        rel_cl_t = rel_cl_t1.to(self.device) + rel_h_t * o_t * (1 - torch.tanh(cl_t) ** 2 + self.epsilon)
    
        # Forget gate를 통해 relevance 전파 (Copy LRP rule 적용)
        rel_cl_t1 = rel_cl_t * f_t
        
        # Input gate의 relevance 전파 (Ω-LRP rule 적용)
        _, _, W_ih_o, W_ih_cl = torch.chunk(W_ih, 4, dim=0)
        
        rel_x_o = torch.matmul(W_ih_o.T, rel_cl_t * o_t)
        rel_x_cl = torch.matmul(W_ih_cl.T, rel_cl_t * torch.tanh(cl_t))
        
        # 최종 relevance 계산
        rel_x_t = rel_x_o + rel_x_cl
        return rel_x_t, rel_cl_t1
        
    def bpp_att(self, r_c, h, w):
        # Attention 단계에서 Ω-LRP rule 적용
        rel_h = torch.zeros_like(h).to(self.device)
        for i in range(h.size(0)):
            rel_h[i] = r_c[i].unsqueeze(0) * w[i].unsqueeze(-1)
        return rel_h
    
    def bpp_fc(self, c, r):
        # Fully Connected Layer에서 LRP-all rule 적용
        fc_W = self.model.fc.weight
        rel_c = torch.zeros_like(c).to(self.device)
        for i in range(r.size(0)):
            for j in range(r.size(1)):
                rel_c[i] += (c[i] * fc_W[j]) * r[i, j] / (fc_W[j].abs().sum() + self.epsilon)
        return rel_c
