import torch
import torch.nn.functional as F

class LRP:
    def __init__(self, model, epsilon=1e-5):
        self.model = model
        self.epsilon = epsilon
        self.model.eval()

    def get_relevance(self, x, target=None):
        """
        입력 데이터 x에 대한 relevance 점수를 계산합니다.
        Args:
            x (torch.Tensor): 입력 데이터 (batch_size, seq_len, input_size)
            target (torch.Tensor): 타겟 클래스 (optional)
        Returns:
            relevance (torch.Tensor): relevance 점수 (batch_size, seq_len)
            attention_weights (torch.Tensor): attention 가중치 (batch_size, seq_len)
        """
        # 순전파 수행
        x.requires_grad = True
        output, hid_outputs, attn_weights = self.forward(x, lstm_outputs=True)  # LSTM outputs: [batch_size, seq_len, hidden_size * 2]

        # 예측된 클래스 선택
        if target is None:
            target = torch.argmax(output, dim=1)
        else:
            target = torch.tensor(target).to(x.device)

        # 모델 예측에 대한 기여도 초기화
        relevance = torch.zeros_like(output)
        for i in range(output.size(0)):
            relevance[i, target[i]] = output[i, target[i]]

        # LRP relevance 역전파 계산
        relevance = self.compute_relevance(hid_outputs, relevance)

        return relevance, output, attn_weights

    def forward(self, x, lstm_outputs=False):
        """모델의 순전파를 수행하고 LSTM hidden outputs 및 attention weights를 반환합니다."""
        return self.model(x, lstm_outputs)

    def compute_relevance(self, hiddens, relevance):
        """
        LSTM 모델에서 relevance 역전파 수행.
        Args:
            hiddens (torch.Tensor): LSTM hidden states (batch_size, seq_len, hidden_size * 2)
            relevance (torch.Tensor): 모델의 출력 레벨에서 계산된 relevance
        Returns:
            total_relevance (torch.Tensor): 각 타임 스텝에 대한 relevance (batch_size, seq_len, hidden_size * 2)
        """
        batch_size, seq_len, hidden_size2 = hiddens.size()  # hidden_size2 = hidden_size * 2
        hidden_size = hidden_size2 // 2
        
        # Forward LSTM에 대한 relevance 계산
        h_fw, h_bw = hiddens[:, :, :hidden_size], hiddens[:, :, hidden_size:]
        rel_fw, rel_bw = torch.zeros_like(h_fw), torch.zeros_like(h_bw)
        
        for t in reversed(range(seq_len)):
            h_t_fw = h_fw[:, t, :]  # [batch_size, hidden_size]
            h_t_bw = h_bw[:, t, :]  # [batch_size, hidden_size]
            
            # Forward 방향 relevance 계산
            z_t_fw = F.linear(h_t_fw, self.model.fc.weight[:, :hidden_size], self.model.fc.bias) + self.epsilon  # [batch_size, 4]
            s_t_fw = relevance / (z_t_fw + self.epsilon)  # [batch_size, 4]
            c_t_fw = torch.matmul(s_t_fw, self.model.fc.weight[:, :hidden_size])  # [batch_size, hidden_size]
            rel_fw[:, t, :] = h_t_fw * c_t_fw  # [batch_size, hidden_size]

            # Backward 방향 relevance 계산
            z_t_bw = F.linear(h_t_bw, self.model.fc.weight[:, hidden_size:], self.model.fc.bias) + self.epsilon  # [batch_size, 4]
            s_t_bw = relevance / (z_t_bw + self.epsilon)  # [batch_size, 4]
            c_t_bw = torch.matmul(s_t_bw, self.model.fc.weight[:, hidden_size:])  # [batch_size, hidden_size]
            rel_bw[:, t, :] = h_t_bw * c_t_bw  # [batch_size, hidden_size]
        
        # Forward와 Backward relevance 합산
        total_relevance = rel_fw + rel_bw  # [batch_size, seq_len, hidden_size * 2]
        
        return total_relevance

