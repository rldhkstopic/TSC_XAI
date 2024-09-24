import torch
import torch.nn as nn
import torch.nn.functional as F

class LRP:
    def __init__(self, model, epsilon=1e-5):
        self.model = model
        self.epsilon = epsilon
        self.model.eval()  # 모델을 평가 모드로 설정

    def forward(self, x):
        return self.model(x)
    
    def get_relevance(self, x, target=None):
        """
        입력 데이터 x와 관심 있는 타겟 클래스를 기반으로 LRP 값을 계산.
        - x: 입력 데이터 (torch.Tensor)
        - target: 관심 있는 클래스 인덱스 (default: None)

        모델의 예측 결과를 기반으로 타겟 클래스의 기여도를 계산하여 relevance를 반환.
        """
        # 순전파 수행
        x.requires_grad = True
        output = self.forward(x)
        
        # 관심 있는 클래스의 기여도를 계산 (target 클래스가 주어지지 않은 경우 전체 기여도를 사용)
        if target is None:
            target = torch.argmax(output, dim=1)  # 예측된 클래스 선택
        else:
            target = torch.tensor(target).to(x.device)
        
        # 모델의 예측 결과 중 관심 있는 클래스를 선택하여 그라디언트를 계산
        relevance = torch.zeros_like(output)
        for i in range(output.size(0)):
            relevance[i, target[i]] = output[i, target[i]]
        
        # 첫 번째 역전파를 통해 예측에 대한 기여도 계산
        relevance = self.compute_relevance(x, relevance)
        return relevance

    def compute_relevance(self, x, relevance):
        """
        모델의 각 층을 거쳐 relevance 역전파 수행.
        BiLSTM 모델의 각 층을 역순으로 순회하며 relevance를 계산.
        """
        # BiLSTM의 각 층에 대해 기여도를 계산하고 역전파
        for module in reversed(list(self.model.modules())):
            if isinstance(module, nn.Linear):
                relevance = self.linear_lrp(module, x, relevance)
            elif isinstance(module, nn.LSTM):
                relevance = self.bilstm_lrp(module, x, relevance)
            elif isinstance(module, nn.ReLU) or isinstance(module, nn.Tanh):
                relevance = self.activation_lrp(module, x, relevance)
        
        return relevance
    
    def linear_lrp(self, layer, x, relevance):
        """
        선형 층의 LRP 계산.
        - layer: nn.Linear 레이어
        - x: 입력 텐서
        - relevance: 기여도 텐서
        """
        weight = layer.weight
        bias = layer.bias
        
        z = F.linear(x, weight, bias) + self.epsilon  # 작은 epsilon 추가
        s = relevance / z
        c = torch.matmul(s, weight)
        return x * c
    
    def bilstm_lrp(self, layer, x, relevance):
        """
        BiLSTM 층의 LRP 계산.
        - layer: nn.LSTM 레이어 (BiLSTM 포함)
        - x: 입력 텐서
        - relevance: 기여도 텐서
        """
        # 양방향 LSTM인 경우 forward와 backward에 대해 기여도를 나누어 계산
        h, (hn, cn) = layer(x)
        
        # 양방향인 경우, forward와 backward 각각에 대해 relevance 계산
        # forward output: [:, :, :hidden_size]
        # backward output: [:, :, hidden_size:]
        hidden_size = layer.hidden_size
        relevance_fwd = relevance[:, :, :hidden_size]
        relevance_bwd = relevance[:, :, hidden_size:]
        
        # Forward LSTM에 대한 기여도 계산
        relevance_fwd = self.compute_lstm_relevance(layer, x, relevance_fwd, direction='forward')
        # Backward LSTM에 대한 기여도 계산
        relevance_bwd = self.compute_lstm_relevance(layer, x, relevance_bwd, direction='backward')
        
        # 전체 기여도 합산
        relevance = relevance_fwd + relevance_bwd
        return relevance
    
    def compute_lstm_relevance(self, layer, x, relevance, direction='forward'):
        """
        LSTM의 각 시간 스텝에 대해 relevance 계산.
        - layer: nn.LSTM 레이어
        - x: 입력 텐서
        - relevance: 기여도 텐서
        - direction: 'forward' or 'backward' (방향 선택)
        """
        # 정방향 또는 역방향 LSTM의 각 시간 스텝에 대해 relevance를 계산
        seq_len = x.size(1)
        
        # 방향에 따라 시퀀스를 정방향 또는 역방향으로 순회하며 기여도 계산
        if direction == 'forward':
            time_steps = range(seq_len)
        elif direction == 'backward':
            time_steps = reversed(range(seq_len))
        
        # 시퀀스를 순회하면서 각 시간 스텝에 대한 기여도 계산
        for t in time_steps:
            h_t = layer(x[:, t, :])[0]  # 현재 시간 스텝의 LSTM 출력
            z = h_t + self.epsilon  # 작은 epsilon 추가
            s = relevance[:, t, :] / z  # relevance 계산
            relevance[:, t, :] = x[:, t, :] * s
        
        return relevance
    
    def activation_lrp(self, layer, x, relevance):
        """
        활성화 함수의 LRP 계산 (ReLU, Tanh 등).
        활성화 함수에서는 LRP를 사용하여 기여도 역전파를 수행.
        """
        # 활성화 함수의 입력 x에 대해 기여도를 직접 역전파
        return relevance * (x > 0).float()  # 활성화된 뉴런만 기여도를 전달
