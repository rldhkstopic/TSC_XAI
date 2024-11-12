import torch
import torch.nn.functional as F
import numpy as np

import shap
from captum.attr import DeepLift


# Integrated Gradients
class IntegratedGradients:
    def __init__(self, model, baseline=None, steps=50):
        self.model = model
        self.baseline = baseline
        self.steps = steps

    def __call__(self, inputs, target_label_idx):
        if self.baseline is None:
            self.baseline = torch.zeros_like(inputs)

        scaled_inputs = [self.baseline + (float(i) / self.steps) * (inputs - self.baseline) for i in range(self.steps + 1)]
        grads = []
        for scaled_input in scaled_inputs:
            scaled_input.requires_grad = True
            output = self.model(scaled_input)
            output[0][target_label_idx].backward(retain_graph=True)
            grads.append(scaled_input.grad.detach().cpu().numpy())
        
        avg_grads = np.mean(grads, axis=0)
        integrated_grads = (inputs.cpu().numpy() - self.baseline.cpu().numpy()) * avg_grads
        return integrated_grads


# SHAP Explainer
class SHAPExplainer:
    def __init__(self, model, background_data):
        self.model = model
        self.background_data = background_data
        self.explainer = shap.DeepExplainer(model, background_data)

    def __call__(self, input_data):
        shap_values = self.explainer.shap_values(input_data)
        return shap_values


# DeepLIFT Explainer
class DeepLIFTExplainer:
    def __init__(self, model):
        self.model = model
        self.deep_lift = DeepLift(model)

    def __call__(self, inputs, target):
        attributions = self.deep_lift.attribute(inputs, target=target)
        return attributions


# Attention 가중치 시각화
class AttentionVisualizer:
    def __init__(self, model):
        self.model = model

    def __call__(self, hidden_states, attention_weights):
        """
        hidden_states: 모델의 출력 (LSTM 레이어의 시퀀스 출력)
        attention_weights: Self-Attention 레이어에서 추출된 Attention 가중치
        """
        # 각 시점의 attention 가중치를 곱해 컨텍스트 벡터 생성
        context_vector = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        return context_vector, attention_weights
    
class GradCAMp:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def __call__(self, x, index=None):
        activations = None
        for name, module in self.model.named_children():
            x = module(x)
            if name == self.target_layer:
                x.register_hook(self.save_gradient)
                activations = x

        if index is None:
            index = x.argmax(dim=1).item()
        output = F.softmax(x, dim=1)
        self.model.zero_grad()
        x[:, index].backward(retain_graph=True)

        grad_2 = self.gradients ** 2
        grad_3 = grad_2 * self.gradients
        global_sum = torch.sum(grad_3, dim=(2, 3), keepdim=True)

        alpha = grad_2 / (2 * grad_2 + global_sum + 1e-7)
        weights = torch.sum(alpha * torch.relu(self.gradients), dim=(2, 3))
        activations = activations.squeeze(0)
        weights = weights.squeeze(0)

        grad_cam_map = torch.sum(weights[:, None, None] * activations, dim=0)
        grad_cam_map = F.relu(grad_cam_map)
        grad_cam_map = grad_cam_map / torch.max(grad_cam_map)
        return grad_cam_map

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def __call__(self, x, index=None):
        x.requires_grad_()  # 입력 텐서에 requires_grad=True 설정
        activations = None
        for name, module in self.model.named_children():
            x = module(x)

            # x가 tuple일 경우 첫 번째 요소만 선택
            if isinstance(x, tuple):
                x = x[0]

            if name == self.target_layer:
                x.register_hook(self.save_gradient)
                activations = x

        if index is None:
            index = x.argmax(dim=1).item()
        output = F.softmax(x, dim=1)
        self.model.zero_grad()
        x[:, index].backward(retain_graph=True)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2])
        for i in range(activations.size(1)):
            activations[:, i, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        return heatmap
    