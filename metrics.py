
import torch.nn as nn
from models._config import *

from captum.attr import IntegratedGradients, Saliency, DeepLift

class BiLSTMWrapper(nn.Module):
    def __init__(self, model):
        super(BiLSTMWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        lengths = torch.tensor([x.size(1)] * x.size(0), dtype=torch.long, device=x.device)
        return self.model(x, lengths)
    
def feature_metric(R_x, wav, fs, alpha=0.8):
    _, _, Z_wav = stft_transform(wav.real, wav.imag, fs=fs)
    _, _, Z_Rx = stft_transform(R_x.real, R_x.imag, fs=fs)

    tau = alpha * np.max(np.abs(Z_wav))
    P_feature = np.abs(Z_wav) > tau
    relevance_in_feature = np.sum(np.abs(Z_Rx) * P_feature)
    total_relevance = np.sum(np.abs(Z_Rx))

    cpo = relevance_in_feature / total_relevance if total_relevance > 0 else 0

    return cpo

def rnr_metrics(R_x, wav, x, fs):  # Relevance_Noise_Robustness
    _, _, oZxx = stft_transform(wav.real, wav.imag, fs)
    _, _, xZxx = stft_transform(x.real, x.imag, fs)
    
    R_x_positive = np.where(R_x > 0, R_x, 0)
    _, _, rZxx_p = stft_transform(R_x_positive.real, R_x_positive.imag, fs)

    ov_noisy = np.sum(np.abs(rZxx_p) * np.abs(xZxx))
    ov_original = np.sqrt(np.sum(np.abs(rZxx_p)**2 * np.abs(oZxx)**2))
    
    score = ov_noisy / ov_original if ov_original > 0 else 0

    return score

def baselines(model, data, labels):
    model.train()
    ig = IntegratedGradients(BiLSTMWrapper(model))
    attr_ig, delta_ig = ig.attribute(inputs=data, target=labels.item(), return_convergence_delta=True)
    saliency = Saliency(BiLSTMWrapper(model))
    attr_saliency = saliency.attribute(data, target=labels.item())
    deeplift = DeepLift(BiLSTMWrapper(model))
    attr_deeplift = deeplift.attribute(data, target=labels.item())
    
    attr_ig = attr_ig.cpu().detach().numpy().squeeze().T
    attr_saliency = attr_saliency.cpu().detach().numpy().squeeze().T
    attr_deeplift = attr_deeplift.cpu().detach().numpy().squeeze().T
    
    return attr_ig, attr_saliency, attr_deeplift


def z_mask(Zxx, threshold=0.6):
    return np.where(np.abs(Zxx) > threshold * np.abs(Zxx).max(), 1, 0)
