
import torch.nn as nn
from models._config import *

from captum.attr import IntegratedGradients, Saliency, DeepLift

    
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


def critical_overlaps(wav, R_x, fs=100e6, alpha=0.8):
    f_w, t_w, Z_w = stft_transform(wav.real, wav.imag, fs=fs)
    f_R, t_R, Z_R = stft_transform(R_x.real, R_x.imag, fs=fs)

    tau = alpha * np.max(np.abs(Z_w))
    
    # 특징점 영역 (Critical Point Set)
    P_feature = np.abs(Z_w) > tau

    relevance_in_feature = np.sum(np.abs(Z_R) * P_feature)
    total_relevance = np.sum(np.abs(Z_R))

    cpo_score = relevance_in_feature / total_relevance if total_relevance > 0 else 0
    return cpo_score

def relevance_noise(x, R_x, fs):
    f_x, t_x, Z_x = stft_transform(x.real, x.imag, fs=fs)
    f_R, t_R, Z_R = stft_transform(R_x.real, R_x.imag, fs=fs)

    diff = np.linalg.norm(Z_R - Z_x, ord=2)
    norm = np.linalg.norm(Z_x, ord=2)

    rnr_score = 1 - (diff / norm if norm > 0 else 0)
    return rnr_score

def sra_metric(Z_gt, Z_input, Z_lrp, alpha=0.8, delta=0.8):
    # Create masks
    M_gt = (np.abs(Z_gt) > alpha * np.max(np.abs(Z_gt))).astype(float)  # Feature mask
    M_noise = 1 - M_gt  # Noise mask

    # Feature Concentration (FC)
    fc_num = np.sum(np.abs(M_gt) * np.abs(Z_lrp))
    fc_den = np.sum(np.abs(M_gt) * np.abs(Z_input))
    fc = fc_num / fc_den if fc_den > 0 else 0

    # Noise Reduction (NR)
    nr_num = np.sum(np.abs(M_noise) * np.abs(Z_lrp))
    nr_den = np.sum(np.abs(M_noise) * np.abs(Z_input))
    nr = 1 - (nr_num / nr_den if nr_den > 0 else 0)

    # SRA Score
    sra = delta * fc + (1 - delta) * nr
    return sra * 100  

class BiLSTMWrapper(nn.Module):
    def __init__(self, model):
        super(BiLSTMWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        lengths = torch.tensor([x.size(1)] * x.size(0), dtype=torch.long, device=x.device)
        return self.model(x, lengths)
    
def baselines(model, data, labels):
    device = next(model.parameters()).device
    data = data.to(device)
    labels = labels.to(device)
    
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
