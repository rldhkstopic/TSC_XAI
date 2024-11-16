import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from models.LSTM import BiLSTM
from models.UNET import UNet, U2Net
from models._config import collate_fn as custom_collate_fn  # Import collate function if available
from ex_models.LRP import LRP

# 데이터 로드
def numpy_load(dir, file):
    file = os.path.join(dir, file)
    return np.load(file, allow_pickle=True)

def custom_collate_fn(batch):
    """
    Collate function to handle padding and relevance mask creation.
    """
    data, labels, lengths, relevance = zip(*batch)

    max_len = max(lengths)
    padded_data = torch.zeros(len(batch), max_len, data[0].size(-1))
    padded_relevance = torch.zeros(len(batch), max_len)
    padded_mask = torch.zeros(len(batch), max_len).bool()

    for i, (d, r, l) in enumerate(zip(data, relevance, lengths)):
        padded_data[i, :l] = d
        padded_relevance[i, :l] = r
        padded_mask[i, :l] = (r > 0)  # Create mask for relevance > 0

    return padded_data, torch.tensor(labels), torch.tensor(lengths), padded_relevance, padded_mask


# 데이터 디렉토리 설정
test_dir = 'dataset/lpi_testset/'
save_dir = 'ckpts/batch_/All_class_dB'
data_batch = numpy_load(save_dir, 'data_batch.npz')
labels_batch = numpy_load(save_dir, 'labels_batch.npz')
relevance_batch = numpy_load(save_dir, 'relevance_batch.npz')
lengths_batch = numpy_load(save_dir, 'lengths_batch.npy')
snr_batch = numpy_load(save_dir, 'snr_batch.npy')
idx_batch = numpy_load(save_dir, 'idx_batch.npy')

# 데이터 변환
data_dict = {key: torch.from_numpy(data_batch[key]) for key in data_batch}
labels_dict = {key: torch.from_numpy(labels_batch[key]) for key in labels_batch}
relevance_dict = {key: torch.from_numpy(relevance_batch[key]) for key in relevance_batch}

# 커스텀 데이터셋 정의
class CustomDataset(Dataset):
    def __init__(self, data_dict, labels_dict, relevance_dict, lengths_dict):
        self.data_dict = data_dict
        self.labels_dict = labels_dict
        self.relevance_dict = relevance_dict
        self.lengths_dict = lengths_dict

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        key = f'arr_{idx}'
        data = self.data_dict[key]
        labels = self.labels_dict[key]
        relevance = self.relevance_dict[key]
        lengths = self.lengths_dict[idx]
        return data, labels, lengths, relevance

# Train 함수
def Train(model_type, data_dict, labels_dict, relevance_dict, lengths_dict, waveforms, batch_size=64, epochs=500, learning_rate=0.001, weight_decay=1e-5, device_ids=[0, 1]):
    dataset = CustomDataset(data_dict, labels_dict, relevance_dict, lengths_dict)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    CEloss = nn.CrossEntropyLoss()

    # 모델 초기화
    if model_type == 'BiLSTM':
        model = BiLSTM(input_size=2, hidden_size=128, num_layers=2, num_classes=len(waveforms))
    elif model_type == 'UNet':
        model = UNet(in_channels=1, out_channels=len(waveforms))
    elif model_type == 'U2Net':
        model = U2Net(in_channels=1, out_channels=len(waveforms))
    else:
        raise ValueError("Invalid model_type. Choose from 'BiLSTM', 'UNet', 'U2Net'.")

    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_loss = float('inf')
    losses = []

    # 학습 루프
    for epoch in range(epochs):
        total_loss = 0.0
        progress = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for batch_idx, (data, labels, lengths, relevance, mask) in progress:
            data, labels = data.cuda(), labels.cuda()
            mask = mask.cuda()

            outputs = model(data, lengths)
            
            # 손실 계산: Relevance > 0 마스크 적용
            outputs = outputs[mask]
            labels = labels[mask]
            
            if len(labels) == 0:  # 유효 데이터 없음
                continue

            loss = CEloss(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            progress.set_postfix({'Batch': f'{batch_idx+1}/{len(dataloader)}', 'Loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

        print(f'Epoch : {epoch+1}/{epochs}, Loss : {avg_loss:.4f}')

        # Best 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'./ckpts/{waveforms[0]}_{model_type}_best_{best_loss:.4f}.pth')
            if best_loss < 0.002:
                break

        torch.cuda.empty_cache()

    # 최종 모델 저장
    torch.save(model.state_dict(), f'./ckpts/{waveforms[0]}_{model_type}_last_{best_loss:.4f}.pth')

# 실행
waveforms = ['waveform_1', 'waveform_2', 'waveform_3']  # 예시
Train(
    model_type='BiLSTM',
    data_dict=data_dict,
    labels_dict=labels_dict,
    relevance_dict=relevance_dict,
    lengths_dict=lengths_batch,
    waveforms=waveforms
)
