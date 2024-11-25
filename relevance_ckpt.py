
import os
import torch
import numpy as np
import models._config as c
import matplotlib.pyplot as plt

from models._config import *
from ex_models.LRP import LRP
from models.LSTM import BiLSTM
from dataset.LPIdataset import LPIDataset
from torch.utils.data import DataLoader

from warnings import filterwarnings
from tqdm import tqdm
import time
filterwarnings("ignore")


def load_models(path='./ckpts/result_loss.pt', device='cuda:0'):
    model = BiLSTM(input_size=2, hidden_size=128, num_layers=2, num_classes=len(c.waveforms)).cuda()
    state_dict = {}
    for k, v in torch.load(path).items():
        nk = k.replace('module.', '')
        state_dict[nk] = v
    model.load_state_dict(state_dict)
    model.eval()
    
    lrp = LRP(model, device=device, epsilon=1e-6)
    return model ,lrp


fs = 100e6
train_dir = '/data/kiwan/LPI_KIWAN/'
test_dir = 'dataset/lpi_testset/'


datatype = c.datatypes[-1]
model_type = c.modeltypes[0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = LPIDataset(train_dir, c.waveforms, data_type=datatype, model_type='BiLSTM')
model, lrp = load_models(path='ckpts/pwnNoisy/BiLSTM_best_0.0638087096589583.pth', device = device)


relevance_batch, data_batch, labels_batch, lengths_batch, snr_batch, idx_batch, gt_batch = ([] for _ in range(7))


target_snr = range(-16, 2, 2)
target = c.waveforms

dataload = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=c.collate)

save_dir = 'ckpts/batch_/All_R>0_class_dB'
os.makedirs(save_dir, exist_ok=True)

from collections import defaultdict

with torch.no_grad():
    start_time = time.time()
    mismatch_count = 0  
    snr_mismatch_count = defaultdict(int)
    log_file = os.path.join(save_dir, 'mismatch_log.txt')  
    
    with open(log_file, 'w') as log: 
        progress = tqdm(enumerate(dataload), total=len(dataload), desc='Batch', unit='batch')
        for i, sample in progress:
            data, labels, lengths, snr, idx = sample
            data = data.to(device)
            labels = labels.to(device)
            relevance, gt_label = lrp.relevance(data, lengths)

            for j in range(len(data)):
                if gt_label[j] != labels[j]:
                    message = (
                        f"[Batch {i+1}/{len(dataload)}] GT: <<{c.waveforms[gt_label[j]]}-{idx[j]}>> "
                        f"at {snr[j]}dB : expected <<{c.waveforms[labels[j]]}>> (mismatch)\n"
                    )
                    log.write(message)  
                    mismatch_count += 1
                    snr_mismatch_count[snr[j]] += 1
                else:
                    gt_batch.append(gt_label)
                    relevance_batch.append(relevance)
                    data_batch.append(data)
                    labels_batch.append(labels)
                    lengths_batch.append(lengths)
                    snr_batch.append(snr)
                    idx_batch.append(idx)

                    elapsed_time = time.time() - start_time
                    estimated_total_time = elapsed_time / (i + 1) * len(dataload)
                    remaining_time = estimated_total_time - elapsed_time

                    snr_stats = ', '.join([f"{snr}dB: {count}" for snr, count in snr_mismatch_count.items()])
                    progress.set_postfix({
                        'Mismatch': mismatch_count,
                        'SNR': snr_stats,
                        'Elapsed': f'{elapsed_time:.2f}s',
                        'Remain': f'{remaining_time:.2f}s'
                    })
                
            np.savez(os.path.join(save_dir, 'data_batch.npz'), *[d.cpu().numpy() for d in data_batch])
            np.savez(os.path.join(save_dir, 'labels_batch.npz'), *[l.cpu().numpy() for l in labels_batch])
            np.savez(os.path.join(save_dir, 'relevance_batch.npz'), *[r.cpu().numpy() for r in relevance_batch])
            np.save(os.path.join(save_dir, 'gt_batch.npy'), np.array([g.cpu().numpy() for g in gt_batch]))
            np.save(os.path.join(save_dir, 'lengths_batch.npy'), lengths_batch)
            np.save(os.path.join(save_dir, 'snr_batch.npy'), snr_batch)
            np.save(os.path.join(save_dir, 'idx_batch.npy'), idx_batch)
                    

print("Data batches saved successfully.")