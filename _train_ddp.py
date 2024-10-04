import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import models._config as c
from dataset.RadarDataset import RadarSignalDataset  
from models.LSTM import BiLSTM


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size, dataset, params):
    print(f"Size of train dataset: {len(dataset)}")
    setup(rank, world_size)
    
    # 모델 생성 및 GPU 할당
    model = BiLSTM(params['input_size'], params['hidden_size'], params['num_layers'], params['num_classes']).to(rank)
    
    # DDP 모델로 래핑
    model = DDP(model, device_ids=[rank])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    # DistributedSampler를 사용하여 각 프로세스가 다른 데이터를 처리하도록 함
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    snr_loader = DataLoader(dataset, batch_size=params['batch_size'], sampler=sampler, collate_fn=model.module.collate)

    print(f"Rank {rank} has {len(snr_loader)} batches")
    best_state, best_loss = model.module.train_model(snr_loader, criterion, optimizer, params['num_epochs'], rank)
    
    # 모델 저장
    if rank == 0:  # Only save the model from rank 0 to avoid conflicts
        torch.save(best_state, f'./ckpts/result_loss_{best_loss:.4f}.pt')

    # DDP 정리
    cleanup()
    print("Train is done.")

def main_ddp(dataset, params):
    world_size = torch.cuda.device_count()  # Number of available GPUs
    mp.spawn(train_ddp, args=(world_size, dataset, params), nprocs=world_size, join=True)

if __name__ == "__main__":
    dataset = RadarSignalDataset(c.TrainData, c.signalTypes[0:c.typeSize], snr_max=17)
    params = {
        'input_size': 128,
        'hidden_size': 256,
        'num_layers': 2,
        'num_classes': 10,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'batch_size': 256,
        'num_epochs': 500
    }
    main_ddp(dataset, params)
