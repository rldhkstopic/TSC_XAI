import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from models.LSTM import BiLSTM_CA
from dataset.LPIdataset import LPIDataset
import os
import models._config as c

def Train_ca(
    model_type='BiLSTM_CrossAttention',
    data_dir='./data',
    waveforms=None,
    datatype='pwnNoisy',
    batch_size=64,
    epochs=500,
    learning_rate=0.001,
    weight_decay=1e-5,
    device_ids=[0, 1],
    query_len=10,  # Horizon-specific Query Length
    val_split=0.2,  # Validation split ratio
):
    if waveforms is None:
        raise ValueError("waveforms must be provided as a list of target classes.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model
    if model_type == 'BiLSTM_CA':
        
        model = BiLSTM_CA(
            input_size=2,  # For time-series data
            hidden_size=128,
            num_layers=2,
            num_classes=len(waveforms),
            factor=1
        )
    else:
        raise ValueError(f"Invalid model_type '{model_type}'. Only 'BiLSTM_CA' is supported.")
    
    model = nn.DataParallel(model, device_ids=device_ids).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    dataset = LPIDataset(data_dir, waveforms, data_type=datatype, model_type=model_type)

    
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=c.collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=c.collate)
    

    # Training loop
    best_loss = float('inf')
    best_state = None
    losses = []
    val_accuracies = []

    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for batch_idx, (data, labels, lengths, snrs, idx) in progress:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(data, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix({'Batch': f'{batch_idx+1}/{len(train_loader)}', 'Loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}')

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for data, labels, lengths, snrs, idx in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data, lengths)

                # Compute loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total
        val_loss /= len(val_loader)
        val_accuracies.append(val_accuracy)

        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4%}")

        # Save best model
        if avg_loss < best_loss and avg_loss < 0.5:
            best_loss = val_loss
            best_state = model.state_dict()
            torch.save(best_state, f'./ckpts/{datatype}/{model_type}4_best_{best_loss:.4f}.pth')
            print(f"New best model saved with loss {best_loss:.4f}")

        # Early stopping condition
        if best_loss < 0.002:
            print("Early stopping: Loss threshold reached.")
            break

        # Clear GPU cache to avoid memory issues
        torch.cuda.empty_cache()

    # Save the last model state
    torch.save(model.state_dict(), f'./ckpts/{datatype}/{model_type}4_last_{best_loss:.4f}.pth')
    print("Training complete. Final model saved.")

    return model, losses, val_accuracies