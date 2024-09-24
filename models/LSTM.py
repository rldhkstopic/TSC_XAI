import visdom
import torch
import torch.nn as nn
import _config as c
import torch.nn.utils.rnn as rnn_utils
from matplotlib import pyplot as plt


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.3)

        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Bidirectional이므로 hidden_size * 2
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, lstm_outputs=False):
        # Initial hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # LSTM output
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out_last = self.dropout(out[:, -1, :])  # Dropout
        out_fc = self.fc(out_last)

        if lstm_outputs:
            return out_fc, out  # (모델 출력, LSTM hidden outputs)
        else:
            return out_fc

    
    def train_model(self, train_loader, criterion, optimizer, num_epochs, device, snr_str, ckpt):
        
        vis = visdom.Visdom()
        assert vis.check_connection(), "이거 치셈 python -m visdom.server"

        losses = []  
        vis_window = vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1,)).cpu(),
            opts=dict(xlabel='Epoch', ylabel='Loss', title=f'Training Loss for SNR {snr_str}', legend=['Loss'])
        )
        
        
        best_loss = float('inf')  # Best loss 초기화
        best_state = None
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            
            for data_batch, labels_batch, _ in train_loader:
                data_batch = data_batch.to(device)
                labels_batch = labels_batch.to(device)

                optimizer.zero_grad() 
                outputs = self(data_batch)  
                
                loss = criterion(outputs, labels_batch)
                loss.backward()  
                optimizer.step() 

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            losses.append(avg_loss)
            
            vis.line(
                X=torch.tensor([epoch+1]).cpu(),
                Y=torch.tensor([avg_loss]).cpu(),
                win=vis_window,
                update='append'
            )

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
            
            # Best loss 갱신
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = self.state_dict()

            if best_loss < 0.01:
                break

        plt.figure()
        plt.plot(losses, label='Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training Loss for SNR {snr_str}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{ckpt}/loss_plot_{snr_str}.png")
        plt.close()
                    
        return best_state, best_loss
    
    @staticmethod        
    def collate(batch):
        data, labels, snrs = zip(*batch)  # batch에서 각 요소를 분리 (data, label, snr)
        data_pad = rnn_utils.pad_sequence([torch.tensor(seq, dtype=torch.float32) for seq in data], batch_first=True)
        # 각 sequence를 tensor로 변환하고, padding을 적용하여 하나의 tensor로 만듦 data는 batch_size x seq_len x input_size
        # 그 배치에서 가장 긴 sequence 길이를 기준으로 padding이 적용됨
        
        labels = torch.tensor([c.label_mapping[label] for label in labels], dtype=torch.long)
        snrs = torch.tensor(snrs, dtype=torch.float32)
        return data_pad, labels, snrs
