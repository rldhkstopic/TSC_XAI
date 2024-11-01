

def eval_set(dataset, mtype='LSTM'):
    print(f"Size of test dataset: {len(dataset)}")
    for snr in range(params['snr_min'], params['snr_max']+1, 2):
        ckpt = os.path.join("./ckpts/", snr_str:=f"SNR-{snr}dB" if snr != 0 else " 0dB")
        ckpts = [f for f in os.listdir(ckpt) if f.endswith(".pt") and f.startswith("BiLSTM")]
        
        model = BiLSTM(params['input_size'], params['hidden_size'], params['num_layers'], params['num_classes']).to(c.device)
        model.load_state_dict(torch.load(f"{ckpt}/{ckpts[0]}"))
        model.eval()
        criterion = nn.CrossEntropyLoss()
        snr_dataset = [(data, label, data_snr) for data, label, data_snr in dataset if data_snr == snr]
        snr_loader = DataLoader(snr_dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=model.collate)
        
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in snr_loader:
                data, labels = batch[0].to(c.device), batch[1].to(c.device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
        acc = correct / total * 100
        avg_loss = total_loss / len(snr_loader)
        
        print(f"SNR -{snr}dB | Accuracy: {acc:.2f}% | Average Loss: {avg_loss:.4f}")