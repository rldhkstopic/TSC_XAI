import argparse


def getParams():
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('-m', '--mode', type=str, default='train', help='Mode of operation (train/eval)')
    parser.add_argument('-t', '--mtype', type=str, default='LSTM', help='Type of model to train (LSTM/Attention)')

    parser.add_argument('-smin','--snr_min', type=int, default=0, help='Minimum SNR value')
    parser.add_argument('-smax','--snr_max', type=int, default=16, help='Maximum SNR value')
    parser.add_argument('--split_size', type=float, default=0.8, help='Train/Test split size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--input_size', type=int, default=2, help='Input size for the model')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size for the model')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the model')
    parser.add_argument('--num_classes', type=int, default=12, help='Number of output classes')

    args = parser.parse_args()

    params = {
        'mode': args.mode,
        'model_type': args.mtype,
        'snr_min': args.snr_min,
        'snr_max': args.snr_max,
        'split_size': args.split_size,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'input_size': args.input_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'num_classes': args.num_classes
    }
    
    return args
