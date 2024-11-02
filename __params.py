import argparse

def getParams():
    parser = argparse.ArgumentParser(description='Training and Evaluation parameters')
    
    # 실행 모드 및 모델 유형
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'eval', 'explain'], 
                        help='Mode of operation (train/eval/explain)')
    parser.add_argument('-t', '--model_type', type=str, default='BiLSTM', choices=['BiLSTM', 'UNet', 'U2Net'], 
                        help='Type of model to use (BiLSTM/UNet/U2Net)')
    
    # 데이터 관련 파라미터
    parser.add_argument('--data_dir', type=str, default='/data/kiwan/LPI_KIWAN/', help='Directory for dataset')
    parser.add_argument('--datatypes', type=str, nargs='+', default=['Signal', 'Noise', 'Noisy', 'pwnNoisy'],
                        help='List of data types to use (default: [Signal, Noise, Noisy, pwnNoisy])')
    parser.add_argument('--waveforms', type=str, nargs='+', default=['Barker', 'Costas', 'Frank', 'LFM', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4'],
                        help='List of waveform types (default: common LPI types)')

    # 훈련 관련 파라미터
    parser.add_argument('--snr_min', type=int, default=0, help='Minimum SNR value')
    parser.add_argument('--snr_max', type=int, default=16, help='Maximum SNR value')
    parser.add_argument('--split_size', type=float, default=0.8, help='Train/Test split size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    
    # 모델 구조 관련 파라미터
    parser.add_argument('--input_size', type=int, default=2, help='Input size for the model')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size for the model (LSTM only)')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the model (LSTM only)')
    parser.add_argument('--num_classes', type=int, default=12, help='Number of output classes')

    # 평가 관련 파라미터
    parser.add_argument('--model_path', type=str, default=None, help='Path to the trained model weights file for evaluation')

    args = parser.parse_args()

    # 딕셔너리로 반환할 경우 사용하기 편리할 수 있음
    params = {
        'mode': args.mode,
        'model_type': args.model_type,
        'data_dir': args.data_dir,
        'datatypes': args.datatypes,
        'waveforms': args.waveforms,
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
        'num_classes': args.num_classes,
        'model_path': args.model_path
    }
    
    return args
