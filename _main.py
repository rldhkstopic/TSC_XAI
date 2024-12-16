from __params import getParams
from _train import Train
from _train_Rp import Train_ca
# from _test import Test
# from _explain import Explain

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    args = getParams()

    data_dir = '/data/kiwan/LPI_KIWAN/'
    rp_dir = '/home/kiwan/TSC_XAI/ckpts/batch_/All_R>0_class_dB/data_batch.npz'
    datatypes = ['Signal', 'Noise', 'Noisy', 'pwnNoisy']
    waveforms = ['Barker', 'Costas', 'Frank', 'LFM', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4']
    
    # 모델 유형 확인 및 유효성 검사
    model_type = args.model_type
    if model_type in ['UNet', 'U2Net']:
        data_dir = '/data/kiwan/LPI_KIWAN_STFT/'
        

    # Train, Test, 또는 Explain 모드 실행
    if args.mode == 'train':        
        Train(
            model_type=model_type,
            batch_size=args.batch_size,
            epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            data_dir=data_dir,
            datatype=datatypes[-1],
            waveforms=waveforms
        )
    elif args.mode == 'eval':
        Train_ca(
            model_type='BiLSTM_CA',
            data_dir=data_dir,
            datatype=datatypes[-1],
            waveforms=waveforms,
            query_len=10,
            val_split=0.1
            )
    elif args.mode == 'explain':
        # Explain 함수 호출 (explain_set 인스턴스를 전달)
        # Explain(explain_set=dataset)
        pass
    else:
        raise ValueError("Invalid mode specified. Choose from 'train', 'eval', 'explain'.")
