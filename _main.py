from models._config import C
from __params import getParams
from _train import Train
# from _test import Test
# from _explain import Explain

c = C()

if __name__ == "__main__":
    args = getParams()
    
    data_dir = '/data/kiwan/LPI_KIWAN/'
    datatypes = ['Signal', 'Noise', 'Noisy', 'pwnNoisy']
    waveforms = ['Barker', 'Costas', 'Frank', 'LFM', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4']
    
    if args.mode == 'train':
        Train(batch_size=args.batch_size, epochs=args.num_epochs)
    # elif args.mode == 'eval':
    #     Test(dataset, mtype=params['model_type'])
    # elif args.mode == 'explain':
    #     explain_set(dataset)