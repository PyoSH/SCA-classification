'''
모델, 오디오, 라벨 데이터를 넣어서 가져온 모델이 입력한 데이터에 대해 예측하는 코드
표승현 2024-04-04
'''
from src.model_definition import *
from src.train_utils import *
from src.dataset import *
from config import cfg, update_config
import numpy as np
import argparse
from loguru import logger

# 디바이스 설정: Apple Silicon의 MPS, CUDA, 또는 CPU
device = None
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
#     logger.info(f'GPU device found: {torch.cuda.get_device_name(0)}')
# else:
#     device = torch.device("cpu")
device = torch.device("cpu")
logger.info(f'selected device: {device}')

parser = argparse.ArgumentParser(description='Running audio classification')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)

args = parser.parse_args()
update_config(cfg, args)

if not os.path.exists(cfg.PATH.TEST_PATH):
    logger.info("There is no data in path :( ")
else:
    logger.info("Data found! :) ")

logger.info("Running proto_test ...")
logger.info(f'MODEL path: {cfg.PATH.MODEL_PATH} \n DATA path: {cfg.PATH.TEST_PATH}')

mfcc_const = MFCC_params(cfg.FEATUREPARAMS.SAMPLING_RATE, cfg.FEATUREPARAMS.NUM_CEPSTRAL_COEFFICIENTS,
                         cfg.FEATUREPARAMS.HOP_LENGTH, cfg.FEATUREPARAMS.LEN_WINDOW)

if __name__ == '__main__':

    # 학습된 모델 불러오기
    model = None
    if cfg.HYPERPARAMS.MODELTYPE == 'RNN':
        model = RNNModel(input_dim=mfcc_const.n_mfcc, hidden_dim=cfg.HYPERPARAMS.HIDDEN_SIZE,
                         num_layers=cfg.HYPERPARAMS.NUM_LAYERS,
                         output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    elif cfg.HYPERPARAMS.MODELTYPE == 'LSTM':
        model = LSTMModel(input_dim=mfcc_const.n_mfcc, hidden_dim=cfg.HYPERPARAMS.HIDDEN_SIZE,
                          num_layers=cfg.HYPERPARAMS.NUM_LAYERS,
                          output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(cfg.PATH.MODEL_PATH, weights_only=True))
    model.eval()

    data_law = np.load(cfg.PATH.TEST_PATH)
    data_audio = data_law[:, 0:-1]
    data_label = data_law[:, -1]
    logger.info("data in - before MFCC")
    data_featureVector = audioProcessing(data_audio, mfcc_const)
    logger.info("data in - after MFCC")
    test_set = AudioDataset(data_featureVector, data_label, input_size=mfcc_const.n_mfcc)
    # test_loader = DataLoader(test_set, batch_size=cfg.HYPERPARAMS.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    logger.info("processed data in ")
    print(eval_metrics_device(model, test_loader, cfg.HYPERPARAMS.LABEL_CLASS, device=device))