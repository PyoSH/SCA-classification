'''
모델, 오디오, 라벨 데이터를 넣어서 가져온 모델이 입력한 데이터에 대해 예측하는 코드
표승현 2024-04-04
'''
from copy import deepcopy

from src.model_definition import *
from src.train_utils import *
from src.dataset import *
from config import cfg, update_config
import numpy as np
import argparse
from loguru import logger

# 디바이스 설정: Apple Silicon의 MPS, CUDA, 또는 CPU
device = None
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f'GPU device found: {torch.cuda.get_device_name(0)}')
else:
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
    model = CRNN_3().to(device)
    model.load_state_dict(torch.load(cfg.PATH.MODEL_PATH, weights_only=True))
    model.eval()

    acc_array = []

    data_raw = np.load(cfg.PATH.TEST_PATH)
    data_audio = data_raw[:, 0:-1]
    data_label = data_raw[:, -1]
    data_audio_std = deepcopy(data_audio)

    # audio standardization to mean 0, variance 1
    for i, row in enumerate(data_audio):
        row_std = (row - np.mean(row)) / np.std(row)
        data_audio_std[i, :] = row

    test_set = RawWaveformDataset(data_audio_std, data_label)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    print(eval_metrics_device(model, test_loader, cfg.HYPERPARAMS.LABEL_CLASS, device=device))