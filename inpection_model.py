'''
모델, 오디오, 라벨 데이터를 넣어서 가져온 모델이 입력한 데이터에 대해 예측하는 코드
표승현 2024-04-04
'''
from copy import deepcopy

from src.model_definition import *
from src.feature_extraction import *
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

logger.info("Running model_inspection ...")
logger.info(f'MODEL path: {cfg.PATH.MODEL_PATH}')


if __name__ == '__main__':
    # (1) 학습된 모델의 CNN layer 에서 가중치 추출
    model = CRNN_3().to(device)
    model.load_state_dict(torch.load(cfg.PATH.MODEL_PATH, weights_only=True))
    model.eval()

    # 모델의 cnn1 레이어 가중치 추출 (shape: [out_channels, in_channels, kernel_size])
    # filters = model.cnn3.weight.data.cpu().numpy()  # CPU로 이동
    # viz_filter_map(filters, 44100)

    # 데이터 불러오고, 모델에 넣을 준비
    data_raw = np.load(cfg.PATH.TEST_PATH)
    data_audio = data_raw[:, 0:-1]
    data_label = data_raw[:, -1]
    data_audio_std = deepcopy(data_audio)

    # audio standardization to mean 0, variance 1
    for i, row in enumerate(data_audio):
        row_std = (row - np.mean(row)) / np.std(row)
        data_audio_std[i, :] = row

    viz_feature_map(model, "cnn3", data_audio_std, device)