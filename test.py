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
from copy import deepcopy

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

logger.info("Running test code ...")
logger.info(f'MODEL path: {cfg.PATH.MODEL_PATH} \n DATA path: {cfg.PATH.TEST_PATH}')

mfcc_const = MFCC_params(cfg.FEATUREPARAMS.SAMPLING_RATE, cfg.FEATUREPARAMS.NUM_CEPSTRAL_COEFFICIENTS,
                         cfg.FEATUREPARAMS.HOP_LENGTH, cfg.FEATUREPARAMS.LEN_WINDOW)

if __name__ == '__main__':

    # 학습된 모델 불러오기
    model = None
    # config에서 model type을 가져오기
    model_type = cfg.HYPERPARAMS.MODELTYPE

    # 모델 선택
    if model_type == 'B1':
        model = B1(input_dim=40, hidden_dim=cfg.HYPERPARAMS.HIDDEN_SIZE,
                   num_layers=cfg.HYPERPARAMS.NUM_LAYERS,
                   output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    elif model_type == 'B2-small':
        model = B2_small(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    elif model_type == 'B2-middle':
        model = B2_middle(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    elif model_type == 'B2-large':
        model = B2_large(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    elif model_type == 'B3':
        model = B3(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    elif model_type == 'B4':
        model = B4(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    elif model_type == 'P':
        model = P(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
        # model = P2(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    else:
        raise ValueError(f"Model type {model_type} is not recognized.")

    # model.load_state_dict(torch.load(cfg.PATH.MODEL_PATH, weights_only=True))
    model.load_state_dict(torch.load(cfg.PATH.MODEL_PATH, map_location='cpu'))
    model.eval()

    data_law = np.load(cfg.PATH.TEST_PATH)
    data_audio = data_law[:, 0:-1]
    data_label = data_law[:, -1]

    test_set = None
    is_HybridModel = True

    if is_HybridModel:
        data_audio_std = deepcopy(data_audio)

        # audio standardization to mean 0, variance 1
        for i, row in enumerate(data_audio):
            row_std = (row - np.mean(row)) / np.std(row)
            data_audio_std[i, :] = row_std

        test_set = RawWaveformDataset(data_audio_std, data_label)
    else:
        logger.info("data in - before MFCC")
        data_featureVector = audioProcessing(data_audio, mfcc_const)
        logger.info("data in - after MFCC")
        test_set = AudioDataset(data_featureVector, data_label, input_size=mfcc_const.n_mfcc)

    # test_loader = DataLoader(test_set, batch_size=cfg.HYPERPARAMS.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    logger.info("processed data in ")
    print(eval_metrics_device(model, test_loader, cfg.HYPERPARAMS.LABEL_CLASS, device=device))