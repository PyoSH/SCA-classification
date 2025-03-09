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
import time

# 디바이스 설정: Apple Silicon의 MPS, CUDA, 또는 CPU
device = None
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f'GPU device found: {torch.cuda.get_device_name(0)}')
else:
    device = torch.device("cpu")
# device = torch.device("cpu")
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

class_labels = cfg.HYPERPARAMS.LABEL_CLASS

if __name__ == '__main__':
    model = None
    # 학습된 모델 불러오기
    if cfg.HYPERPARAMS.MODELTYPE == 'RNN':
        model = RNNModel(input_dim=mfcc_const.n_mfcc, hidden_dim=cfg.HYPERPARAMS.HIDDEN_SIZE,
                         num_layers=cfg.HYPERPARAMS.NUM_LAYERS,
                         output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    elif cfg.HYPERPARAMS.MODELTYPE == 'LSTM':
        model = LSTMModel(input_dim=mfcc_const.n_mfcc, hidden_dim=cfg.HYPERPARAMS.HIDDEN_SIZE,
                          num_layers=cfg.HYPERPARAMS.NUM_LAYERS,
                          output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    elif cfg.HYPERPARAMS.MODELTYPE == 'CRNN':
        model = CRNN_3().to(device)
    elif cfg.HYPERPARAMS.MODELTYPE == 'CLSTM':
        model = CLSTM_3().to(device)

    model.load_state_dict(torch.load(cfg.PATH.MODEL_PATH, weights_only=True))
    model.eval()

    data_raw = np.load(cfg.PATH.TEST_PATH)
    data_audio = data_raw[:, 0:-1]
    data_label = data_raw[:, -1]
    data_audio_std = deepcopy(data_audio)

    # audio standardization to mean 0, variance 1
    for i, row in enumerate(data_audio):
        row_std = (row - np.mean(row)) / np.std(row)
        data_audio_std[i, :] = row

    tensor_audio = None
    feature_vector = None
    sum_diff = 0

    for idx in range(np.size(data_label)):
        raw_audio = data_audio_std[idx, :]
        logger.info(f'start, {raw_audio.shape}')
        start_t = time.time()

        if cfg.HYPERPARAMS.MODELTYPE != 'CRNN':
            featureVector = get_frame_to_mfcc(raw_audio, samplingRate=mfcc_const.sr,
                                              num_cepstralCoefficient=mfcc_const.n_mfcc,
                                              hop_length=mfcc_const.hop_length, len_fft=mfcc_const.len_fft)
            logger.info('audio MFCC processed')
            tensor_audio = torch.tensor(featureVector[:, :, :mfcc_const.n_mfcc], dtype=torch.float32).to(device)  # 입력 텐서 주의!!!!

        elif cfg.HYPERPARAMS.MODELTYPE == 'CRNN':
            tensor_audio = torch.tensor(raw_audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        logger.info(f'predict start, {raw_audio.shape}')
        with torch.no_grad():
            model.eval()
            outputs = model(tensor_audio)

        end_t = time.time()
        logger.info('predict done')

        diff = end_t - start_t
        sum_diff += diff

        _, predicted = torch.max(outputs.data, 1)
        predicted_class = class_labels[predicted.item()]
        logger.info(f'Predicted: {predicted_class}, diff t: {diff*1000:.3f} ms')
