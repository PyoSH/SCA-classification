'''
MFCC 특징 벡터를 시각화하는 코드
표승현 2025-03-05
'''
from src.model_definition import *
from src.train_utils import *
from src.dataset import *
from config import cfg, update_config
import numpy as np
import argparse
from loguru import logger
from copy import deepcopy

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

logger.info("Running viz_MFCC ...")

mfcc_const = MFCC_params(cfg.FEATUREPARAMS.SAMPLING_RATE, cfg.FEATUREPARAMS.NUM_CEPSTRAL_COEFFICIENTS,
                         cfg.FEATUREPARAMS.HOP_LENGTH, cfg.FEATUREPARAMS.LEN_WINDOW)

if __name__ == '__main__':

    data_law = np.load(cfg.PATH.TEST_PATH)
    data_audio = data_law[:, 0:-1]
    data_label = data_law[:, -1]

    logger.info("data in - before MFCC")
    data_featureVector = audioProcessing(data_audio, mfcc_const)
    logger.info("data in - after MFCC")


    # 배치 차원 제거 (단일 샘플에 대한 활성화만 시각화)
    feature_maps = data_featureVector.squeeze(0)  # shape: [channels, time]
    # feature_sorted = feature_maps[sorted_idx, :]


    # 시각화: 각 행은 하나의 채널, 열은 시간축에 따른 활성화 값
    plt.figure(figsize=(10, 8))
    plt.imshow(data_featureVector.T,
               aspect='auto',
               origin='lower',
               interpolation='nearest', cmap='viridis')
    plt.colorbar(label='value')
    plt.xlabel('Reduced Time Index')
    plt.ylabel('features')
    plt.title(f'Feature vectors from MFCC')
    plt.show()