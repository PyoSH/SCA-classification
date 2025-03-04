from copy import deepcopy

from src.model_definition import *
from src.train_utils import *
from src.dataset import *
from config import cfg, update_config
import argparse
from loguru import logger
import numpy as np

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

if not os.path.exists(cfg.PATH.TRAIN_PATH):
    logger.info("There is no dataset to learn :( ")
else:
    logger.info("Dataset found :) ")

logger.info("Running train code ...")
logger.info(f'DATA path: {cfg.PATH.TRAIN_PATH}')

mfcc_const = MFCC_params(cfg.FEATUREPARAMS.SAMPLING_RATE, cfg.FEATUREPARAMS.NUM_CEPSTRAL_COEFFICIENTS,
                         cfg.FEATUREPARAMS.HOP_LENGTH, cfg.FEATUREPARAMS.LEN_WINDOW)

if __name__ == '__main__':

    # 데이터셋 구성 & 음향 전처리
    datas_law = np.load(cfg.PATH.TRAIN_PATH)
    data_audio = datas_law[:, 0:-1] #오디오 피처 그대로임, 이거 MFCC로 특징 벡터 뽑아야 함
    data_label = datas_law[:, -1]

    data_audio_input = None
    if cfg.HYPERPARAMS.MODELTYPE != 'CRNN':

        logger.info("data in - before MFCC")
        data_featureVector = audioProcessing(data_audio, mfcc_const)
        logger.info("data in - after MFCC")
        data_audio_input = data_featureVector

    elif cfg.HYPERPARAMS.MODELTYPE == 'CRNN':
        data_audio_std = deepcopy(data_audio)

        # audio standardization to mean 0, variance 1
        for i, row in enumerate(data_audio):
            row_std = (row - np.mean(row)) / np.std(row)
            data_audio_std[i, :] = row

        data_audio_input = data_audio_std

    # 데이터셋 분할
    X_train, X_test, y_train, y_test = train_test_split(data_audio_input, data_label, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # 데이터셋 객체 생성
    train_dataset = None
    valid_dataset = None
    test_dataset = None

    if cfg.HYPERPARAMS.MODELTYPE != 'CRNN':
        train_dataset = AudioDataset(X_train, y_train, input_size=mfcc_const.n_mfcc)
        valid_dataset = AudioDataset(X_val, y_val, input_size=mfcc_const.n_mfcc)
        test_dataset = AudioDataset(X_test, y_test, input_size=mfcc_const.n_mfcc)

    elif cfg.HYPERPARAMS.MODELTYPE == 'CRNN':
        train_dataset = RawWaveformDataset(X_train, y_train)
        valid_dataset = RawWaveformDataset(X_val, y_val)
        test_dataset = RawWaveformDataset(X_test, y_test)

    # 데이터로더 객체 생성
    train_loader = DataLoader(train_dataset, batch_size=cfg.HYPERPARAMS.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.HYPERPARAMS.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.HYPERPARAMS.BATCH_SIZE, shuffle=False)

    # 모델 인스턴스 생성
    model = None
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

    # 손실 함수 및 최적화 알고리즘 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.HYPERPARAMS.LEARNING_RATE)

    # 모델 훈련
    train_model_device(model=model, train_loader=train_loader, test_loader=test_loader, criterion=criterion,
                       optimizer=optimizer, num_epochs=cfg.HYPERPARAMS.NUM_EPOCHS, device=device)

    # 모델 평가
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    print(eval_metrics_device(model, test_loader, cfg.HYPERPARAMS.LABEL_CLASS, device))

    # 모델 저장 - 중간중간 저장하는 기능 필요?
    torch.save(model.state_dict(), cfg.PATH.MODEL_PATH)
    logger.info(f"Model saved: {cfg.PATH.MODEL_PATH}")