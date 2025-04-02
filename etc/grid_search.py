import seaborn
from src.model_definition import *
from src.train_utils import *
from src.dataset import *
from config import cfg, update_config
import argparse
from loguru import logger
import numpy as np
from itertools import product
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

def train_model_grid(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    train_losses = []  # 각 epoch의 training loss 기록
    test_accuracies = []  # 각 epoch의 test accuracy 기록

    for epoch in range(num_epochs):
        model.train()  # 모델을 training 모드로 설정

        idx = 0
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

parser = argparse.ArgumentParser(description='Running audio classification')
parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
args = parser.parse_args()
update_config(cfg, args)

if not os.path.exists(cfg.PATH.TRAIN_PATH):
    logger.info("There is no dataset to learn :( ")
else:
    logger.info("Dataset found :) ")

logger.info("Running proto_train ...")
logger.info(f'DATA path: {cfg.PATH.TRAIN_PATH}')

mfcc_const = MFCC_params(cfg.FEATUREPARAMS.SAMPLING_RATE, cfg.FEATUREPARAMS.NUM_CEPSTRAL_COEFFICIENTS,
                         cfg.FEATUREPARAMS.HOP_LENGTH, cfg.FEATUREPARAMS.LEN_WINDOW)

if __name__ == '__main__':
    # 데이터셋 구성 & 음향 전처리
    datas_law = np.load(cfg.PATH.TRAIN_PATH)
    data_audio = datas_law[:, 0:-1]
    data_label = datas_law[:, -1]
    data_featureVector = audioProcessing(data_audio, mfcc_const)

    # 데이터셋 분할
    X_train, X_test, y_train, y_test = train_test_split(data_featureVector, data_label, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # 하이퍼파라미터 튜닝 범위 설정
    hidden_dims = [32, 64, 128, 256]
    num_layers = [1, 2, 3]
    batch_sizes = [16, 32, 64, 128]
    epochs = [10, 20, 30]

    hyperparameters = list(product(hidden_dims, num_layers, batch_sizes, epochs))

    best_accuracy = 0.0
    best_hyperparams = None
    train_dataset=None
    valid_dataset = None
    test_dataset = None

    for hidden_dim, num_layer, batch_size, epoch in hyperparameters:
        # 데이터셋 객체 생성
        train_dataset = AudioDataset(X_train, y_train, input_size=mfcc_const.n_mfcc)
        valid_dataset = AudioDataset(X_val, y_val, input_size=mfcc_const.n_mfcc)
        test_dataset = AudioDataset(X_test, y_test, input_size=mfcc_const.n_mfcc)

        # 데이터로더 객체 생성
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 모델 인스턴스 생성
        model = None
        if cfg.HYPERPARAMS.MODELTYPE == 'RNN':
            model = RNNModel(input_dim=mfcc_const.n_mfcc, hidden_dim=hidden_dim,
                             num_layers=num_layer, output_dim=cfg.HYPERPARAMS.NUM_CLASSES)
        elif cfg.HYPERPARAMS.MODELTYPE == 'LSTM':
            model = LSTMModel(input_dim=mfcc_const.n_mfcc, hidden_dim=hidden_dim,
                              num_layers=num_layer, output_dim=cfg.HYPERPARAMS.NUM_CLASSES)

        # 손실 함수 및 최적화 알고리즘 정의
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.HYPERPARAMS.LEARNING_RATE)

        # 모델 훈련
        train_model_grid(model=model, train_loader=train_loader, test_loader=valid_loader,
                    criterion=criterion, optimizer=optimizer, num_epochs=epoch)

        # 모델 평가
        test_accuracy = evaluate_model(model, test_loader)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_hyperparams = (hidden_dim, num_layer, batch_size, epoch)

        logger.info(f'Hyperparams: Hidden Dim: {hidden_dim}, Num Layers: {num_layer}, Batch Size: {batch_size}, Epochs: {epoch}')
        logger.info(f'Test Accuracy: {test_accuracy:.2f}%')

    logger.info(f'Best Hyperparameters: {best_hyperparams}')
    logger.info(f'Best Test Accuracy: {best_accuracy:.2f}%')

    # 최적 하이퍼파라미터로 최종 모델 학습
    hidden_dim, num_layer, batch_size, epoch = best_hyperparams
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = None
    if cfg.HYPERPARAMS.MODELTYPE == 'RNN':
        model = RNNModel(input_dim=mfcc_const.n_mfcc, hidden_dim=hidden_dim,
                         num_layers=num_layer, output_dim=cfg.HYPERPARAMS.NUM_CLASSES)
    elif cfg.HYPERPARAMS.MODELTYPE == 'LSTM':
        model = LSTMModel(input_dim=mfcc_const.n_mfcc, hidden_dim=hidden_dim,
                          num_layers=num_layer, output_dim=cfg.HYPERPARAMS.NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.HYPERPARAMS.LEARNING_RATE)

    train_model(model=model, train_loader=train_loader, test_loader=valid_loader,
                criterion=criterion, optimizer=optimizer, num_epochs=epoch)

    # 모델 평가 및 저장
    logger.info(f"Final Test Accuracy: {evaluate_model(model, test_loader):.2f}%")
    torch.save(model.state_dict(), cfg.PATH.MODEL_PATH)
    logger.info(f"Model saved: {cfg.PATH.MODEL_PATH}")
