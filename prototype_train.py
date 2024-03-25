from src.model_definition import *
from src.train_utils import *
from src.feature_extraction import *
from src.dataset import *

import numpy as np

# model
hidden_size = 128
num_layers = 2
num_classes = 3
batch_size = 32
num_epochs = 3
class_labels = ['idling', 'cutting', 'HardCutting']

mfcc_conts = MFCC_params(44100, 40, 512, 2048)

# datapath_local = "C:/Users/user/Desktop/3. 작업판단/data"
datapath_local = 'data'

if __name__ == '__main__':

    # 데이터셋 구성 & 음향+라벨 전처리
    datas = ProtoDataset(datapath_local, class_labels, mfcc_conts)
    datas.label_processed = labelProcessing(datas.label_raw, datas.featureVector.shape[0], mfcc_conts)

    # 데이터셋 분할
    X_train, X_test, y_train, y_test = train_test_split(datas.featureVector, datas.label_processed, test_size=0.2, random_state=42)

    # 예를 들어, 시퀀스 길이가 1인 경우 ????
    X_train = X_train[:, np.newaxis, :mfcc_conts.n_mfcc]
    X_test = X_test[:, np.newaxis, :mfcc_conts.n_mfcc]

    print("X_train shape before creating AudioDataset:", X_train.shape)

    # 데이터셋 객체 생성
    train_dataset = AudioDataset(X_train, y_train, input_size=mfcc_conts.n_mfcc)
    test_dataset = AudioDataset(X_test, y_test, input_size=mfcc_conts.n_mfcc)

    # 데이터로더 객체 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 모델 인스턴스 생성
    model = LSTMModel(input_dim=mfcc_conts.n_mfcc, hidden_dim=hidden_size, num_layers=num_layers, output_dim=num_classes)

    # 손실 함수 및 최적화 알고리즘 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 모델 훈련
    train_model(model=model, train_loader=train_loader, test_loader=test_loader, criterion=criterion, optimizer=optimizer, num_epochs=num_epochs)

    # 모델 평가
    # evaluate_model(model, test_loader)
    input_size = X_train.shape[2]
    print(f"Input size: {input_size}")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Test indices:", len(X_test))

    # 모델 저장
    # model_saved_path = datapath_local+'/learned'+f'/model_1_{1}.pth'
    model_saved_path = os.path.join('results', f'model_1_{1}.pth')
    print(model_saved_path)
    torch.save(model.state_dict(), model_saved_path)