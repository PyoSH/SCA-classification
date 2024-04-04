from src.model_definition import *
from src.train_utils import *
from src.feature_extraction import *
from src.dataset import *

import numpy as np
import copy

# model
hidden_size = 128
num_layers = 2
num_classes = 3
batch_size = 128
num_epochs = 4

mfcc_const = MFCC_params(44100, 40, 512, 2048)

datapath_local = os.path.join('data','set_50ms','data_50ms.npy')

if __name__ == '__main__':

    # 데이터셋 구성 & 음향 전처리
    datas_law = np.load(datapath_local)
    data_audio = datas_law[:, 0:-1] #오디오 피처 그대로임, 이거 MFCC로 특징 벡터 뽑아야 함
    data_label = datas_law[:, -1]
    data_featureVector = audioProcessing(data_audio, mfcc_const)
    print(data_featureVector.shape)
    ## (batch size, seq len, num_feature) = (128, data_featureVector.shape[1],mfcc_conts.n_mfcc)

    # 데이터셋 분할
    X_train, X_test, y_train, y_test = train_test_split(data_featureVector, data_label, test_size=0.2, random_state=42)

    # 데이터셋 객체 생성
    train_dataset = AudioDataset(X_train, y_train, input_size=mfcc_const.n_mfcc)
    test_dataset = AudioDataset(X_test, y_test, input_size=mfcc_const.n_mfcc)

    # 데이터로더 객체 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 모델 인스턴스 생성
    model = LSTMModel(input_dim=mfcc_const.n_mfcc, hidden_dim=hidden_size, num_layers=num_layers, output_dim=num_classes)

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

    # 모델 저장 - 중간중간 저장하는 기능 필요?
    model_saved_path = os.path.join('results', f'model_trained.pth')
    print(model_saved_path)
    torch.save(model.state_dict(), model_saved_path)