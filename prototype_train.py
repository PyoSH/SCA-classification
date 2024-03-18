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

mfcc_conts = MFCC_params(48000, 24, 512, 2048)

if __name__ == '__main__':

    label_df = pd.read_csv(f'C:\\Users\\KRISO\\Desktop\\유진\\label3\\test1.txt', header=None, sep='\t')
    label_df.columns = ['start', 'end', 'label']

    # 레이블을 숫자로 매핑
    label_mapping = {'idling': 0, 'cutting': 1, 'hardcutting': 2}

    # MFCC 데이터 로드
    mfccs = np.load(f'C:\\Users\\KRISO\\Desktop\\유진\\MFCC_data\\TestSet_1_1_mfcc_normalized.npy')


    # 레이블 데이터 준비
    labels = np.zeros((mfccs.shape[0],))  # MFCC 프레임 수에 맞는 레이블 배열 초기화
    for _, row in label_df.iterrows():
        start_frame = int(row['start'] * mfcc_conts.sr / mfcc_conts.hop_length)
        end_frame = int(row['end'] * mfcc_conts.sr / mfcc_conts.hop_length)
        labels[start_frame:end_frame] = label_mapping[row['label']]

    # 데이터셋 분할
    X_train, X_test, y_train, y_test = train_test_split(mfccs, labels, test_size=0.2, random_state=42)

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
    model = LSTMModel(input_size=mfcc_conts.n_mfcc, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)

    # 손실 함수 및 최적화 알고리즘 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 모델 훈련
    num_epochs = 10  # 수정 가능
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # 모델 평가
    evaluate_model(model, test_loader)

    # 모델 저장
    torch.save(model.state_dict(), 'C:\\Users\\KRISO\\Desktop\\유진\\learned_LSTM\\1_n.pth')

