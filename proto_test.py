'''
모델, 오디오, 라벨 데이터를 넣어서 가져온 모델이 입력한 데이터에 대해 예측하는 코드
표승현 2024-04-04
'''

from src.model_definition import *
from src.train_utils import *
from src.dataset import *
import numpy as np

# model - 나중에 yaml 파일 읽는 방식으로 하면 더 좋을듯!
mfcc_const = MFCC_params(44100, 40, 512, 2048)

hidden_size = 128
num_layers = 2
num_classes = 3
batch_size = 128
num_epochs = 10
seq_len = int(0.05 * mfcc_const.sr)

datapath_local = os.path.join('data','set_50ms', 'test_50ms.npy')
model_saved_path = os.path.join('results', f'model_trained.pth')

if __name__ == '__main__':

    # 학습된 모델 불러오기
    model = LSTMModel(input_dim=mfcc_const.n_mfcc, hidden_dim=hidden_size, num_layers=num_layers,
                      output_dim=num_classes)
    model.load_state_dict(torch.load(model_saved_path))

    acc_array = []

    for item in range (3,4):
        # 데이터셋 구성 & 음향+라벨 전처리

        data_law = np.load(datapath_local)
        data_audio = data_law[:, 0:-1]
        data_label = data_law[:, -1]
        data_featureVector = audioProcessing(data_audio, mfcc_const)
        print(data_featureVector.shape)
        test_set = AudioDataset(data_featureVector, data_label, input_size=mfcc_const.n_mfcc)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        avg_loss, avg_acc = evaluate_test(model, test_loader)
        print(f'{item} th dataset avg loss : {avg_loss}, avg acc : {avg_acc}')