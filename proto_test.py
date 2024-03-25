import torch

from src.model_definition import *
from src.train_utils import *
from src.feature_extraction import *
from src.dataset import *
import numpy as np

# model - 나중에 yaml 파일 읽는 방식으로 하면 더 좋을듯!
mfcc_conts = MFCC_params(44100, 40, 512, 2048)

hidden_size = 128
num_layers = 2
num_classes = 3
batch_size = 32
num_epochs = 10
seq_len = 0.005 * mfcc_conts.sr
class_labels = ['idling', 'cutting', 'HardCutting']


datapath = 'data'
model_saved_path = os.path.join('results', f'model_1_{1}.pth')

if __name__ == '__main__':

    # 데이터셋 구성 & 음향+라벨 전처리
    datas = ProtoDataset(datapath, class_labels, mfcc_conts)
    datas.label_processed = labelProcessing(datas.label_raw, datas.featureVector.shape[0], mfcc_conts)

    #입력 텐서 생성 [batch size, seq len, feature num]
    # inputTensor = torch.tensor(datas.featureVector[np.newaxis, :, :], dtype=torch.float32)
    inputTensor = torch.tensor(datas.featureVector[:batch_size, :seq_len, :mfcc_conts.n_mfcc], dtype=torch.float32)
    print("inputTensor shape:", inputTensor.shape)
    # inputTensor = datas.featureVector

    # 학습된 모델 불러오기
    model = LSTMModel(input_dim=mfcc_conts.n_mfcc, hidden_dim=hidden_size, num_layers=num_layers, output_dim=num_classes)
    model.load_state_dict(torch.load(model_saved_path))

    # 모델 예측
    with torch.no_grad():
        model.eval()
        outputTensor = model(inputTensor)
        predicted = torch.max(outputTensor.data, 1) # 문제 가능성

    # 예측된 클래스 출력 - One hot encoding 해야 하는데.. 귀찮군
    predicted_class = int(predicted.argmax())  # Use .argmax() instead of .item()
    predicted_label = class_labels[predicted_class]

