from src.model_definition import *
from src.train_utils import *
from src.dataset import *
import numpy as np

# model - 나중에 yaml 파일 읽는 방식으로 하면 더 좋을듯!
mfcc_conts = MFCC_params(44100, 40, 512, 2048)

hidden_size = 128
num_layers = 2
num_classes = 3
batch_size = 32
num_epochs = 10
seq_len = int(0.05 * mfcc_conts.sr)
class_labels = ['idling', 'cutting', 'HardCutting']

datapath = 'data'
model_saved_path = os.path.join('results', f'model_1_{1}.pth')

if __name__ == '__main__':

    # 학습된 모델 불러오기
    model = LSTMModel(input_dim=mfcc_conts.n_mfcc, hidden_dim=hidden_size, num_layers=num_layers,
                      output_dim=num_classes)
    model.load_state_dict(torch.load(model_saved_path))

    acc_array = []

    for item in range (2,10):
        # 데이터셋 구성 & 음향+라벨 전처리
        datas = ProtoDataset(datapath, class_labels, mfcc_conts, item)
        datas.label_processed = labelProcessing(datas.label_raw, datas.featureVector.shape[0], mfcc_conts)

        features = datas.featureVector[:, np.newaxis, :mfcc_conts.n_mfcc]

        test_set = AudioDataset(features, datas.label_processed, input_size=mfcc_conts.n_mfcc)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        avg_loss, avg_acc = evaluate_test(model, test_loader)
        print(f'{item} th dataset avg loss : {avg_loss}, avg acc : {avg_acc}')