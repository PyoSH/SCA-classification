import seaborn
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

    for item in range (2,3):
        # 데이터셋 구성 & 음향+라벨 전처리
        datas = ProtoDataset(datapath, class_labels, mfcc_conts, item)
        datas.label_processed = labelProcessing(datas.label_raw, datas.featureVector.shape[0], mfcc_conts)

        X_train, X_test, y_train, y_test = train_test_split(datas.featureVector, datas.label_processed, test_size=0.99, random_state=42)
        X_test = X_test[:, np.newaxis, :mfcc_conts.n_mfcc]

        test_dataset = AudioDataset(X_test, y_test, input_size=mfcc_conts.n_mfcc)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        y_pred = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                for idx in predicted:
                    y_pred.append(predicted[idx])

        cm = confusion_matrix(y_test, y_pred)

    print(cm)
    seaborn.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

        # test_accuracy = evaluate_model(model, test_loader)
        # acc_array.append(test_accuracy)
        # print(f'{item} th dataset accuracy = {test_accuracy}')



    # for i in range(0, len(datas.label_processed), batch_size):
    # for i, label in enumerate(datas.label_processed):
    #
    #     # 입력 텐서 생성 [batch size, seq len, feature num] ??
    #     inputTensor = torch.tensor(datas.featureVector[np.newaxis, :, :], dtype=torch.float32)
    #     # inputTensor = torch.tensor(datas.featureVector[:batch_size, :seq_len, :mfcc_conts.n_mfcc], dtype=torch.float32)
    #     print("inputTensor shape:", inputTensor.shape)
    #
    #     with torch.no_grad():
    #         model.eval()
    #         outputTensor = model(inputTensor)
    #         predicted = torch.max(outputTensor.data, 1) # 문제 가능성
    #
    #     # 예측된 클래스 출력 - One hot encoding 해야 하는데.. 귀찮군
    #     predicted_class = int(predicted.argmax())  # Use .argmax() instead of .item()
    #     predicted_label = class_labels[predicted_class]
    #
    #     if

