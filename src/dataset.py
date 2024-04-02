import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.feature_extraction import *

class AudioDataset(Dataset):
    def __init__(self, X, y, input_size):
        # self.X를 사용하여 마지막 차원 input_size로 설정 -> 텐서 크기 맞추기 위한 임시방편으로 보임.
        self.X = torch.tensor(X[:, :, :input_size], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
class InputDataset(Dataset):
    def __init__(self, X, y, batch_size, frame_size, sample_rate, input_size):
        # self.X = torch.tensor(X[:batch_size, :int(frame_size*sample_rate), :input_size], dtype=torch.float64)
        # self.y = torch.tensor(y[:batch_size, :int(frame_size*sample_rate), :], dtype=torch.long)
        self.X = torch.tensor(X[:batch_size, :, :input_size], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, item):
        return self.X[item], self.y[item]

class ProtoDataset(Dataset):
    def __init__(self, dataPath, featureParams, item=1):
        self.audioPath = os.path.join(dataPath,'audio',f'TestSet_{item}_1.mp3')
        self.labelPath = os.path.join(dataPath,'label', f'test{item}.txt')
        self.featureParams = featureParams

        self.featureVector = get_mp3_to_mfcc(self.audioPath, self.featureParams.sr, self.featureParams.n_mfcc,
                                             self.featureParams.hop_length, self.featureParams.len_fft).T # 전치!!!
        self.label_raw = pd.read_csv(self.labelPath, header=None, sep='\t')
        self.label_processed = None

    def __len__(self):
        return len(self.featureVector)

    def __getitem__(self, idx):
        return self.featureVector[idx], self.label_processed[idx]

def labelProcessing(label_raw, vectorShape, label_class, sampleRate, len_frame):
    label_raw.columns = ['start', 'end', 'label']

    # 레이블을 숫자로 매핑
    # label_mapping = {'idling': 0, 'cutting': 1, 'hardcutting': 2}

    # 레이블 데이터 준비
    label_processed = np.zeros(vectorShape)  # 모델 입력 차원(= 특징 벡터 열 개수)에 맞는 레이블 배열 초기화
    for _, row in label_raw.iterrows():
        start_frame = int(row['start'] * sampleRate / len_frame) # audio frame 단위.
        end_frame = int(row['end'] * sampleRate / len_frame)
        label_processed[start_frame:end_frame] = label_class[row['label']]

    return label_processed

def zeropad1d(A, length):
    retVal = np.zeros(length)
    retVal[:len(A)] = A
    return retVal
