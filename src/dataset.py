import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.feature_extraction import *

class AudioDataset(Dataset):
    def __init__(self, X, y, input_size):
        # self.X를 사용하여 마지막 차원을 n_mfcc 대신에 input_size로 설정 -> 텐서 크기 맞추기 위한 임시방편으로 보임.
        self.X = torch.tensor(X[:, :, :input_size], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class protoDataset(Dataset):
    def __init__(self, dataPath, feature_params):
        super().__init__()
        self.audioPath = os.path.join(dataPath,f'TestSet_{1}_1.mp3')
        self.labelPath = os.path.join(dataPath, f'text{1}.txt')
        self.feature_params = feature_params

        self.featureVector = get_mp3_to_mfcc(self.audioPath, feature_params.sr,
                                             feature_params.n_mfcc, feature_params.hop_length, feature_params.len_fft)
        self.label_raw = pd.read_csv(self.labelPath, header=None, sep='\t')
        self.label_processed = np.zeros((self.featureVector.shape[0],))  # MFCC 프레임 수에 맞는 레이블 배열 초기화

    def __len__(self):
        return len(self.featureVector)

    def __getitem__(self, idx):
        return self.featureVector[idx], self.label_processed[idx]

    def labelProcessing(self):
        self.label_raw.columns = ['start', 'end', 'label']

        # 레이블을 숫자로 매핑
        label_mapping = {'idling': 0, 'cutting': 1, 'hardcutting': 2}

        # 레이블 데이터 준비
        for _, row in self.label_raw.iterrows():
            start_frame = int(row['start'] * self.feature_params.sr / self.feature_params.hop_length)
            end_frame = int(row['end'] * self.feature_params.sr / self.feature_params.hop_length)
            self.label_processed[start_frame:end_frame] = label_mapping[row['label']]
