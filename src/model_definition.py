import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.name = "LSTM"

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNNModel, self).__init__()
        self.name = "RNN"

        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # RNN의 마지막 출력을 사용
        return out

class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, pad):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)

    def forward(self, x):
        # x.shape = [batch_size, in_channels, time] = [batch_size, 1, 6615]
        x = self.cnn(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)

        return x

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.cnnBlock1 = CNNBlock(in_ch=1, out_ch=64, kernel_size=80, stride=4, pad=0)
        self.cnnBlock2 = CNNBlock(in_ch=64, out_ch=64, kernel_size=3, stride=1, pad=0)
        self.cnnBlock3 = CNNBlock(in_ch=64, out_ch=128, kernel_size=3, stride=1, pad=0)

    def forward(self, x):
        # x.shape = [batch_size, in_channels, time] = [batch_size, 1, 6615]
        x = self.cnnBlock1(x)
        x = self.cnnBlock2(x)
        x = self.cnnBlock3(x)

        return x

class CLSTM_3(nn.Module):
    def __init__(self, output_dim, input_dim=128, hidden_dim=128, num_layers=2):
        """
        Args:
            output_dim (int): 최종 출력 차원 (클래스 개수 등)
            input_dim (int): LSTM에 입력되는 feature 차원
            hidden_dim (int): LSTM 은닉 상태 차원
            num_layers (int): LSTM 층 수
        """
        super(CLSTM_3, self).__init__()
        self.name = "CLSTM_3"

        # 별도의 CNN feature 추출기
        self.feature_extractor = CNNFeatureExtractor()

        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # 최종 완전연결 레이어
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x.shape = [batch_size, 1, 6615]
        # 1) CNN을 통해 특징 맵 추출
        x = self.feature_extractor(x)
        # x.shape 예) [batch_size, 128, new_time_length]

        # 2) [batch, channel, time] → [batch, time, channel]
        x = x.transpose(1, 2)

        # 3) LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out.shape = [batch_size, new_time_length, hidden_dim]

        # 4) 마지막 타임스텝 벡터만 추출
        x = lstm_out[:, -1, :]  # [batch_size, hidden_dim]

        # 5) FC를 통한 최종 출력
        out = self.fc(x)

        return out

class CRNN_3(nn.Module):
    def __init__(self, output_dim, input_dim=128, hidden_dim=128, num_layers=2):
        super(CRNN_3, self).__init__()
        self.name = "CRNN_3"

        # 별도의 CNN feature 추출기
        self.feature_extractor = CNNFeatureExtractor()

        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)  # 여기의 input_dim은 pool2에서 나오는 크기여야 함!!!
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x.shape = [batch_size, 1, 6615]
        # 1) CNN을 통해 특징 맵 추출
        x = self.feature_extractor(x)
        # x.shape 예) [batch_size, 128, new_time_length]

        # 2) [batch, channel, time] → [batch, time, channel]
        x = x.transpose(1, 2)

        rnn_out, _ = self.rnn(x)
        # print("After RNN:", rnn_out.shape)

        x = rnn_out[:, -1, :]
        # print("After selecting last timestep:", x.shape)

        out = self.fc(x)
        # print("Final output:", out.shape)

        return out