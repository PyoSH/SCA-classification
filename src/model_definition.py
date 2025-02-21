import torch.nn as nn
import torch.nn.functional as F
from sympy import transpose


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # RNN의 마지막 출력을 사용
        return out

class CRNN_base(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=3):
        super(CRNN_base, self).__init__()
        self.cnn1 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=80,
            stride=4,
            padding=0
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.cnn2 = nn.Conv1d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True) # 여기의 input_dim은 pool2에서 나오는 크기여야 함!!!
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # "x.shape = [batch_size, 1, 6615]"
        # print("Input:", x.shape)  # [batch, 1, 6615]
        x = self.cnn1(x)
        # print("After cnn1:", x.shape)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        # print("After pool1:", x.shape)

        x = self.cnn2(x)
        # print("After cnn2:", x.shape)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        # print("After pool2:", x.shape)

        lstm_out, _ = self.lstm(x)
        # print("After LSTM:", lstm_out.shape)

        x = lstm_out[:, -1, :]
        # print("After selecting last timestep:", x.shape)

        out = self.fc(x)
        # print("Final output:", out.shape)
        return out