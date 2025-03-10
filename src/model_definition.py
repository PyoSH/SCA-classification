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

class CRNN_base(nn.Module):
    def __init__(self, input_dim=101, hidden_dim=128, num_layers=2, output_dim=3):
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

class CRNN_3(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=2, output_dim=3):
        super(CRNN_3, self).__init__()
        self.name = "CRNN_3"

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

        self.cnn3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers,
                            batch_first=True)  # 여기의 input_dim은 pool2에서 나오는 크기여야 함!!!
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

        x = self.cnn3(x)
        # print("After cnn3:", x.shape)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        # print("After pool3:", x.shape)

        # transpose [batch size, features, seq len] to [batch size, seq len, features]
        x = x.transpose(1,2)
        # print("transpose:", x.shape)

        rnn_out, _ = self.rnn(x)
        # print("After LSTM:", lstm_out.shape)

        x = rnn_out[:, -1, :]
        # print("After selecting last timestep:", x.shape)

        out = self.fc(x)
        # print("Final output:", out.shape)
        return out

class CLSTM_3(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=2, output_dim=3):
        super(CLSTM_3, self).__init__()
        self.name = "CLSTM_3"

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

        self.cnn3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True)  # 여기의 input_dim은 pool2에서 나오는 크기여야 함!!!
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

        x = self.cnn3(x)
        # print("After cnn3:", x.shape)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        # print("After pool3:", x.shape)

        # transpose [batch size, features, seq len] to [batch size, seq len, features]
        x = x.transpose(1,2)
        # print("transpose:", x.shape)

        lstm_out, _ = self.lstm(x)
        # print("After LSTM:", lstm_out.shape)

        x = lstm_out[:, -1, :]
        # print("After selecting last timestep:", x.shape)

        out = self.fc(x)
        # print("Final output:", out.shape)
        return out