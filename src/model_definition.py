import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.name = "LSTM"

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNNModel, self).__init__()
        self.name = "RNN"

        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
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
        self._initialize_weights()

    def forward(self, x):
        # x.shape = [batch_size, in_channels, time] = [batch_size, 1, 6615]
        # print(f"input size: [{x.shape}]")

        x = self.cnn(x)
        # print(f"CNN : [{x.shape}]")

        x = F.relu(x)

        x = self.bn(x)
        # print(f"batch norm: [{x.shape}]")

        x = self.pool(x)
        # print(f"max pooling size: [{x.shape}]")

        return x

    def _initialize_weights(self):
        """He initialize implement"""
        # init.xavier_uniform_(self.cnn.weight)
        init.kaiming_normal_(self.cnn.weight, nonlinearity='relu')

        if self.cnn.bias is not None:
            init.zeros_(self.cnn.bias)

class CNNBlock_test(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, pad):
        super(CNNBlock_test, self).__init__()
        self.cnn = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self._initialize_weights()

    def forward(self, x):
        # x.shape = [batch_size, in_channels, time] = [batch_size, 1, 6615]
        # print(f"input size: [{x.shape}]")

        x = self.cnn(x)
        # print(f"CNN : [{x.shape}]")

        x = self.bn(x)
        # print(f"batch norm: [{x.shape}]")

        x = F.relu(x)

        x = self.pool(x)
        # print(f"max pooling size: [{x.shape}]")

        return x

    def _initialize_weights(self):
        """He initialize implement"""
        # init.xavier_uniform_(self.cnn.weight)
        init.kaiming_normal_(self.cnn.weight, nonlinearity='relu')

        if self.cnn.bias is not None:
            init.zeros_(self.cnn.bias)

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.cnnBlock1 = CNNBlock(in_ch=1, out_ch=64, kernel_size=80, stride=4, pad=0)
        self.cnnBlock2 = CNNBlock(in_ch=64, out_ch=64, kernel_size=3, stride=1, pad=0)
        self.cnnBlock3 = CNNBlock(in_ch=64, out_ch=128, kernel_size=3, stride=1, pad=0)

    def forward(self, x):
        # x.shape = [batch_size, in_channels, time] = [batch_size, 1, 6615]
        # print(f"CNN FE input size: [{x.shape}]")
        # print("1st CNN layer")
        x = self.cnnBlock1(x)

        # print("2nd CNN layer")
        x = self.cnnBlock2(x)

        # print("3rd CNN layer")
        x = self.cnnBlock3(x)
        return x

class CNNFeatureExtractor_test(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor_test, self).__init__()
        self.cnnBlock1 = CNNBlock(in_ch=1, out_ch=64, kernel_size=160, stride=4, pad=0)
        self.cnnBlock2 = CNNBlock(in_ch=64, out_ch=64, kernel_size=3, stride=1, pad=0)
        self.cnnBlock3 = CNNBlock(in_ch=64, out_ch=128, kernel_size=3, stride=1, pad=0)

    def forward(self, x):
        # x.shape = [batch_size, in_channels, time] = [batch_size, 1, 6615]
        # print("1st CNN layer")
        x = self.cnnBlock1(x)

        # print("2nd CNN layer")
        x = self.cnnBlock2(x)

        # print("3rd CNN layer")
        x = self.cnnBlock3(x)
        return x

class CNNFeatureExtractor_CRNN8(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor_CRNN8, self).__init__()
        self.block1 = CNNBlock(in_ch=1, out_ch=64, kernel_size=80, stride=4, pad=0)  # C(64, 80/4)
        self.block2 = CNNBlock(in_ch=64, out_ch=64, kernel_size=3, stride=1, pad=1)  # C(64, 3)
        self.block3 = nn.Sequential(
            CNNBlock(in_ch=64, out_ch=128, kernel_size=3, stride=1, pad=1),
            CNNBlock(in_ch=128, out_ch=128, kernel_size=3, stride=1, pad=1),
            CNNBlock(in_ch=128, out_ch=128, kernel_size=3, stride=1, pad=1),
        )
        self.block4 = nn.Sequential(
            CNNBlock(in_ch=128, out_ch=256, kernel_size=3, stride=1, pad=1),
            CNNBlock(in_ch=256, out_ch=256, kernel_size=3, stride=1, pad=1),
        )
        self.block5 = nn.Sequential(
            CNNBlock(in_ch=256, out_ch=512, kernel_size=3, stride=1, pad=1),
            CNNBlock(in_ch=512, out_ch=512, kernel_size=3, stride=1, pad=1),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

class CNNFeatureExtractor_MultiScale(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor_MultiScale, self).__init__()
        self.block1 = CNNBlock(in_ch=1, out_ch=64, kernel_size=80, stride=4, pad=0)  # C(64, 80/4)
        self.block2 = CNNBlock(in_ch=64, out_ch=64, kernel_size=3, stride=1, pad=1)  # C(64, 3)


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


# Attention module (simple channel attention)
class AttentionModule(nn.Module):
    def __init__(self, in_dim):
        super(AttentionModule, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, in_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn_weights = self.attn(x)
        return x * attn_weights

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
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)

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

class C_test(nn.Module):
    def __init__(self, output_dim, input_dim=128, hidden_dim=128, num_layers=2):
        """
        Args:
            output_dim (int): 최종 출력 차원 (클래스 개수 등)
            input_dim (int): LSTM에 입력되는 feature 차원
            hidden_dim (int): LSTM 은닉 상태 차원
            num_layers (int): LSTM 층 수
        """
        super(C_test, self).__init__()
        self.name = "C_test"

        # 별도의 CNN feature 추출기
        self.feature_extractor = CNNFeatureExtractor_test()

        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)

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

class C_CRNN8(nn.Module):
    def __init__(self, output_dim, hidden_dim=128, num_layers=1):
        super(C_CRNN8, self).__init__()
        self.name = "C_CRNN8"
        self.feature_extractor = CNNFeatureExtractor_CRNN8()
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.feature_extractor(x)       # [B, 512, T]
        x = x.transpose(1, 2)               # [B, T, 512]
        lstm_out, _ = self.lstm(x)          # [B, T, H]
        x = lstm_out[:, -1, :]              # [B, H]
        return self.fc(x)

class C_MultiScale_1st(nn.Module):
    def __init__(self, output_dim, hidden_dim=128, num_layers=1):
        super(C_MultiScale_1st, self).__init__()
        self.name="C-MultiScale"

        # 1. multi scale CNN blocks
        self.cnnBlock_1_small = CNNBlock_test(in_ch=1, out_ch=64, kernel_size=44, stride=1, pad=0)      # 1ms
        self.cnnBlock_1_medium = CNNBlock_test(in_ch=1, out_ch=64, kernel_size=220, stride=1, pad=0)    # 5ms
        self.cnnBlock_1_large = CNNBlock_test(in_ch=1, out_ch=64, kernel_size=441, stride=1, pad=0)     # 10ms

        # # 2. Feature projection
        # self.cnnBlock_2_small = CNNBlock_test(in_ch=64, out_ch=128, kernel_size=44, stride=4, pad=0)  # 10ms
        # self.cnnBlock_2_medium = CNNBlock_test(in_ch=64, out_ch=128, kernel_size=, stride=4, pad=0)  # 10ms
        # self.cnnBlock_2_large = CNNBlock_test(in_ch=64, out_ch=128, kernel_size=44, stride=4, pad=0)  # 10ms

        # Attention after concat
        self.attention = AttentionModule(64*3)

        self.lstm = nn.LSTM(input_size=64*3, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        s = self.cnnBlock_1_small(x)
        m = self.cnnBlock_1_medium(x)
        l = self.cnnBlock_1_large(x)

        min_time = min(s.shape[2], m.shape[2], l.shape[2])
        s = s[:, :, :min_time]
        m = m[:, :, :min_time]
        l = l[:, :, :min_time]

        combined = torch.cat([s,m,l], dim=1)
        combined = combined.transpose(1,2)

        combined = self.attention(combined)

        lstm_out, _ = self.lstm(combined)  # [B, T, H]
        out = lstm_out[:, -1, :]  # [B, H]
        return self.fc(out)

class CRNN_3(nn.Module):
    def __init__(self, output_dim, input_dim=128, hidden_dim=128, num_layers=2):
        super(CRNN_3, self).__init__()
        self.name = "CRNN_3"

        # 별도의 CNN feature 추출기
        self.feature_extractor = CNNFeatureExtractor()

        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)  # 여기의 input_dim은 pool2에서 나오는 크기여야 함!!!
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