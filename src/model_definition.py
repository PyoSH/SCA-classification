import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
import torchaudio

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.name = "MFCC+LSTM"

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
        x = self.cnn(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x

    def _initialize_weights(self):
        """He initialize implement"""
        # init.xavier_uniform_(self.cnn.weight)
        init.kaiming_normal_(self.cnn.weight, nonlinearity='relu')

        if self.cnn.bias is not None:
            init.zeros_(self.cnn.bias)

class CNNFeatureExtractor(nn.Module):
    def __init__(self, kernel_init=80):
        super(CNNFeatureExtractor, self).__init__()
        self.cnnBlock1 = CNNBlock(in_ch=1, out_ch=40, kernel_size=kernel_init, stride=4, pad=0)
        self.cnnBlock2 = CNNBlock(in_ch=40, out_ch=40, kernel_size=3, stride=1, pad=0)
        self.cnnBlock3 = CNNBlock(in_ch=40, out_ch=128, kernel_size=3, stride=1, pad=0)
        self.cnnBlock4 = CNNBlock(in_ch=128, out_ch=256, kernel_size=3, stride=1, pad=0)

    def forward(self, x):
        # x.shape = [batch_size, in_channels, time] = [batch_size, 1, 6615]
        # print(f"CNN FE input size: [{x.shape}]")
        x = self.cnnBlock1(x)
        x = self.cnnBlock2(x)
        x = self.cnnBlock3(x)
        x = self.cnnBlock4(x)

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

def initialize_mel_filter(conv_layer, sr, kernel_size, n_filters):
    # Generate mel filterbank (torch)
    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=kernel_size,
        f_min=0.0,
        f_max=sr / 2,
        n_mels=n_filters,
        sample_rate=sr,
        norm=None
    )  # [kernel_size, n_mels]

    mel_fb = mel_fb.T  # [n_mels, kernel_size]
    mel_fb = mel_fb.unsqueeze(1)  # [n_mels, 1, kernel_size]

    with torch.no_grad():
        conv_layer.weight.copy_(mel_fb)
        # conv_layer.weight.requires_grad = False  # 학습 중 가중치 고정
        conv_layer.weight.requires_grad = True  # 학습 중 가중치 갱신

class B2_small(nn.Module):
    '''
    SingleScale_layer4_small
    '''
    def __init__(self, output_dim, input_dim=128, hidden_dim=128, num_layers=2):
        super(B2_small, self).__init__()
        self.name = "SingleScale CNN-small"

        self.feature_extractor = CNNFeatureExtractor(kernel_init=64)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x.shape = [batch_size, 1, 6615]
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)  # lstm_out.shape = [batch_size, new_time_length, hidden_dim]
        x = lstm_out[:, -1, :]  # [batch_size, hidden_dim]

        return self.fc(x)

class B2_middle(nn.Module):
    '''
    SingleScale_layer4_middle
    '''

    def __init__(self, output_dim, input_dim=128, hidden_dim=128, num_layers=2):
        super(B2_middle, self).__init__()
        self.name = "SingleScale CNN-middle"

        self.feature_extractor = CNNFeatureExtractor(kernel_init=256)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x.shape = [batch_size, 1, 6615]
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)  # lstm_out.shape = [batch_size, new_time_length, hidden_dim]
        x = lstm_out[:, -1, :]  # [batch_size, hidden_dim]

        return self.fc(x)

class B2_large(nn.Module):
    '''
    SingleScale_layer4_large
    '''

    def __init__(self, output_dim, input_dim=128, hidden_dim=128, num_layers=2):
        super(B2_large, self).__init__()
        self.name = "SingleScale CNN-large"

        self.feature_extractor = CNNFeatureExtractor(kernel_init=4096)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x.shape = [batch_size, 1, 6615]
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)  # lstm_out.shape = [batch_size, new_time_length, hidden_dim]
        x = lstm_out[:, -1, :]  # [batch_size, hidden_dim]

        return self.fc(x)

class B3(nn.Module):
    '''
    B3 : Multiscale with no Mel initialization, no attention
    '''
    def __init__(self, output_dim, hidden_dim=128, num_layers=2):
        super(B3, self).__init__()
        self.name="MultiScale (Kaiming)"

        # 1. multi scale CNN blocks
        self.feature_small = CNNFeatureExtractor(kernel_init=64)
        self.feature_medium = CNNFeatureExtractor(kernel_init=256)
        self.feature_large = CNNFeatureExtractor(kernel_init=4096)

        self.lstm = nn.LSTM(input_size=256*3, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        s = self.feature_small(x)
        m = self.feature_medium(x)
        l = self.feature_large(x)

        min_time = min(s.shape[2], m.shape[2], l.shape[2])
        s = s[:, :, :min_time]
        m = m[:, :, :min_time]
        l = l[:, :, :min_time]

        combined = torch.cat([s,m,l], dim=1)
        combined = combined.transpose(1,2)

        lstm_out, _ = self.lstm(combined)  # [B, T, H]
        out = lstm_out[:, -1, :]  # [B, H]
        return self.fc(out)

class B4(nn.Module):
    '''
    B4 : Multiscale with Mel initialization, no attention
    '''
    def __init__(self, output_dim, hidden_dim=128, num_layers=2, sr=44100):
        super(B4, self).__init__()
        self.name="MultiScale+Mel init"

        self.feature_small = CNNFeatureExtractor(kernel_init=64)
        self.feature_medium = CNNFeatureExtractor(kernel_init=256)
        self.feature_large = CNNFeatureExtractor(kernel_init=4096)

        # Mel filter 초기화
        initialize_mel_filter(self.feature_medium.cnnBlock1.cnn, sr=sr, kernel_size=256, n_filters=40)
        initialize_mel_filter(self.feature_large.cnnBlock1.cnn, sr=sr, kernel_size=4096, n_filters=40)

        self.lstm = nn.LSTM(input_size=256 * 3, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        s = self.feature_small(x)
        m = self.feature_medium(x)
        l = self.feature_large(x)

        min_time = min(s.shape[2], m.shape[2], l.shape[2])
        s, m, l = s[:, :, :min_time], m[:, :, :min_time], l[:, :, :min_time]

        combined = torch.cat([s, m, l], dim=1).transpose(1, 2)

        lstm_out, _ = self.lstm(combined)
        out = self.fc(lstm_out[:, -1, :])
        return out

class P(nn.Module):
    '''
    Proposed Model
    '''
    def __init__(self, output_dim, hidden_dim=128, num_layers=2, sr=44100):
        super(P, self).__init__()
        self.name="Proposed Model"

        self.feature_small = CNNFeatureExtractor(kernel_init=64)
        self.feature_medium = CNNFeatureExtractor(kernel_init=256)
        self.feature_large = CNNFeatureExtractor(kernel_init=4096)

        # Mel filter 초기화
        initialize_mel_filter(self.feature_medium.cnnBlock1.cnn, sr=sr, kernel_size=256, n_filters=40)
        initialize_mel_filter(self.feature_large.cnnBlock1.cnn, sr=sr, kernel_size=4096, n_filters=40)

        self.attention = AttentionModule(256 * 3)
        self.lstm = nn.LSTM(input_size=256 * 3, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        s = self.feature_small(x)
        m = self.feature_medium(x)
        l = self.feature_large(x)

        min_time = min(s.shape[2], m.shape[2], l.shape[2])
        s, m, l = s[:, :, :min_time], m[:, :, :min_time], l[:, :, :min_time]

        combined = torch.cat([s, m, l], dim=1).transpose(1, 2)
        combined = self.attention(combined)

        lstm_out, _ = self.lstm(combined)
        out = self.fc(lstm_out[:, -1, :])
        return out