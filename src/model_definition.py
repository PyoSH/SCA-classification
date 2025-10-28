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

class B1(nn.Module):
    '''
    MFCC+LSTM
    '''
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(B1, self).__init__()
        self.name = "B1"

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # torchaudio.transforms.MFCC 초기화
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=44100,
            n_mfcc=40,
            melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 40}  # n_mels와 n_freqs 조정
        )

    def forward(self, x):
        # 1. x의 shape을 (batch_size * n_frames, 4410)로 변경하여 MFCC 계산을 위한 형태로 변환
        x_reshaped = x.view(-1, 4410)  # (batch_size * n_frames, 4410)

        # 2. MFCC 계산
        mfcc_features = self.mfcc_transform(x_reshaped)  # MFCC 계산

        # 3. MFCC는 (batch_size * n_frames, n_mfcc, n_frames) 형태로 반환되므로, 다시 차원 수정
        mfcc_features = mfcc_features.transpose(1,2)

        # LSTM 연산
        out, _ = self.lstm(mfcc_features)
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
    def __init__(self, output_dim, input_dim=256, hidden_dim=128, num_layers=2):
        super(B2_small, self).__init__()
        self.name = "B2-small"

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

    def __init__(self, output_dim, input_dim=256, hidden_dim=128, num_layers=2):
        super(B2_middle, self).__init__()
        self.name = "B2-middle"

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

    def __init__(self, output_dim, input_dim=256, hidden_dim=128, num_layers=2):
        super(B2_large, self).__init__()
        self.name = "B2-large"

        self.feature_extractor = CNNFeatureExtractor(kernel_init=2048)
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
        self.name="B3"

        # 1. multi scale CNN blocks
        self.feature_small = CNNFeatureExtractor(kernel_init=64)
        self.feature_medium = CNNFeatureExtractor(kernel_init=256)
        self.feature_large = CNNFeatureExtractor(kernel_init=2048)

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
        self.name="B4"

        self.feature_small = CNNFeatureExtractor(kernel_init=64)
        self.feature_medium = CNNFeatureExtractor(kernel_init=256)
        self.feature_large = CNNFeatureExtractor(kernel_init=2048)

        # Mel filter 초기화
        initialize_mel_filter(self.feature_medium.cnnBlock1.cnn, sr=sr, kernel_size=256, n_filters=40)
        initialize_mel_filter(self.feature_large.cnnBlock1.cnn, sr=sr, kernel_size=2048, n_filters=40)

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
        self.name="P"

        self.feature_small = CNNFeatureExtractor(kernel_init=64)
        self.feature_medium = CNNFeatureExtractor(kernel_init=256)
        self.feature_large = CNNFeatureExtractor(kernel_init=2048)

        # Mel filter 초기화
        initialize_mel_filter(self.feature_medium.cnnBlock1.cnn, sr=sr, kernel_size=256, n_filters=40)
        initialize_mel_filter(self.feature_large.cnnBlock1.cnn, sr=sr, kernel_size=2048, n_filters=40)

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

class P2(nn.Module):
    '''
    Proposed Model
    '''
    def __init__(self, output_dim, hidden_dim=128, num_layers=2, sr=44100):
        super(P2, self).__init__()
        self.name="P"

        self.feature_small = CNNFeatureExtractor(kernel_init=64)
        self.feature_medium = CNNFeatureExtractor(kernel_init=256)

        # Mel filter 초기화
        initialize_mel_filter(self.feature_medium.cnnBlock1.cnn, sr=sr, kernel_size=256, n_filters=40)

        self.attention = AttentionModule(256 * 2)
        self.lstm = nn.LSTM(input_size=256 * 2, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        s = self.feature_small(x)
        m = self.feature_medium(x)

        min_time = min(s.shape[2], m.shape[2])
        s, m = s[:, :, :min_time], m[:, :, :min_time]

        combined = torch.cat([s, m], dim=1).transpose(1, 2)
        combined = self.attention(combined)

        lstm_out, _ = self.lstm(combined)
        out = self.fc(lstm_out[:, -1, :])
        return out

class SVM(nn.Module):
    '''
    MFCC+LSTM
    '''
    def __init__(self, output_dim):
        super(SVM, self).__init__()
        self.name = "SVM"

        # SVM
        self.feature_dim = 40 *2
        self.fc = nn.Linear(self.feature_dim, output_dim)

        # torchaudio.transforms.MFCC 초기화
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=44100,
            n_mfcc=40,
            melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 40}  # n_mels와 n_freqs 조정
        )

    def forward(self, x):
        # 1. x의 shape을 (batch_size * n_frames, 4410)로 변경하여 MFCC 계산을 위한 형태로 변환
        x_reshaped = x.view(-1, 4410)  # (batch_size * n_frames, 4410)

        # 2. MFCC 계산
        mfcc_features = self.mfcc_transform(x_reshaped)  # MFCC 계산

        ### 3. statistical pooling 수행
        # 3.1 시간 축(dim=2)에 대해 평균(Mean) 계산
        mean_features = torch.mean(mfcc_features, dim=2)
        # mean_features.shape: torch.Size([32, 40])

        # 3.2 시간 축(dim=2)에 대해 표준편차(StdDev) 계산
        std_features = torch.std(mfcc_features, dim=2)
        # std_features.shape: torch.Size([32, 40])

        # 3.3 두 특징을 특징 축(dim=1) 기준으로 결합
        # (32, 40)과 (32, 40)을 합쳐 (32, 80)으로 만듦
        pooled_vector = torch.cat((mean_features, std_features), dim=1)

        # 4. SVM 연산
        out = self.fc(pooled_vector)

        return out

class AST(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(AST, self).__init__()
        self.name = "AST"

        self.svm = nn.LinearSVC(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # torchaudio.transforms.MFCC 초기화
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=44100,
            n_mfcc=40,
            melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 40}  # n_mels와 n_freqs 조정
        )

    def forward(self, x):
        # 1. x의 shape을 (batch_size * n_frames, 4410)로 변경하여 MFCC 계산을 위한 형태로 변환
        x_reshaped = x.view(-1, 4410)  # (batch_size * n_frames, 4410)

        # 2. MFCC 계산
        mfcc_features = self.mfcc_transform(x_reshaped)  # MFCC 계산

        # 3. MFCC는 (batch_size * n_frames, n_mfcc, n_frames) 형태로 반환되므로, 다시 차원 수정
        mfcc_features = mfcc_features.transpose(1,2)

        # LSTM 연산
        out, _ = self.lstm(mfcc_features)
        out = self.fc(out[:, -1, :])
        return out

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torchaudio.transforms as T # 1. torchaudio.transforms 임포트

class AST(nn.Module):
    """
    Hugging Face의 사전 학습된 AST 모델을 로드하고
    전처리기(FeatureExtractor)를 포함하는 End-to-End 모듈.

    입력: (Batch, Audio_Length)의 원본 오디오 텐서
    출력: (Batch, Num_Classes)의 로짓(Logits)
    """

    def __init__(self, output_dim=4, device=torch.device("cpu")):
        super(AST, self).__init__()
        self.name = "AST"

        model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

        self.model = AutoModelForAudioClassification.from_pretrained(
            model_name,
            num_labels=output_dim,
            ignore_mismatched_sizes=True
        ).to(device)

        self.device = device

        # --- [수정 1/2] ---
        # 44.1kHz (원본) -> 16kHz (AST 요구) 리샘플러를 초기화합니다.
        self.resampler = T.Resample(
            orig_freq=44100,
            new_freq=16000
        ).to(device)
        # --- [수정 끝] ---

    def forward(self, x):
        # x: (Batch_Size, Audio_Length) @ 44100Hz

        # --- [수정 2/2] ---
        # 1. 리샘플링 (GPU에서 바로 수행)
        # (Batch, 44.1k_Length) -> (Batch, 16k_Length)
        x_resampled = self.resampler(x)

        # 2. 전처리 (16kHz로 리샘플링된 오디오 사용)
        raw_audio_list = [audio.cpu().numpy() for audio in x_resampled]

        inputs = self.feature_extractor(
            raw_audio_list,
            sampling_rate=16000,  # 3. 이제 오디오가 16kHz라고 '확인'시켜 줍니다.
            return_tensors="pt"
        )
        # --- [수정 끝] ---

        # 4. 전처리된 입력을 모델 디바이스로 이동
        inputs = inputs.to(self.device)

        # 5. 모델 추론
        outputs = self.model(**inputs)

        # 6. 로짓 반환
        return outputs.logits