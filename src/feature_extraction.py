import os
import librosa
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt


import plotly.express as px
import pandas as pd

class MFCC_params():
    def __init__(self, samplingRate, num_CepstralCs, hop_length, len_window):
        self.sr = samplingRate
        self.n_mfcc = num_CepstralCs
        self.hop_length = hop_length # 2의 배수일 것, hop_length = n_fft /4
        self.len_fft = len_window # n_fft 의미

def save_mp3_to_mfcc(mp3_dir, save_dir, sr, n_mfcc, hop_length, len_fft):
    try:
        # Load audio file using librosa with specified parameters
        audio_data, _ = librosa.load(mp3_dir, sr=sr)

        # Extract MFCC features with specified parameters
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft = len_fft)

        # Normalize MFCC data using StandardScaler
        scaler = StandardScaler()
        mfccs_normalized = scaler.fit_transform(mfccs)

        # Save normalized MFCC data as a NumPy file
        mfcc_normalized_file_path = os.path.join(save_dir, os.path.basename(mp3_dir).replace('.mp3', f'_mfcc_normalized.npy'))
        np.save(mfcc_normalized_file_path, mfccs_normalized)

        print(f"Conversion successful: {mp3_dir} -> {mfcc_normalized_file_path}")
    except Exception as e:
        print(f"Error processing {mp3_dir}: {e}")
'''
유진님 코드에서 가져와 구현해놨고, 이전 ProtoDataset에서 사용했으나
frame_to_mfcc가 seq_len 조절이 가능해서 안씀
표승현 2024-04-04 
'''
def get_mp3_to_mfcc(mp3_dir, sr, n_mfcc, hop_length, len_fft):
    try:
        # Load audio file using librosa with specified parameters
        audio_data, _ = librosa.load(mp3_dir, sr=sr)

        # Extract MFCC features with specified parameters
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft = len_fft)

        # Normalize MFCC data using StandardScaler
        scaler = StandardScaler()
        mfccs_normalized = scaler.fit_transform(mfccs)

        return mfccs_normalized

    except Exception as e:
        print(f"Error processing {mp3_dir}: {e}")

'''
< for real time, data means frame!!>
leng_fft = n_fft
num_cepstralCoefficient = n_mfcc
'''
def get_frame_to_mfcc(data, samplingRate, num_cepstralCoefficient, hop_length, len_fft):
    # 1-1 프레임 대한 정규화 (-1~1) (before mfcc)
    # 16-bit 정수를 부동 소수점으로 변환
    frame_float = data.astype(np.float32) / 32767.0
    frame_norm = None
    # 1-2 프레임 대한 표준화
    if np.max(frame_float) == 0:
        frame_norm = frame_float
    else:
        frame_norm = (frame_float - np.mean(frame_float)) / np.std(frame_float)

    # 2 프레임 정규화 X
    # frame_norm = copy.deepcopy(data.astype(np.float32))

    # 특징 벡터 추출
    featureVector = librosa.feature.mfcc(y=frame_norm, sr=samplingRate, hop_length=hop_length, n_mfcc=num_cepstralCoefficient,
                                 n_fft=len_fft).T # 전치!!!

    # 3 프레임 정규화 (0~1)
    # featureVector = minmax_scale(copy.deepcopy(featureVector), feature_range=(0, 1), axis=0)
    #차원 추가!!!
    featureVector = featureVector[np.newaxis, :, :]
    return featureVector

def get_frame_standardization(data):
    # 16-bit 정수를 부동 소수점으로 변환
    frame_float = data.astype(np.float32) / 32767.0
    frame_norm = None
    # 1-2 프레임 대한 표준화
    if np.max(frame_float) == 0:
        frame_norm = frame_float
    else:
        frame_norm = (frame_float - np.mean(frame_float)) / np.std(frame_float)

    return frame_norm

'''
CRNN 모델의 합성곱 필터 학습이 어떻게 되었는지 확인하는 함수 
어디에 둬야 할지 모르겠으니 여기에 둔다
2025_02_26
'''
def viz_filter_map(model, section, layer_type, idx_ch, sampling_rate):
    cnn_block =  getattr(model.feature_extractor, section)
    layer = getattr(cnn_block, layer_type)

    filters = layer.weight.data.cpu().numpy()  # CPU로 이동
    out_channels, in_channels, ksize = filters.shape

    filters_avg = filters.mean(axis=1)
    # (2) 각 필터마다 FFT 수행 & 진폭 계산 ???
    specs = []
    for i in range(out_channels):
        w_time = filters[i, idx_ch, :]
        # w_time = filters_avg[i, :]

        w_freq = np.fft.rfft(w_time)
        amp = np.abs(w_freq)
        specs.append(amp)

    specs = np.stack(specs, axis=0)  # shape : [out_channels, freq_bins]

    # (3) 주파수 축 계산
    freqs = np.fft.rfftfreq(ksize, d=1.0 / sampling_rate)

    # (4) 필터를 중심 주파수? 등으로 정렬?
    center_freqs = []
    for i in range(out_channels):
        # 필터 i에서 가장 강한 주파수(peak) 위치 찾기
        peak_idx = np.argmax(specs[i])
        center_freq = freqs[peak_idx]
        center_freqs.append(center_freq)

    # 정렬 인덱스 구하기
    sorted_idx = np.argsort(center_freqs)
    specs_sorted = specs[sorted_idx, :]

    # (5) 시각화
    plt.figure(figsize=(8, 6))

    # specs_sorted를 imshow로 표현
    # extent=[x_min, x_max, y_min, y_max]로 실제 주파수 범위를 표시
    plt.imshow(specs_sorted,
               aspect='auto',
               origin='lower',
               extent=[freqs[0], freqs[-1], 0, out_channels])

    plt.colorbar(label="Amplitude")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Filters (sorted by center frequency)")
    plt.title(f"Learned Filters of {section} in Frequency Domain")
    # plt.show()

    ch_str = None
    if idx_ch == -1: ch_str = "last"
    else: ch_str = str(idx_ch)
    prefix = os.path.join('pics', 'CNN', 'weight','ordered')
    file_name_str = f'{section}_{ch_str}.png'
    plt.savefig(os.path.join(prefix, file_name_str))

    return sorted_idx

def viz_feature_map(model, section, layer_type, ch_idx, sorted_idx, data_audio_std, device, working_state):
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()  # 미분 기록 제거하고 저장

        return hook

    cnn_block = getattr(model.feature_extractor, section)
    layer = getattr(cnn_block, layer_type)
    layer.register_forward_hook(get_activation(section))

    input_audio = torch.tensor(data_audio_std[:, :], dtype=torch.float32).unsqueeze(1).to(device)

    model.eval()

    output = model(input_audio)
    _, predicted = torch.max(output.data, 1)
    pred_np = predicted.cpu().numpy()
    print(pred_np)

    feature_maps = activations[section].cpu()

    # 배치 차원 제거 (단일 샘플에 대한 활성화만 시각화)
    feature_maps = feature_maps.squeeze(0)  # shape: [channels, time]
    feature_sorted = feature_maps[sorted_idx, :]

    # 시각화: 각 행은 하나의 채널, 열은 시간축에 따른 활성화 값
    plt.figure(figsize=(10, 8))
    plt.imshow(feature_sorted,
               aspect='auto',
               origin='lower',
               interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Activation')
    plt.xlabel('Reduced Time Index')
    plt.ylabel('Channels (sorted from weights)')
    plt.title(f'Feature Maps from {section} Layer, {ch_idx}th channel')
    # plt.show()

    ch_str = None
    if layer_type == "cnn":
        prefix = os.path.join('pics', 'CNN', 'feature', 'ordered', working_state)
    elif layer_type == "pool":
        prefix = os.path.join('pics', 'pooling','ordered',working_state)

    if ch_idx == -1: ch_str = "last"
    else: ch_str = str(ch_idx)

    file_name_str = f'{section}_{ch_str}.png'
    plt.savefig(os.path.join(prefix, file_name_str))

def extract_feature_embeddings(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    indices = []

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            x = None

            if model.name == 'C-MultiScale':
                ## AMS-CNN 최종 출력 임베딩
                # Forward pass through the model
                s = model.cnnBlock_1_small(inputs)
                m = model.cnnBlock_1_medium(inputs)
                l = model.cnnBlock_1_large(inputs)

                x = m

                # # Make sure the time dimension is consistent
                # min_time = min(s.shape[2], m.shape[2], l.shape[2])
                # s = s[:, :, :min_time]
                # m = m[:, :, :min_time]
                # l = l[:, :, :min_time]

                # # Concatenate the outputs from all the branches
                # combined = torch.cat([s, m, l], dim=1)
                # combined = combined.transpose(1, 2)  # [batch_size, time_steps, channels]
                # x = model.attention(combined)
            elif model.name == 'C_test':
                ## CNN 최종 출력 임베딩
                x = model.feature_extractor(inputs)
            elif model.name == 'no feature, before FC':
                ## CNN → LSTM → 최종 출력이 아닌, LSTM의 임베딩 출력까지 사용
                x = x.transpose(1, 2)  # [batch, time, channels]
                lstm_out, _ = model.lstm(x)

                # 마지막 타임스텝의 출력 벡터 사용
                embedding = lstm_out[:, -1, :].cpu().numpy()

            # 시각화를 위해 (batch, channels, time)를 (batch, -1)로 평탄화
            # embedding = x.view(x.size(0), -1).cpu().numpy()
            embedding = x.reshape(x.size(0), -1).cpu().numpy()

            features.append(embedding)
            labels.append(targets.cpu().numpy())
            indices.extend([idx] * embedding.shape[0])

    return np.vstack(features), np.hstack(labels), np.array(indices)

def plot_tsne(features, labels, class_names=None, perplexity=30, title='t-SNE Visualization'):
    """
    features : (N, D) numpy array of high-dim embeddings
    labels   : (N,) array of class labels (0, 1, 2, ...)
    """
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(12, 10))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(reduced[idx, 0], reduced[idx, 1],
                    label=class_names[int(label)] if class_names else f'Class {label}',
                    alpha=0.6)

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlabel("TSNE 1")
    plt.ylabel("TSNE 2")
    plt.tight_layout()
    plt.show()

def plot_tsne_3d(features, labels, class_names=None, perplexity=30, title='t-SNE 3D Visualization'):
    """
    features : (N, D) high-dim embeddings
    labels   : (N,) class labels (0, 1, 2, ...)
    """
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    reduced = tsne.fit_transform(features)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for label in np.unique(labels):
        idx = labels == label
        ax.scatter(reduced[idx, 0], reduced[idx, 1], reduced[idx, 2],
                   label=class_names[int(label)] if class_names else f'Class {int(label)}',
                   alpha=0.6, s=100)

    ax.set_title(title)
    ax.set_xlabel("TSNE 1")
    ax.set_ylabel("TSNE 2")
    ax.set_zlabel("TSNE 3")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_tsne_interactive(features, labels, indices, class_names=None, perplexity=30, title='t-SNE Interactive'):
    # 차원 축소
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced = tsne.fit_transform(features)

    # 데이터프레임 구성
    df = pd.DataFrame({
        "TSNE1": reduced[:, 0],
        "TSNE2": reduced[:, 1],
        "Label": [class_names[int(l)] if class_names else str(int(l)) for l in labels],
        "Index": indices
    })

    # Plotly 시각화
    fig = px.scatter(
        df,
        x="TSNE1",
        y="TSNE2",
        color="Label",
        hover_data=["Index"],
        title=title,
        width=1000,
        height=800
    )

    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.show()

def compute_gaussian_similarity_matrix(features, sigma=1.0):
    """
    features: (N, D) 형태의 numpy array
    sigma: 가우시안 커널 폭 (값이 작을수록 민감해짐)
    """
    N = features.shape[0]
    sim_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            diff = features[i] - features[j]
            dist_sq = np.dot(diff, diff)
            sim_matrix[i, j] = np.exp(-dist_sq / (2 * sigma ** 2))

    return sim_matrix
def plot_similarity_matrix(sim_matrix, labels=None, title="Gaussian Similarity Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap='viridis', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    plt.show()
def plot_similarity_matrix_with_class_colors(sim_matrix, labels, class_names=None, title="Gaussian Similarity Matrix"):
    plt.figure(figsize=(10, 8))

    # 라벨 → 클래스 이름 (선택)
    if class_names:
        label_names = [class_names[int(l)] for l in labels]
    else:
        label_names = [str(int(l)) for l in labels]

    # 클래스별 색 지정
    unique_classes = sorted(set(label_names))
    palette = sns.color_palette("husl", len(unique_classes))
    class_color_map = {cls: palette[i] for i, cls in enumerate(unique_classes)}
    row_colors = [class_color_map[label] for label in label_names]

    # DataFrame으로 변환
    df = pd.DataFrame(sim_matrix)

    sns.clustermap(df, row_colors=row_colors, col_colors=row_colors, cmap="viridis", xticklabels=False,
                   yticklabels=False)
    plt.suptitle(title)
    plt.show()