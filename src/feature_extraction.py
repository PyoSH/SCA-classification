import os
import librosa
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

'''
CRNN 모델의 합성곱 필터 학습이 어떻게 되었는지 확인하는 함수 
어디에 둬야 할지 모르겠으니 여기에 둔다
2025_02_26
'''
def viz_filter_map(model, section, idx_ch, sampling_rate):
    layer = getattr(model, section)
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

def viz_feature_map(model, section, ch_idx, sorted_idx, data_audio_std, device, working_state):
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()  # 미분 기록 제거하고 저장

        return hook

    layer = getattr(model, section)
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
    if "cnn" in section:
        prefix = os.path.join('pics', 'CNN', 'feature', 'ordered', working_state)
    elif "pool" in section:
        prefix = os.path.join('pics', 'pooling','ordered',working_state)

    if ch_idx == -1: ch_str = "last"
    else: ch_str = str(ch_idx)

    file_name_str = f'{section}_{ch_str}.png'
    plt.savefig(os.path.join(prefix, file_name_str))

