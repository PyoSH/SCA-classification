import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
import pyaudio
import copy

samplingRate = 44100
hop_length = 512
num_CC = 40
len_fft = 2048

# 전체 오디오 데이터에서 MFCC 추출
filePath = os.path.join('data','audio','Testset_8_1.mp3')
audio_data, _ = librosa.load(filePath, sr=samplingRate)
audio_data = np.array(audio_data)

# 정규화 X
mfccs = librosa.feature.mfcc(y=copy.deepcopy(audio_data).astype(float), sr=samplingRate, hop_length=hop_length, n_mfcc=num_CC, n_fft=len_fft)

# 추출 후 정규화 (0~1 범위로)
norm_after_mfccs = minmax_scale(copy.deepcopy(mfccs), feature_range=(np.min(mfccs), np.max(mfccs)), axis=0)

# 추출 전 정규화 (0~1 범위로)
# 16-bit 정수를 부동 소수점으로 변환
frame_float = copy.deepcopy(audio_data).astype(np.float32) / 32767.0
# normalization 추가!!
frame_norm = None
if np.max(frame_float) == 0:
    frame_norm = frame_float
else:
    frame_norm = (frame_float - np.mean(frame_float)) / np.std(frame_float)
norm_before_mfccs = librosa.feature.mfcc(y=frame_norm, sr=samplingRate, hop_length=hop_length, n_mfcc=num_CC,
                             n_fft=len_fft)

# 히트맵 플로팅
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='s', cmap='viridis_r', hop_length=512)
plt.colorbar(format="%+1.0f dB")
plt.title('MFCC Heatmap')
plt.xlabel('Time (s)')
plt.ylabel('MFCC Coefficients')
plt.show()

plt.figure(figsize=(10, 4))
librosa.display.specshow(norm_after_mfccs, x_axis='s', cmap='viridis_r', hop_length=512)
plt.colorbar(format="%+1.0f dB")
plt.title('Normalized MFCC Heatmap - after')
plt.xlabel('Time (s)')
plt.ylabel('MFCC Coefficients')
plt.show()

plt.figure(figsize=(10, 4))
librosa.display.specshow(norm_before_mfccs, x_axis='s', cmap='viridis_r', hop_length=512)
plt.colorbar(format="%+1.0f dB")
plt.title('Normalized MFCC Heatmap - before')
plt.xlabel('Time (s)')
plt.ylabel('MFCC Coefficients')
plt.show()