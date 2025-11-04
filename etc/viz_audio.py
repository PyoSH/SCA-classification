import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. 오디오 파일 불러오기
# 'your_audio_file.wav'를 실제 파일 경로로 변경하세요.
file_path = '../data/inspection/idling/audio/test2-idling200.wav'

y, sr = librosa.load(file_path, sr=None)

# 2. 멜-스펙트로그램 계산 (Audacity 설정에 맞게 파라미터 수정)
S = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=1024,
    hop_length=256,
    n_mels=256,       # 멜 밴드 개수를 늘려 세밀함 추가
    fmax=20000        # 최대 주파수를 20000Hz로 명시 (Audacity 설정과 동일하게)
)

# 3. 파워를 데시벨(dB)로 변환
S_DB = librosa.power_to_db(S, ref=np.max)

# 4. 그래프 그리기
# 두 개의 서브플롯(하나는 음압, 하나는 멜-스펙트로그램)을 생성합니다.
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(16, 9))

# 첫 번째 그래프: 음압(Waveform)
# y축은 진폭(Amplitude), x축은 시간(Time)을 나타냅니다.

librosa.display.waveshow(y, sr=sr, ax=ax[0])
ax[0].set_ylim([-1, 1]) # Y축 범위를 -1에서 1로 고정
ax[0].set_title('Waveform (Sound Pressure)', fontsize=15)
ax[0].set_ylabel('Amplitude')
ax[0].grid(True)

# 두 번째 그래프: 멜-스펙트로그램
# y축은 주파수(Hz, Mel-scale), x축은 시간(Time)을 나타냅니다.
# 컬러바는 해당 시간-주파수의 에너지(dB)를 의미합니다.
# img = librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax[1])

img = librosa.display.specshow(
    S_DB,
    sr=sr,
    x_axis='time',
    y_axis='mel',
    fmax=20000,       # y축의 최대 주파수도 동일하게 설정
    cmap='magma',     # 색상 맵 변경
    ax=ax[1]
)

ax[1].set_title('Mel-Spectrogram', fontsize=15)
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Frequency (Hz)')

# 컬러바 추가
fig.colorbar(img, ax=ax[1], format='%+2.0f dB', label='Intensity (dB)')

# 레이아웃 최적화 및 그래프 출력
plt.tight_layout()
plt.show()