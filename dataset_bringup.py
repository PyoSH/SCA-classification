import numpy as np

from src.dataset import *
from src.feature_extraction import *

# label_class = {'base': 0, 'idling': 1, 'cutting': 2, 'hardcutting': 3}
label_class = {'idling': 0, 'cutting': 1, 'hardcutting': 2}

sample_rate = 44100
len_frame_time = 50 * 0.001 # 50 ms
len_frame_sample = int(len_frame_time * sample_rate) # sample num = 2205

mp3Path = os.path.join('data','audio',f'TestSet_{1}_1.mp3')
labelPath = os.path.join('data','label',f'test{1}.txt')


if __name__ == '__main__':

    audioData, _ = librosa.load(mp3Path, sr=sample_rate)
    num_frame = len(audioData) // len_frame_sample
    audio_processed = np.zeros((num_frame,len_frame_sample), dtype=np.float32)

    for idx in range(0, num_frame):
        idx_start = idx*len_frame_sample
        idx_end = (idx+1)*len_frame_sample

        singleFrame = audioData[idx_start: idx_end]
        audio_processed[idx, :] = singleFrame[:]

    label_raw = pd.read_csv(labelPath, header=None, sep='\t')
    label_processed = None
    label_processed = labelProcessing(label_raw, num_frame, label_class=label_class, sampleRate=sample_rate, len_frame=len_frame_sample)

    print(audio_processed.shape, label_processed.shape)
