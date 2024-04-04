'''
오디오, 라벨 데이터를 집어넣었을 때 프레임 크기를 맞춰서 나눠주는 코드.
표승현 2024-04-04
'''

from src.dataset import *
from src.feature_extraction import *

# label_class = {'base': 0, 'idling': 1, 'cutting': 2, 'hardcutting': 3}
label_class = {'idling': 0, 'cutting': 1, 'hardcutting': 2}

sample_rate = 44100
len_frame_time = 100 * 0.001 # 50 ms
len_frame_sample = int(len_frame_time * sample_rate) # sample num = 2205

dataPath = 'data'

if __name__ == '__main__':


    dataSetMat = None
    iterated = False

    for item in range(1,10):

        mp3Path = os.path.join(dataPath, 'audio', f'TestSet_{item}_1.mp3')
        labelPath = os.path.join(dataPath, 'label', f'test{item}.txt')

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

        temp2dMat = np.zeros((num_frame, len_frame_sample +1), dtype=np.float32) #공간 낭비 아깝긴 한데... 생각한건 이거다.
        temp2dMat[:,0:len_frame_sample] = audio_processed
        temp2dMat[:, -1] = label_processed

        print(f'{item:2d}th dataset processing : num frame {audio_processed.shape[0]:.4f}, num_label {label_processed.shape}, temp dataset {temp2dMat.shape}')

        if (item == 1) or (iterated == False):
            dataSetMat = copy.deepcopy(temp2dMat)
            iterated = True
        else:
            dataSetMat = np.concatenate((dataSetMat, temp2dMat), axis=0)

    print(f'dataset processed : dataset {dataSetMat.shape}')
    dataSetType = int(len_frame_time*1000)
    dataSet_path = os.path.join(dataPath,f'set_{dataSetType}ms',f'data_{dataSetType}ms')
    np.save(dataSet_path, dataSetMat)
    print(f'{dataSetType}ms Dataset saved in {dataSet_path}')
