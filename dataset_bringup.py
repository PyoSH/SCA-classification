'''
오디오, 라벨 데이터를 집어넣었을 때 프레임 크기를 맞춰서 나눠주는 코드.
표승현 2024-04-04
'''

from src.dataset import *
from src.feature_extraction import *
from config import cfg, update_config
import argparse
from loguru import logger

parser = argparse.ArgumentParser(description='Running audio classification')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)

args = parser.parse_args()
update_config(cfg, args)

logger.info("Running dataset_bringup ...")
logger.info(f'DATA path: {cfg.PATH.TRAIN_PATH}')

label_class = cfg.HYPERPARAMS.LABEL_CLASS
sample_rate = cfg.FEATUREPARAMS.SAMPLING_RATE
len_frame_time = cfg.HYPERPARAMS.LEN_FRAME * 0.001 # 50 ms
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

        logger.info(f'{item:2d}th dataset processing : num frame {audio_processed.shape[0]:.4f}, num_label {label_processed.shape}, temp dataset {temp2dMat.shape}')

        if (item == 1) or (iterated == False):
            dataSetMat = copy.deepcopy(temp2dMat)
            iterated = True
        else:
            dataSetMat = np.concatenate((dataSetMat, temp2dMat), axis=0)

    logger.info(f'dataset processed : dataset {dataSetMat.shape}')
    dataSet_path = cfg.PATH.TRAIN_PATH
    np.save(dataSet_path, dataSetMat)
    logger.info(f'{cfg.HYPERPARAMS.LEN_FRAME}ms Dataset saved in {dataSet_path}')
