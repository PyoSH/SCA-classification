'''
오디오, 라벨 데이터를 집어넣었을 때 프레임 크기를 맞춰서 나눠주는 코드.
표승현 2024-04-04
'''
import os
from copy import deepcopy

import librosa
import numpy as np
from numba.typed.listobject import list_is

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
dataSet_path = cfg.PATH.TRAIN_PATH # !!!!!TRAIN_PATH or TEST_PATH
# dataSet_path = cfg.PATH.TEST_PATH # !!!!!TRAIN_PATH or TEST_PATH
logger.info("Running dataset_bringup ...")
logger.info(f'DATA path: {dataSet_path}')

label_class = cfg.HYPERPARAMS.LABEL_CLASS
sample_rate = cfg.FEATUREPARAMS.SAMPLING_RATE
len_frame_time = cfg.HYPERPARAMS.LEN_FRAME * 0.001 # 100 ms = 0.1 s
len_frame_sample = int(len_frame_time * sample_rate) # sample num = 2205, 100ms frame = 4410 samples.

dataPath = os.path.join('data', 'train_uw', 'class4') # or 'train' !!!!!!
# dataPath = os.path.join('data', 'test_uw', 'class4') # or 'train' !!!!!!
# dataPath = os.path.join('data', 'samples' ,'all') # or 'train' !!!!!!
# dataPath = os.path.join('data', 'inspection' ,'44100Hz_standby') # or 'train' !!!!!!
audioPathList = os.path.join(dataPath, 'audio')
labelPathList = os.path.join(dataPath, 'label')

if __name__ == '__main__':
    dataSetMat = None
    iterated = False
    isDownsampling = ('8' or '16' or '32') in dataSet_path
    isBalanced = 'balanced' in dataSet_path
    cnt_cutting = 0
    logger.info(f"downsampling : {isDownsampling}")
    logger.info(f"class balancing : {isBalanced}")

    audio_filenames = list_audio_files(audioPathList)
    label_filenames = list_label_files(labelPathList)
    for item in range(len(audio_filenames)):
        tempAudioPath = audio_filenames[item]
        tempLabelPath = label_filenames[item]

        audioData, _ = librosa.load(tempAudioPath, sr=sample_rate)
        if isDownsampling:
            new_sample_rate = 32000
            audioData = librosa.resample(audioData, orig_sr=sample_rate, target_sr=new_sample_rate)
            sample_rate = new_sample_rate
            len_frame_sample = int(len_frame_time * sample_rate)  # sample num = 2205, 100ms frame = 4410 samples.

        tgt_len_msec = 100
        k_tgt = int(sample_rate * tgt_len_msec * 0.001)  # 4410 frames = sample_rate * 100ms (음향 길이) * 0.001 (milli 단위환산)
        k_curr = len_frame_sample  # 22050 frames

        n = len(audioData)
        num_frame_tgt = n // k_tgt
        size_stride = 0 if num_frame_tgt == 1 else (n - k_curr) // (num_frame_tgt - 1)

        num_frame = 1 if size_stride == 0 else (n - k_curr) // size_stride + 1
        audio_processed = np.zeros((num_frame, len_frame_sample), dtype=np.float32)

        for idx in range(num_frame):
            idx_start = idx * size_stride
            idx_end = k_curr + idx * size_stride
            audio_processed[idx, :] = audioData[idx_start:idx_end]

        label_raw = pd.read_csv(tempLabelPath, header=None, sep='\t')
        label_processed = labelProcessing(label_raw, num_frame, label_class=label_class, sampleRate=sample_rate, len_frame=len_frame_sample, stride=size_stride)
        list_label = list(label_processed)
        cnt_cutting += list_label.count(3)

        temp2dMat = np.zeros((num_frame, len_frame_sample +1), dtype=np.float32) #공간 낭비 아깝긴 한데... 생각한건 이거다.
        temp2dMat[:,0:len_frame_sample] = audio_processed
        temp2dMat[:, -1] = label_processed

        logger.info(f'{item:2d}th dataset-{tempLabelPath} processing : num frame {audio_processed.shape[0]:.4f}, num_label {label_processed.shape}, temp dataset {temp2dMat.shape}')

        if (item == 0) or (iterated == False):
            dataSetMat = copy.deepcopy(temp2dMat)
            iterated = True
        else:
            dataSetMat = np.concatenate((dataSetMat, temp2dMat), axis=0)

    logger.info(f'UNBALANCED dataset processed : dataset {dataSetMat.shape}')

    arr_balancedDataSet = []
    if isBalanced:
        np.random.seed(7)

        arr_class = list(dataSetMat[:,-1])

        mat_standby = np.zeros((cnt_cutting, len_frame_sample +1), dtype=np.float32)
        mat_idling = deepcopy(mat_standby)
        mat_cutting = deepcopy(mat_standby)

        list_idx_standby = []
        list_idx_idling = []
        list_idx_cutting = []

        # 1. standby 추가

        for idx in range(0, len(arr_class)):
            item = arr_class[idx]
            if item == 1:
                list_idx_standby.append(idx)
            elif item == 2:
                list_idx_idling.append(idx)
            elif item == 3:
                list_idx_cutting.append(idx)
            else:
                pass

        idx_selected_standby = np.random.choice(list_idx_standby, size=cnt_cutting, replace=False)
        for idx_new, idx in enumerate(idx_selected_standby):
            mat_standby[idx_new, :] = dataSetMat[idx, :]
        arr_balancedDataSet.append(mat_standby)

        # # 2. idling 추가
        idx_selected_idling = np.random.choice(list_idx_idling, size=cnt_cutting, replace=False)
        for idx_new, idx in enumerate(idx_selected_idling):
            mat_idling[idx_new, :] = dataSetMat[idx, :]
        arr_balancedDataSet.append(mat_idling)

        # # 3. cutting 추가
        for idx_new, idx in enumerate(list_idx_cutting):
            mat_cutting[idx_new, :] = dataSetMat[idx, :]
        arr_balancedDataSet.append(mat_cutting)
    balancedDataSet = np.concatenate(arr_balancedDataSet, axis=0)

    np.save(dataSet_path, balancedDataSet)
    logger.info(f'{cfg.HYPERPARAMS.LEN_FRAME}ms Dataset saved in {dataSet_path}')
