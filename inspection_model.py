'''
모델, 오디오, 라벨 데이터를 넣어서 가져온 모델이 입력한 데이터에 대해 예측하는 코드
표승현 2024-04-04
'''
from copy import deepcopy

from src.model_definition import *
from src.feature_extraction import *
from config import cfg, update_config
import numpy as np
import argparse
from loguru import logger

# 디바이스 설정: Apple Silicon의 MPS, CUDA, 또는 CPU
device = None
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f'GPU device found: {torch.cuda.get_device_name(0)}')
else:
    device = torch.device("cpu")
logger.info(f'selected device: {device}')

parser = argparse.ArgumentParser(description='Running audio classification')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)

args = parser.parse_args()
update_config(cfg, args)

logger.info("Running model_inspection ...")
logger.info(f'MODEL path: {cfg.PATH.MODEL_PATH}')


if __name__ == '__main__':

    def extract_label(path):
        return path.split("/")[-1].rsplit("_", 1)[-1].split(".")[0]

    working_state = extract_label(cfg.PATH.INSPECTION_PATH)

    model = None
    model_type = cfg.HYPERPARAMS.MODELTYPE

    # 모델 선택
    if model_type == 'B1':
        model = B1(input_dim=40, hidden_dim=cfg.HYPERPARAMS.HIDDEN_SIZE,
                   num_layers=cfg.HYPERPARAMS.NUM_LAYERS,
                   output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    elif model_type == 'B2-small':
        model = B2_small(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    elif model_type == 'B2-middle':
        model = B2_middle(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    elif model_type == 'B2-large':
        model = B2_large(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    elif model_type == 'B3':
        model = B3(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    elif model_type == 'B4':
        model = B4(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    elif model_type == 'P':
        model = P(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
        # model = P2(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    else:
        raise ValueError(f"Model type {model_type} is not recognized.")

    model.load_state_dict(torch.load(cfg.PATH.MODEL_PATH, weights_only=True))
    model.eval()

    data_raw = np.load(cfg.PATH.INSPECTION_PATH)
    data_audio = data_raw[:, 0:-1]
    data_label = data_raw[:, -1]
    data_audio_std = deepcopy(data_audio)

    for i, row in enumerate(data_audio):
        row_std = (row - np.mean(row)) / np.std(row)
        data_audio_std[i, :] = row_std

    if cfg.HYPERPARAMS.MODELTYPE == 'C-test':
        for i in [1,2,3]:
            block_name = f"cnnBlock{i}"

            ch_arr = []
            if i == 1:
                ch_arr=[0]
            elif i == 2:
                ch_arr = [0, 31, -1]
            elif i == 3:
                ch_arr = [0, 63, -1]

            for j in ch_arr:
                ch_idx = int(j)
                # 모델의 cnn1 레이어 가중치 추출 (shape: [out_channels, in_channels, kernel_size])
                sorted_idx = viz_filter_map(model, block_name, "cnn",ch_idx,44100)

                viz_feature_map(model, block_name, "pool", ch_idx, sorted_idx, data_audio_std, device, working_state)
                viz_feature_map(model, block_name, "cnn", ch_idx, sorted_idx, data_audio_std, device, working_state)
    elif model_type == 'P':
        kernel_arr = ["small", "medium", "large"]
        # kernel_arr = ["small", "medium"]
        for kernel in kernel_arr:
            # block_name = f"cnnBlock_1_{kernel}"
            block_name = f"feature_{kernel}"

            ch_idx = 0
            sorted_idx = viz_filter_map(model, block_name, "cnn", ch_idx, 44100)



            # viz_feature_map(model, block_name, "pool", ch_idx, sorted_idx, data_audio_std, device, working_state)
            # viz_feature_map(model, block_name, "cnn", ch_idx, sorted_idx, data_audio_std, device, working_state)