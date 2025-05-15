import numpy as np
from torch.utils.data import DataLoader
from src.dataset import RawWaveformDataset
from src.model_definition import *
from src.feature_extraction import *
import torch
from config import cfg, update_config
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
    model = None
    # (1) 학습된 모델의 CNN layer 에서 가중치 추출
    if cfg.HYPERPARAMS.MODELTYPE == 'C-LSTM':
        model = CLSTM_3(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    elif cfg.HYPERPARAMS.MODELTYPE == 'C-RNN':
        model = CRNN_3(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    elif cfg.HYPERPARAMS.MODELTYPE == 'C-test':
        model = C_test(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    elif cfg.HYPERPARAMS.MODELTYPE == 'C-MultiScale':
        model = C_MultiScale_1st(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)


    model.load_state_dict(torch.load(cfg.PATH.MODEL_PATH, weights_only=True))

    # data load
    data_raw = np.load(cfg.PATH.SAMPLES_PATH)
    data_audio = data_raw[:, 0:-1]
    data_label = data_raw[:, -1]
    data_audio_std = np.zeros_like(data_audio)

    # audio standardization to mean 0, variance 1
    for i, row in enumerate(data_audio):
        row_std = (row - np.mean(row)) / np.std(row)
        data_audio_std[i, :] = row

    rawDataset = RawWaveformDataset(data_audio_std, data_label)
    data_loader = DataLoader(rawDataset, batch_size=1, shuffle=False)

    features, labels, indices = extract_feature_embeddings(model=model, dataloader=data_loader, device=device)
    
    # plot_tsne(features, data_label, class_names=cfg.HYPERPARAMS.LABEL_CLASS)
    plot_tsne_3d(features, data_label, class_names=cfg.HYPERPARAMS.LABEL_CLASS)
    # plot_tsne_interactive(features, labels, indices, class_names=cfg.HYPERPARAMS.LABEL_CLASS)