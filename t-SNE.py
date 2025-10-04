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


    # model.load_state_dict(torch.load(cfg.PATH.MODEL_PATH, weights_only=True))
    model.load_state_dict(torch.load(cfg.PATH.MODEL_PATH, map_location='cpu'))
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
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

    plot_tsne(features, data_label, class_names=cfg.HYPERPARAMS.LABEL_CLASS)
    # plot_tsne_3d(features, data_label, class_names=cfg.HYPERPARAMS.LABEL_CLASS)
    # plot_tsne_interactive(features, labels, indices, class_names=cfg.HYPERPARAMS.LABEL_CLASS)