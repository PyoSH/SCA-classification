from pyexpat import features
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from src.dataset import RawWaveformDataset
from src.model_definition import *
import torch
from config import cfg, update_config
import argparse
from loguru import logger

def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # CNN 최종 출력 임베딩
            x = model.feature_extractor(inputs)

            # 시각화를 위해 (batch, channels, time)를 (batch, -1)로 평탄화
            cnn_embedding = x.view(x.size(0), -1).cpu().numpy()

            features.append(cnn_embedding)
            labels.append(targets.cpu().numpy())

    return np.vstack(features), np.hstack(labels)

def plot_tsne_with_index(features, labels, sample_indices, class_names=None, perplexity=30, title='t-SNE Visualization'):
    """
    features       : (N, D) numpy array of high-dim embeddings
    labels         : (N,) array of class labels (0, 1, 2, ...)
    sample_indices : (N,) array of index 번호 (같은 샘플 내에서 나온 벡터 구분용)
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(12, 10))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=class_names[label] if class_names else f'Class {label}', alpha=0.6)

        # 인덱스 번호 텍스트 표시
        for i in np.where(idx)[0]:
            plt.text(reduced[i, 0], reduced[i, 1], str(sample_indices[i]), fontsize=8, alpha=0.7)

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlabel("TSNE 1")
    plt.ylabel("TSNE 2")
    plt.tight_layout()
    plt.show()

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

    model.load_state_dict(torch.load(cfg.PATH.MODEL_PATH, weights_only=True))

    # data load
    data_raw = np.load(cfg.PATH.TEST_PATH)
    data_audio = data_raw[:, 0:-1]
    data_label = data_raw[:, -1]
    data_audio_std = np.zeros_like(data_audio)

    # audio standardization to mean 0, variance 1
    for i, row in enumerate(data_audio):
        row_std = (row - np.mean(row)) / np.std(row)
        data_audio_std[i, :] = row

    rawDataset = RawWaveformDataset(data_audio_std, data_label)
    data_loader = DataLoader(rawDataset, batch_size=1, shuffle=False)

    features, labels = extract_features(model=model, dataloader=data_loader, device=device)
    
    plot_tsne_with_index(features, data_label, )