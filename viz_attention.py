from src.model_definition import *
from config import cfg, update_config
from loguru import logger
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

device = None
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f'GPU device found: {torch.cuda.get_device_name(0)}')
else:
    device = torch.device("cpu")
# device = torch.device("cpu")
logger.info(f'selected device: {device}')


model = None
# config에서 model type을 가져오기
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

for input_waveform in data_audio_std:
    x = input_waveform.unsqueeze(0).to(device)  # (1, 1, T)

    with torch.no_grad():
        s = model.feature_small(x)
        m = model.feature_medium(x)
        l = model.feature_large(x)

        min_time = min(s.shape[2], m.shape[2], l.shape[2])
        s, m, l = s[:, :, :min_time], m[:, :, :min_time], l[:, :, :min_time]

        combined = torch.cat([s, m, l], dim=1).transpose(1, 2)  # (B, T, C)
        attended, temp_attn = model.attention(combined, return_weights=True)

    # 시각화
    plt.plot(temp_attn.squeeze().cpu().numpy())
    plt.title("Temporal Attention")
    plt.xlabel("Time Step")
    plt.ylabel("Average Attention Weight")
    plt.show()
