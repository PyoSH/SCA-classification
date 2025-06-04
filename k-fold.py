import os.path

from src.model_definition import *
from src.train_utils import *
from src.dataset import *

from sklearn.model_selection import KFold
from config import cfg, update_config
import argparse
from loguru import logger
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # 연산 결정적 수행
        torch.backends.cudnn.benchmark = False     # 성능 대신 재현성

# 디바이스 설정: Apple Silicon의 MPS, CUDA, 또는 CPU
device = (
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)

if device.type == "cuda":
    logger.info(f'GPU device found: {torch.cuda.get_device_name(0)}')
    
logger.info(f'selected device: {device}')

parser = argparse.ArgumentParser(description='Running audio classification')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)

args = parser.parse_args()
update_config(cfg, args)

if not os.path.exists(cfg.PATH.TRAIN_PATH):
    logger.info("There is no dataset to learn :( ")
else:
    logger.info("Dataset found :) ")

logger.info("Running train code ...")
logger.info(f'DATA path: {cfg.PATH.TRAIN_PATH}')


# ✅ loguru 설정 (로그 파일 저장 가능)
logger.add("training.log", format="{time} {level} {message}", level="INFO", rotation="1 MB")

# ✅ 하이퍼파라미터 설정
num_epochs = 50
batch_size = 32
learning_rate = 0.001  # 기존보다 낮춘 학습률0.0001
k_folds = 2  # K-Fold 개수

if __name__ == '__main__':
    set_seed(42)

    # ✅ 데이터셋 로드
    datas_law = np.load(cfg.PATH.TRAIN_PATH)
    data_audio = datas_law[:, 0:-1]  # 오디오 피처 그대로임, 이거 MFCC로 특징 벡터 뽑아야 함
    data_label = datas_law[:, -1]

    data_audio_std = np.zeros_like(data_audio)
    # ✅ 오디오 표준화 (mean=0, std=1)
    for i, row in enumerate(data_audio):
        row_std = (row - np.mean(row)) / np.std(row)
        data_audio_std[i, :] = row_std
    data_audio_input = data_audio_std

    # ✅ K-Fold 설정
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # ✅ K-Fold Cross Validation 시작
    all_train_losses = []
    all_test_losses = []
    all_test_accs = []
    model = None

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data_audio_input, data_label)):
        logger.info(f"Fold [{fold+1}/{k_folds}] 학습 시작...")


        X_train, y_train = data_audio_input[train_idx], data_label[train_idx]
        X_val, y_val = data_audio_input[val_idx], data_label[val_idx]

        train_dataset = RawWaveformDataset(X_train, y_train)
        valid_dataset = RawWaveformDataset(X_val, y_val)

        # ✅ 데이터로더 생성
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        # ✅ 모델 초기화 (각 Fold마다 새로 학습해야 함)
        # model = C_test(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
        # model = C_MultiScale_1st(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
        # model = C_MultiScale_2nd(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
        # model = C_MultiScale_3rd(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
        model = C_MultiScale_4th(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_loss, test_loss, test_acc = train_model_device(
            model=model, train_loader=train_loader, test_loader=test_loader, criterion=criterion,
            optimizer=optimizer, num_epochs=cfg.HYPERPARAMS.NUM_EPOCHS, device=device)

        all_train_losses.append(train_loss)
        all_test_losses.append(test_loss)
        all_test_accs.append(test_acc)

        # print(eval_metrics_device(model, test_loader, cfg.HYPERPARAMS.LABEL_CLASS, device))

        # ✅ Fold별 모델 저장 (옵션)
        model_path = f"{cfg.PATH.MODEL_PATH.removesuffix('.pth')}_fold{fold+1}.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Fold [{fold+1}] 모델 저장 완료: {model_path}")

    # ✅ 최종 평균 성능 계산
    plot_kfold_curves(model.name, all_train_losses, all_test_losses, all_test_accs)
    logger.info(f"5-Fold Cross Validation 완료!")