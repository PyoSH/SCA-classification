from src.model_definition import *
from src.train_utils import *
from src.dataset import *

from sklearn.model_selection import KFold
from config import cfg, update_config
import argparse
from loguru import logger

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
learning_rate = 0.0001  # 기존보다 낮춘 학습률
k_folds = 5  # K-Fold 개수

if __name__ == '__main__':
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
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(data_audio_input))):
        logger.info(f"\n📌 Fold [{fold+1}/{k_folds}] 학습 시작...")

        X_train, y_train = data_audio_input[train_idx], data_label[train_idx]
        X_val, y_val = data_audio_input[val_idx], data_label[val_idx]

        train_dataset = RawWaveformDataset(X_train, y_train)
        valid_dataset = RawWaveformDataset(X_val, y_val)

        # ✅ 데이터로더 생성
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        # ✅ 모델 초기화 (각 Fold마다 새로 학습해야 함)
        model = C_test(output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # ✅ 학습 및 검증 루프
        for epoch in range(num_epochs):
            # ✅ 훈련 과정
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # ✅ 검증 과정
            model.eval()
            val_loss = 0.0
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # ✅ 결과 출력 (logger 사용)
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_acc = correct / total

            logger.info(f"📌 Fold [{fold+1}/{k_folds}] | Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # ✅ 현재 Fold의 성능 저장
        fold_results.append(val_acc)

        # ✅ Fold별 모델 저장 (옵션)
        model_path = f"model_fold{fold+1}.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"✅ Fold [{fold+1}] 모델 저장 완료: {model_path}")

    # ✅ 최종 평균 성능 계산
    logger.info(f"\n✅ 5-Fold Cross Validation 완료!")
    logger.info(f"✅ 각 Fold 정확도: {fold_results}")
    logger.info(f"✅ 평균 정확도: {np.mean(fold_results):.4f}")