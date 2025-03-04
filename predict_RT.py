from src.feature_extraction import *
from src.model_definition import *
from src.dataset import *
from src.audio_utils import *
from config import cfg, update_config
import numpy as np
import argparse
from loguru import logger
import time

# 디바이스 설정: Apple Silicon의 MPS, CUDA, 또는 CPU
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

parser = argparse.ArgumentParser(description='Running audio classification')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)

args = parser.parse_args()
update_config(cfg, args)

if not os.path.exists(cfg.PATH.MODEL_PATH):
    logger.info("There is no model in path :( ")
else:
    logger.info("Model found! :) ")

logger.info("Running proto_test ...")
logger.info(f'MODEL path: {cfg.PATH.MODEL_PATH}, CHUNK: {cfg.FEATUREPARAMS.CHUNK}')

mfcc_const = MFCC_params(cfg.FEATUREPARAMS.SAMPLING_RATE, cfg.FEATUREPARAMS.NUM_CEPSTRAL_COEFFICIENTS,
                         cfg.FEATUREPARAMS.HOP_LENGTH, cfg.FEATUREPARAMS.LEN_WINDOW)
CHUNK = cfg.FEATUREPARAMS.CHUNK
class_labels = cfg.HYPERPARAMS.LABEL_CLASS

if __name__ == '__main__':
    # 학습된 모델 불러오기
    model = None
    if cfg.HYPERPARAMS.MODELTYPE == 'RNN':
        model = RNNModel(input_dim=mfcc_const.n_mfcc, hidden_dim=cfg.HYPERPARAMS.HIDDEN_SIZE,
                         num_layers=cfg.HYPERPARAMS.NUM_LAYERS,
                         output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    elif cfg.HYPERPARAMS.MODELTYPE == 'LSTM':
        model = LSTMModel(input_dim=mfcc_const.n_mfcc, hidden_dim=cfg.HYPERPARAMS.HIDDEN_SIZE,
                          num_layers=cfg.HYPERPARAMS.NUM_LAYERS,
                          output_dim=cfg.HYPERPARAMS.NUM_CLASSES).to(device)
    elif cfg.HYPERPARAMS.MODELTYPE == 'CRNN':
        model = CRNN_3().to(device)

    model.load_state_dict(torch.load(cfg.PATH.MODEL_PATH, weights_only=True))

    # 입력장치 선택
    p = pyaudio.PyAudio()
    dev_idx = select_input_device()

    # 오디오
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=mfcc_const.sr,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=dev_idx)

    iterate_n = 0
    sum_diff = 0

    try:
        while True:
            iterate_n += 1

            # logger.info('[0,1)')
            logger.info('audio read start')
            input_raw = stream.read(CHUNK)
            input_np = np.frombuffer(input_raw, dtype=np.int16) # 여기서 음질이 좀 뭉게지려나
            # logger.info('(1,2]')

            start_t = time.time()
            logger.info(f'audio read done, {input_np.shape}')

            tensor_audio = None
            if cfg.HYPERPARAMS.MODELTYPE != 'CRNN':
                featureVector = get_frame_to_mfcc(input_np, samplingRate=mfcc_const.sr,
                                                  num_cepstralCoefficient=mfcc_const.n_mfcc,
                                                  hop_length=mfcc_const.hop_length, len_fft=mfcc_const.len_fft)
                logger.info('audio MFCC processed')
                tensor_audio = torch.tensor(featureVector[:, :, :mfcc_const.n_mfcc], dtype=torch.float32).to(device)  # 입력 텐서 주의!!!!

            elif cfg.HYPERPARAMS.MODELTYPE == 'CRNN':
                tensor_audio = torch.tensor(input_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            # logger.info('(2,3]')
            with torch.no_grad():
                model.eval()
                outputs = model(tensor_audio)

            end_t = time.time()
            logger.info('predict done')

            diff = end_t - start_t
            sum_diff += diff

            _, predicted = torch.max(outputs.data, 1)
            predicted_class = class_labels[predicted.item()]
            logger.info(f'Predicted: {predicted_class}, diff t: {diff:.3f} ms')

    except KeyboardInterrupt:
        logger.info("실시간 오디오 분류 종료.")
        stream.stop_stream()
        stream.close()
        p.terminate()

    logger.info(f'total diff t: {sum_diff / iterate_n * 100:.3f} ms')