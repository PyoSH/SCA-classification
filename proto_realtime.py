from src.feature_extraction import *
from src.model_definition import *
from src.dataset import *
from src.audio_utils import *
from config import cfg, update_config
import numpy as np
import argparse
from loguru import logger

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
                         output_dim=cfg.HYPERPARAMS.NUM_CLASSES)
    elif cfg.HYPERPARAMS.MODELTYPE == 'LSTM':
        model = LSTMModel(input_dim=mfcc_const.n_mfcc, hidden_dim=cfg.HYPERPARAMS.HIDDEN_SIZE,
                          num_layers=cfg.HYPERPARAMS.NUM_LAYERS,
                          output_dim=cfg.HYPERPARAMS.NUM_CLASSES)
    model.load_state_dict(torch.load(cfg.PATH.MODEL_PATH))

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

    try:
        while True:
            input_raw = stream.read(CHUNK)
            input_np = np.frombuffer(input_raw, dtype=np.int16) # 여기서 음질이 좀 뭉게지려나
            featureVector = get_frame_to_mfcc(input_np, samplingRate=mfcc_const.sr, num_cepstralCoefficient=mfcc_const.n_mfcc,
                              hop_length=mfcc_const.hop_length, len_fft=mfcc_const.len_fft)
            input_tensor = torch.tensor(featureVector[:, :, :mfcc_const.n_mfcc], dtype=torch.float32) # 입력 텐서 주의!!!!

            with torch.no_grad():
                model.eval()
                outputs = model(input_tensor)

            _, predicted = torch.max(outputs.data, 1)
            predicted_class = class_labels[predicted.item()]
            logger.info(f'Predicted: {predicted_class}')

    except KeyboardInterrupt:
        logger.info("실시간 오디오 분류 종료.")
        stream.stop_stream()
        stream.close()
        p.terminate()