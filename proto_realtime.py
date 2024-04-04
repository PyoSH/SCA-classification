from src.feature_extraction import *
from src.model_definition import *
from src.dataset import *
from src.audio_utils import *
import numpy as np

# model - 나중에 yaml 파일 읽는 방식으로 하면 더 좋을듯!
mfcc_conts = MFCC_params(48000, 40, 512, 2048)
CHUNK = 2304
hidden_size = 128
num_layers = 2
num_classes = 3
batch_size = 128
# seq_len = CHUNK // mfcc_conts.hop_length +1

class_labels = ['idling', 'cutting', 'HardCutting']

datapath = 'data'
model_saved_path = os.path.join('results', f'model_trained.pth')

if __name__ == '__main__':
    # 학습된 모델 불러오기
    model = LSTMModel(input_dim=mfcc_conts.n_mfcc, hidden_dim=hidden_size, num_layers=num_layers,
                      output_dim=num_classes)
    model.load_state_dict(torch.load(model_saved_path))

    # 입력장치 선택
    p = pyaudio.PyAudio()
    dev_idx = select_input_device()

    # 오디오
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=mfcc_conts.sr,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=dev_idx)

    try:
        while True:
            input_raw = stream.read(CHUNK)
            input_np = np.frombuffer(input_raw, dtype=np.int16) # 여기서 음질이 좀 뭉게지려나
            featureVector = get_frame_to_mfcc(input_np, samplingRate=mfcc_conts.sr, num_cepstralCoefficient=mfcc_conts.n_mfcc,
                              hop_length=mfcc_conts.hop_length, len_fft=mfcc_conts.len_fft)
            input_tensor = torch.tensor(featureVector[:, :, :mfcc_conts.n_mfcc], dtype=torch.float32) # 입력 텐서 주의!!!!

            with torch.no_grad():
                model.eval()
                outputs = model(input_tensor)

            _, predicted = torch.max(outputs.data, 1)
            predicted_class = class_labels[predicted.item()]
            print(f'{input_tensor.shape}Predicted Class: {predicted_class}, {predicted.item()}')

    except KeyboardInterrupt:
        print("실시간 오디오 분류 종료.")
        stream.stop_stream()
        stream.close()
        p.terminate()