# Seabed Crushing Audio-Classification

2025 RA-L (in submission)
```
author = {Seunghyun Pyo and Tae-Kyeong Yeu and Yeongjun Lee and Jong-Boo Han and Daegil Park}
title = {Multi-scale Convolutional Recurrent Neural Networks for Real-Time Classification of the Seabed Crushing Process}
journal = {IEEE Oceanic Engineering}
year = {2025}
note = {[Under Review]}
```
## What is Seabed Crushing?
<img width="1897" height="688" alt="Image" src="https://github.com/user-attachments/assets/3a68440d-2dc6-42be-a342-c43b233855df" />
수중 파쇄는 ~ 인데, 

## Why is this method needed? Automated feature extraction! 
<img width="785" height="348" alt="Image" src="https://github.com/user-attachments/assets/f00a8b56-8703-48cc-894d-a81aa6bf635b" />
음향을 입력해서 상황/혹은 무언가를 분류하기 위해서는 음향으로부터 적절한 특징을 추출해야 함을 의미.
그런데 기존의 모델링 기반 방법(MFCC, GFCC)을 포함해 transformer, CNN 등의 학습 기반 방법까지, 대부분의 방법은 Fourier transform을 기반으로 함.
이는 window size, hop length 등 파라미터를 상황에 맞게 튜닝하는 절차를 요구하게 되는데, 다양한 음향적 상황에서 튜닝하는 것은 인간의 노력을 너무 소모함.

원본 음향 파형을 입력하는 것을 통해, 적절한 특징을 모델이 추출 & 분류할 수 있도록 한 것!

## What I've designed?
<img width="4508" height="2154" alt="Image" src="https://github.com/user-attachments/assets/4fcdde25-b29b-4f97-aedc-d39acf71e201" />
음향 정보를 진폭&멜-스펙트로그램으로 관찰하고, 일반화를 위해서 저, 중간, 고주파의 특징을 추출할 수 있는 모델 구조를 설계함.

그리고 이 디자인 요소들을 확인하기 위해, ablation study를 진행함.
<img width="928" height="333" alt="Image" src="https://github.com/user-attachments/assets/0b5472f3-4750-41ac-8fb5-dc271bcf939f" />

## How did I train & tested?

KRISO의 심해저 집광수조에서 수행된 실험을 통해 데이터 수집 & 학습.
<img width="897" height="144" alt="Image" src="https://github.com/user-attachments/assets/7bf8b3e0-3709-4066-b158-64dfe1d03223" />
데이터셋 수집은 다음같이 되었고, 학습 파이프라인은 아래와 같음.
<img width="2795" height="1291" alt="image" src="https://github.com/user-attachments/assets/dd9ec839-9bef-40c2-88e9-1f0a8f258459" />

학습한 뒤, 각 pathway의 첫 번째 CNN의 가중치 맵을 통해서 모델이 주파수 대역별로 학습되었는지, 즉 의도대로 작은 receptive field -> 고주파, 큰 receptive field -> 저주파 해당하는 특징들에 대해 가중치가 높아졌는지. 확인.
<img width="2945" height="858" alt="Image" src="https://github.com/user-attachments/assets/a4ed5b1e-8cb8-41b2-b941-cceffe2cc507" />

### Real World Application
To use this model in the CPOS project, I made an [simple socket comm(with ROS) & prediction program](https://github.com/PyoSH/realtime_audio_classification_ros). 
