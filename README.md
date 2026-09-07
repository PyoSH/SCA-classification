# Seabed Crushing Audio Classification

**End-to-end multi-scale 1D CNN–LSTM for real-time working-state recognition of a seabed mining robot from raw hydrophone audio.**

This repository contains the models, training pipeline, and analysis tools behind:

> S. Pyo, T.-K. Yeu, Y. Lee, J.-B. Han, and D. Park,
> "Multi-scale Convolutional Recurrent Neural Networks for Real-Time Classification of the Seabed Crushing Process,"
> *IEEE Journal of Oceanic Engineering*, under review (2026).

Earlier versions of this work appeared at **IEEE IROS 2025** (end-to-end multi-scale approach) and in the **Journal of Ocean Engineering and Technology, 2025** (RNN on MFCC features).

<img width="1897" height="688" alt="Seabed crushing overview" src="https://github.com/user-attachments/assets/3a68440d-2dc6-42be-a342-c43b233855df" />

---

## 1. Background: what is seabed crushing, and why listen to it?

<!-- TODO: verify the domain description below against the paper's introduction and adjust wording -->

In deep-sea mining, a crawler-type mining robot breaks seabed material (e.g., manganese nodules or crust) with a rotating cutter before the material is collected and lifted to the surface. Whether the cutter is actually engaging the seabed — and how — is surprisingly hard to observe:

- Sediment plumes blind cameras during cutting.
- Motor current and torque signals are tool-specific and lag the mechanical event.
- Operators on the surface vessel need the information in real time, from a robot several kilometres away.

The **sound** of cutting, however, is distinct, immediate, and cheap to acquire with a single hydrophone. This project treats the problem as **working-state classification from raw acoustic time series**, with four states:

| Label | Meaning |
|---|---|
| `standby` | Robot powered, cutter stationary |
| `idling` | Cutter rotating, not in contact with the seabed |
| `cutting` | Cutter engaging the seabed |
| `unknown` | Everything else (transients, noise, unlabeled segments) |

Classification is done on **100 ms frames** (4410 samples at 44.1 kHz), which is what makes real-time operation possible.

## 2. Motivation: skip the hand-tuned front end

<img width="785" height="348" alt="Feature extraction comparison" src="https://github.com/user-attachments/assets/f00a8b56-8703-48cc-894d-a81aa6bf635b" />

Almost every acoustic classifier — from MFCC/GFCC pipelines to spectrogram CNNs and Audio Spectrogram Transformers — sits on top of a Fourier front end. That front end has parameters (window length, hop length, filterbank layout) that must be re-tuned whenever the acoustic situation changes. Underwater, the acoustic situation changes constantly.

The idea here is to **feed the raw waveform** and let the network learn the front end. The design question then becomes: how do you give a 1D CNN enough frequency resolution at low frequencies *and* enough time resolution at high frequencies, without a spectrogram?

## 3. Method: three pathways, three time scales

<img width="4508" height="2154" alt="Model architecture" src="https://github.com/user-attachments/assets/4fcdde25-b29b-4f97-aedc-d39acf71e201" />

The proposed model (`P` in `src/model_definition.py`) runs three parallel 1D-CNN pathways on the same raw frame, differing only in the kernel size of the first convolution:

| Pathway | First-layer kernel | Receptive field @ 44.1 kHz | Intended band |
|---|---|---|---|
| small | 64 samples | ≈ 1.5 ms | high frequency |
| medium | 256 samples | ≈ 5.8 ms | mid frequency |
| large | 2048 samples | ≈ 46 ms | low frequency |

Each pathway is a 4-block stack of `Conv1d → BatchNorm → ReLU → MaxPool` (`CNNFeatureExtractor`), producing 256 channels. Two further design choices:

- **Mel-filterbank initialization.** The first convolution of the medium and large pathways is initialized with a mel filterbank (`initialize_mel_filter`), so the network starts from a physically sensible frequency decomposition instead of random filters, but is free to move away from it.
- **Attention + LSTM.** The three pathway outputs are concatenated (768 channels), weighted by a learned attention module, and passed to a 2-layer LSTM (hidden 128) whose final state is classified by a linear layer.

### Ablation ladder

The design was checked one component at a time. All variants live in `src/model_definition.py` with a matching config in `config/`:

| Model | Input | Scales | Mel init | Attention | Purpose |
|---|---|---|---|---|---|
| `B1` | MFCC | – | – | – | classical front-end baseline |
| `B2-small / -middle / -large` | raw | single (64 / 256 / 2048) | – | – | does any single scale suffice? |
| `B3` | raw | multi | – | – | effect of multi-scale alone |
| `B4` | raw | multi | ✓ | – | effect of mel initialization |
| `P` | raw | multi | ✓ | ✓ | proposed |
| `SVM` | MFCC | – | – | – | non-deep baseline |
| `AST` | spectrogram | – | – | – | transformer baseline |

<img width="928" height="333" alt="Ablation results" src="https://github.com/user-attachments/assets/0b5472f3-4750-41ac-8fb5-dc271bcf939f" />

## 4. Data and training

<img width="897" height="144" alt="Experimental setup" src="https://github.com/user-attachments/assets/7bf8b3e0-3709-4066-b158-64dfe1d03223" />

Audio was recorded during cutting experiments in KRISO's deep-sea mining test tank, labeled by working state, and segmented into 100 ms frames.

<img width="2795" height="1291" alt="Training pipeline" src="https://github.com/user-attachments/assets/dd9ec839-9bef-40c2-88e9-1f0a8f258459" />

- Sampling rate 44.1 kHz, frame length 100 ms, per-frame standardization
- **5-fold cross-validation** (`k-fold.py`), Adam, lr 1e-3, batch 128, 50 epochs
- A held-out `inspection` set (cutting only) is used for filter/feature-map analysis

### Results

<!-- TODO: fill from the paper's results table -->

| Model | Accuracy (5-fold mean ± std) | Params | Inference / frame |
|---|---|---|---|
| B1 (MFCC + LSTM) | – | – | – |
| B2-middle | – | – | – |
| B3 | – | – | – |
| B4 | – | – | – |
| AST | – | – | – |
| **P (proposed)** | **–** | – | – |

## 5. Does the network learn what it was designed to learn?

<img width="2945" height="858" alt="First-layer filter maps per pathway" src="https://github.com/user-attachments/assets/a4ed5b1e-8cb8-41b2-b941-cceffe2cc507" />

After training, the first-layer filters of each pathway are inspected in the frequency domain (`viz_MFCC.py`, `inspection_model.py`, `src/feature_extraction.py::viz_filter_map`). The small-kernel pathway concentrates its energy at high frequencies and the large-kernel pathway at low frequencies, i.e., the scales separate by band as intended rather than collapsing onto the same features.

Additional analysis tools:

- `t-SNE.py`, `t-SNE_mfcc.py` — embedding separation of the four classes, learned features vs. MFCC
- `etc/gaussian_similarity_check.py` — class-wise similarity structure of the embeddings
- `viz_attention.py` — attention weights over the three pathways

## 6. Real-time deployment

`predict_RT.py` runs the model frame by frame and logs per-frame inference latency (CPU / CUDA / Apple MPS). For deployment on the mining robot, a separate ROS-connected socket server wraps the model:
[realtime_audio_classification_ros](https://github.com/PyoSH/realtime_audio_classification_ros).

## 7. Repository layout

```
├── config/            # one YAML per experiment (P, B1–B4, SVM, AST, …); default.py holds the schema
├── src/
│   ├── model_definition.py   # all model variants
│   ├── dataset.py            # frame dataset
│   ├── feature_extraction.py # MFCC utilities, filter/feature-map and t-SNE visualization
│   ├── audio_utils.py
│   └── train_utils.py
├── train.py           # single train/test split
├── k-fold.py          # 5-fold cross-validation
├── test.py
├── predict_RT.py      # real-time inference with latency logging
├── pseudo_RT.py       # offline replay of a recording at real-time pace
├── dataset_bringup.py # build .npy datasets from recordings
├── inspection_model.py, viz_*.py, t-SNE*.py   # analysis
├── results/           # trained weights (.pth)
└── etc/               # exploratory notebooks and scripts
```

## 8. Quick start

```bash
pip install torch torchaudio numpy scikit-learn yacs loguru matplotlib   # TODO: pin versions in requirements.txt

# build datasets from recordings
python dataset_bringup.py

# 5-fold cross-validation of the proposed model
python k-fold.py --cfg config/exp-P.yaml

# run an ablation variant
python k-fold.py --cfg config/exp-B3.yaml

# real-time inference with a trained model
python predict_RT.py --cfg config/exp-P.yaml
```

Dataset `.npy` files are not included in the repository. <!-- TODO: state availability (on request / link) -->

## 9. Citation

```bibtex
@article{pyo2026seabed,
  author  = {Pyo, Seunghyun and Yeu, Tae-Kyeong and Lee, Yeongjun and Han, Jong-Boo and Park, Daegil},
  title   = {Multi-scale Convolutional Recurrent Neural Networks for Real-Time Classification of the Seabed Crushing Process},
  journal = {IEEE Journal of Oceanic Engineering},
  year    = {2026},
  note    = {Under review}
}
```

Related:

- S. Pyo et al., "Learning to perceive from raw time-series: An end-to-end, multi-scale approach for real-time task classification," *IEEE/RSJ IROS*, 2025.
- S. Pyo et al., "Real-time acoustic signal classification using RNN for underwater cutting process monitoring and situational awareness," *Journal of Ocean Engineering and Technology*, pp. 92–102, 2025. doi:10.26748/ksoe.2024.069
