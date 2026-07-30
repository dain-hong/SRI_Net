# SRI_Net_Reproduction
Reproduction of SRI-Net for deepfake detection using Phong illumination decomposition

## Overview

SRI-Net(Specular-Reflection-Inconsistency-Network, ICLR 2026)은 딥페이크 탐지를 위해
Phong illumination model 기반으로 얼굴 색상을 **texture / ambient light / direct light / specular reflection**
네 가지 요소로 분해합니다.

생성형 모델은 실제 얼굴에서 나타나는 조명과 반사 특성을 물리적으로 일관되게 재현하기 어렵습니다. SRI-Net은 이 과정에서 발생하는 물리적 불일치를 탐지 단서로 활용하여 딥페이크를 탐지합니다.


이 레포지토리는 원 논문의 아키텍처를 재현한 것입니다.

> Fei, H. et al. "Exploring Specular Reflection Inconsistency for Generalizable Face Forgery Detection." ICLR 2026.
>
## Project Structure

```text
sri_net/
├── config/
│   └── config.yaml              # Training and model configuration
├── data/
│   └── dataset.py               # Dataset loader
├── model/
│   ├── sri_net.py               # SRI-Net architecture
│   └── xception.py              # Xception backbone
├── preprocessing/
│   ├── extract_npz.py           # UV map & NPZ generation
│   ├── compute_meanstd.py       # Dataset normalization statistics
│   └── config.yaml              # Preprocessing configuration
├── train.py                     # Training script
├── test.py                      # Evaluation script
└── README.md
```

## Preprocessing
입력 얼굴 이미지는 조명 및 반사 성분을 분해하여 학습용 NPZ 형식으로 변환됩니다.
<img width="892" height="253" alt="image" src="https://github.com/user-attachments/assets/e24be0ac-8cfc-46de-8955-874ba4b1b25e" />

### Pipeline
1. **3D Face Reconstruction**
   - 3DDFA를 이용하여 얼굴의 3D 형상을 복원

2. **Texture Estimation**
   - Multi-Scale Retinex(MSR)를 이용하여 Texture를 추정

3. **Illumination Separation**
   - Spherical Harmonics(SH)를 이용하여 Ambient Light와 Direct Light를 추정하고, Specular Reflection을 계산

4. **UV Map Generation**
   - 각 조명 성분을 UV Map으로 변환

5. **NPZ Generation**
   - `image`, `tex_uv`, `amb_uv`, `dl_uv`, `spr_uv`, `label`을 저장
  
## 전처리 실행 방법

```text
git clone https://github.com/cleardusk/3DDFA.git
python export_npz.py --json_path <FF++ json> --img_dir <원본 프레임 경로> --out_dir <npz 저장 경로>
python compute_meanstd.py --npz_base <npz train 경로> --out meanstd.json --mask_zero
```

## 구현 참고 사항

원 논문과 달리, 본 구현에서는 MSR 출력에 `exp`를 적용하여 Texture를 선형 도메인으로 변환한 뒤 조명 성분을 분해합니다.

이는 다음 조명 모델이 선형 도메인에서 성립하도록 하기 위한 구현상의 차이입니다.

```text
I = (Ambient + Direct) × Texture
```
## Model
각 브랜치는 Xception Backbone을 통해 특징을 추출하며, 추출된 특징을 융합하여 최종적으로 Real/Fake를 분류합니다.
<img width="912" height="379" alt="image" src="https://github.com/user-attachments/assets/8c529745-c928-41bb-82cb-c143eb85e0c1" />

## Datasets

### 학습 데이터셋

- FaceForensics++ (c23)
  - Deepfakes
  - Face2Face
  - FaceSwap
  - NeuralTextures

### 평가 데이터셋

- FaceForensics++ Test (In-domain)
- DF40 (Cross-dataset)
- Celeb-DF-v1 (Cross-dataset)

### Dataset References
- DeepfakeBench: https://github.com/SCLBD/DeepfakeBench
- DF40: https://github.com/YZY-stack/DF40
> 본 저장소에는 원본 데이터셋이 포함되어 있지 않습니다.

## Train / Test 실행 방법

```text
python train.py --config config/config.yaml
python test.py --config config/config.yaml
```

## Results
### Cross-manipulation (DF40)

| Manipulation | SRI-Net (Paper) | Reproduction |
|--------------|----------------:|-------------:|
| UniFace | 92.0 | 81.74 |
| E4S | 89.4 | 63.85 |
| FaceDancer | 95.3 | 61.99 |
| FSGAN | 94.2 | 79.12 |
| InSwap | 91.1 | 80.64 |
| SimSwap | 83.3 | 74.01 |
| **Average** | **90.9** | **73.21** |

### Cross-dataset (Celeb-DF-v1)

| Dataset | SRI-Net (Paper) | Reproduction |
|---------|----------------:|-------------:|
| Celeb-DF-v1 | **91.3** | **71.11** |


> Differences from the reported performance may arise from implementation details, preprocessing, training settings, and hardware environments.
