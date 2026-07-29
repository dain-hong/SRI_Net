# SRI_Net
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
<img width="892" height="253" alt="image" src="https://github.com/user-attachments/assets/e24be0ac-8cfc-46de-8955-874ba4b1b25e" />

입력 얼굴 이미지는 조명 및 반사 성분을 분해하여 학습용 NPZ 형식으로 변환됩니다.

### Pipeline

1. **3D Face Reconstruction**
   - 3DDFA를 이용하여 얼굴의 3D 형상을 복원합니다.

2. **Texture Estimation**
   - Multi-Scale Retinex(MSR)를 이용하여 Texture를 추정합니다.

3. **Illumination Separation**
   - Spherical Harmonics(SH)를 이용하여 Ambient Light와 Direct Light를 추정하고, Specular Reflection을 계산합니다.

4. **UV Map Generation**
   - 각 조명 성분을 UV Map으로 변환합니다.

5. **NPZ Generation**
   - `image`, `tex_uv`, `amb_uv`, `dl_uv`, `spr_uv`, `label`을 저장합니다.


## 구현 참고 사항

원 논문과 달리, 본 구현에서는 MSR 출력에 `exp`를 적용하여 Texture를 선형 도메인으로 변환한 뒤 조명 성분을 분해합니다.

이는 다음 조명 모델이 선형 도메인에서 성립하도록 하기 위한 구현상의 차이입니다.

```text
I = (Ambient + Direct) × Texture





