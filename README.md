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
