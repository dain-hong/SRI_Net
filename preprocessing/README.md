# Deepfake Detection Preprocessing (NPZ 생성)

3DDFA 기반 3D reconstruction + Retinex MSR을 이용해
illumination(ambient/direct)-reflectance-specular(SPR) 분해된 UV map을 생성합니다.
FF++, DF40, Celeb-DF-v1 세 데이터셋을 하나의 CLI로 처리합니다.

## 구조
```
preprocessing/
├── extract_npz.py        # 전처리 스크립트 (공유 로직 + 데이터셋별 로더 + CLI, 단일 파일)
├── config.example.yaml   # 설정 예시 (git에 포함)
├── config.yaml           # 실제 경로 (.gitignore, 직접 생성)
└── README.md
```

`extract_npz.py` 내부는 3개 섹션으로 구성:
1. **공유 로직** (`retinex_msr`, `sh9_basis_vertex`, `process_single_row`, `init_worker`) — 데이터셋 무관, 3D reconstruction + illumination decomposition 계산
2. **데이터셋별 로더** (`load_tasks_ffpp`, `load_tasks_df40`, `load_tasks_celebdf`) — 어떤 이미지를 어떤 라벨로 처리할지 목록만 생성
3. **CLI** (`main`, `parse_args`, `load_config`) — 위 둘을 엮어서 실행

## 사전 준비

### 1. 3DDFA 클론
```bash
git clone https://github.com/cleardusk/3DDFA.git
export THREEDDFA_ROOT=/path/to/3DDFA
```
`phase1_wpdc_vdc.pth.tar`는 3DDFA 레포 README의 다운로드 링크에서 직접 받을 것 (재배포하지 않음).

### 2. 데이터셋 준비
- **FF++**: https://github.com/ondyari/FaceForensics (공식 신청 필요), DeepfakeBench 포맷 `dataset_json` 필요
- **DF40**: 공식 GitHub/신청 경로 참고
- **Celeb-DF-v1**: 저자 공식 신청 폼

세 데이터셋 모두 원본 데이터는 이 레포에 포함하지 않습니다.

### 3. Python 패키지
```bash
pip install dlib torch torchvision opencv-python scipy pyyaml tqdm
```

## 사용법
```bash
cp config.example.yaml config.yaml
# config.yaml 열어서 본인 경로로 수정: model_path, tri_path, uv_path, npz_root,
# ffpp.*, df40.*, celebdf.* 섹션

python extract_npz.py --config config.yaml --dataset ffpp      # FF++만
python extract_npz.py --config config.yaml --dataset df40      # DF40 6개 서브셋
python extract_npz.py --config config.yaml --dataset celebdf   # Celeb-DF-v1
python extract_npz.py --config config.yaml --dataset all       # 전체

# FF++ meanstd.json은 자동 계산되지 않음 — 필요할 때 별도로 실행
python extract_npz.py --config config.yaml --dataset stats
```

## 출력
`{npz_root}/{ffpp,df40,celebdf}/...` 아래 NPZ 파일 생성. 각 NPZ는:

| 키 | 내용 |
|---|---|
| `image` | 원본 프레임 (리사이즈, RGB) |
| `tex_uv` | Retinex MSR reflectance (log→exp로 선형 복귀, UV map) |
| `dl_uv` | direct light (1~8차 SH 최소자승 피팅) |
| `amb_uv` | ambient light (0차 SH 최소자승 피팅) |
| `spr_uv` | specular residual = (I - (amb+dl)×T_msr) / T_msr |
| `label` | 0=real, 1=fake |

FF++ NPZ 처리와 통계 계산은 분리되어 있습니다. `{npz_root}/meanstd.json` (채널별 mean/std)은 `--dataset stats`로 별도 실행해야 생성됩니다.
**DF40/Celeb-DF는 자체 통계를 계산하지 않고 이 FF++ train 통계를 그대로 재사용합니다** (모든 데이터셋이 동일 스케일 분포를 갖도록 통일).
mean/std 계산은 이미지 단위 numpy 벡터 연산으로 처리되어(픽셀 단위 순회 없음) FF++ 전체 train NPZ를 대상으로도 비교적 빠르게 끝납니다.

## 파이프라인 요약
1. dlib 얼굴 검출 → 3DDFA로 3D vertex 복원 (mobilenet_v1 기반 3DMM 파라미터 회귀)
2. Vertex normal + 9차 Spherical Harmonics(SH9) basis 계산
3. Retinex MSR로 reflectance 추출 (3-scale Gaussian, log-domain → exp로 선형 복귀)
4. Ambient(0차 SH) → Direct(1~8차 SH, ambient 잔차 피팅) → Specular(diffuse 모델 잔차) 순차 OLS 피팅
5. Vertex 값들을 UV 평면에 렌더링(`crender_colors`) 후 NPZ로 압축 저장

## 데이터셋별 차이 (extract_npz.py 내 로더 함수들)
- **FF++**: train/val/test 전체 split 처리, DeepfakeBench JSON 포맷(c23 기준)
- **DF40**: test split만 처리, 서브셋별(e4s/facedancer/uniface/simswap/inswap/fsgan) 별도 JSON, 원본 JSON 경로 접두사를 로컬 경로로 치환(`path_prefix_map`)
- **Celeb-DF-v1**: test split만 처리, real/fake 별도 키

## 주의
- FF++/Celeb-DF/DF40 등 원본 데이터셋과 여기서 생성된 NPZ는 재배포 금지 대상이므로 이 레포에는 포함하지 않음 (`.gitignore` 참고)
- `config.yaml`은 서버별 절대경로가 들어가므로 커밋하지 않음 — `config.example.yaml`만 참고용으로 제공