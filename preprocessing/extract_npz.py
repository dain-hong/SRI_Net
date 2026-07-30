#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FF++ / DF40 / Celeb-DF 통합 NPZ 전처리 (단일 파일 버전)

3DDFA 기반 3D reconstruction + Retinex MSR을 이용해
illumination(ambient/direct)-reflectance-specular(SPR) 분해된 UV map을 생성.

사용 예:
    python extract_npz.py --config config.yaml --dataset ffpp
    python extract_npz.py --config config.yaml --dataset df40
    python extract_npz.py --config config.yaml --dataset celebdf
    python extract_npz.py --config config.yaml --dataset all
"""

import os
import sys
import cv2
import dlib
import json
import yaml
import torch
import argparse
import traceback
import numpy as np
import scipy.io as sio
import torchvision.transforms as transforms
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- 3DDFA 레포 경로: 환경변수 THREEDDFA_ROOT로 지정 ---
THREEDDFA_ROOT = os.environ.get('THREEDDFA_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(THREEDDFA_ROOT)
import mobilenet_v1
from utils.ddfa import ToTensorGjz, NormalizeGjz, reconstruct_vertex
from utils.inference import parse_roi_box_from_bbox, crop_img, predict_dense
from utils.render import crender_colors


# =========================================================
# 1. 공유 로직 (retinex_msr, SH9, process_single_row, init_worker)
#    — 데이터셋 무관, 수정 없음, 한 곳에만 존재
# =========================================================
global_model = global_detector = global_tri = global_uv_coords = global_transform = None
CFG = {}


def retinex_msr(img_u8, sigmas):
    """0~255 스케일 입력 -> 3-scale log-domain MSR -> exp로 선형 reflectance 복귀.
    이 exp가 핵심: 이후 I = (ambient+direct) x T_msr 의 선형 방정식이 성립하려면
    T_msr이 log가 아니라 선형(양수, ~1 근방) 스케일이어야 함."""
    img   = np.maximum(img_u8.astype(np.float32), 1.0)
    log_I = np.log(img)
    msr   = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        for s in sigmas:
            blur        = cv2.GaussianBlur(img[..., c], (0, 0), s)
            msr[..., c] += log_I[..., c] - np.log(np.maximum(blur, 1.0))
        msr[..., c] /= len(sigmas)
    return np.exp(msr)


def compute_vertex_normals(ver, tri):
    normals = np.zeros_like(ver)
    v0, v1, v2 = ver[tri[:, 0]], ver[tri[:, 1]], ver[tri[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    for i in range(3):
        np.add.at(normals, tri[:, i], fn)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return normals


def sh9_basis_vertex(normals):
    nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]
    c1 = np.sqrt(1.0 / (4 * np.pi))
    c2 = np.sqrt(3.0 / (4 * np.pi))
    c4 = np.sqrt(15.0 / (4 * np.pi))
    c7 = np.sqrt(5.0 / (16 * np.pi))
    c9 = np.sqrt(15.0 / (16 * np.pi))
    return np.stack([
        c1 * np.ones(len(nx)),
        c2 * nx, c2 * ny, c2 * nz,
        c4 * (nx * ny), c4 * (ny * nz),
        c7 * (2 * nz**2 - nx**2 - ny**2),
        c4 * (nx * nz), c9 * (nx**2 - ny**2)
    ], axis=1).astype(np.float32)


def init_worker(cfg):
    """멀티프로세싱 워커 초기화: 3DDFA 모델/얼굴검출기/삼각형 메시/UV 좌표 로드."""
    global global_model, global_detector, global_tri, global_uv_coords, global_transform, CFG
    CFG = cfg
    torch.set_num_threads(1)
    ckpt = torch.load(cfg['model_path'], map_location='cpu', weights_only=False)['state_dict']
    global_model = mobilenet_v1.mobilenet_1(num_classes=62)
    global_model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    global_model.eval()
    global_detector  = dlib.get_frontal_face_detector()
    global_tri       = sio.loadmat(cfg['tri_path'])['tri'].T - 1
    global_uv_coords = np.load(cfg['uv_path'])
    global_transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(127.5, 128)])


def process_single_row(row_data):
    """(img_path, label, final_save_path) -> NPZ 저장.
    NPZ: image, tex_uv(reflectance,exp), dl_uv(direct), amb_uv(ambient), spr_uv(specular residual), label"""
    img_path, label, final_save_path = row_data
    uv_size, out_size = CFG['uv_size'], CFG['out_size']
    sigmas = CFG.get('sigmas', (15.0, 80.0, 120.0))

    if os.path.exists(final_save_path): return "SKIP", None
    if not os.path.exists(img_path): return "FAIL", (img_path, "이미지 파일 없음")
    os.makedirs(os.path.dirname(final_save_path), exist_ok=True)

    try:
        img = cv2.imread(img_path)
        if img is None: return "FAIL", (img_path, "로딩 실패")
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        rects = global_detector(img_rgb, 1)
        if len(rects) == 0: return "FAIL", (img_path, "얼굴 없음")
        rect = rects[0]
        roi  = parse_roi_box_from_bbox([rect.left(), rect.top(), rect.right(), rect.bottom()])
        crop = cv2.resize(crop_img(img, roi), (120, 120))
        inp  = global_transform(crop).unsqueeze(0)

        with torch.no_grad():
            param = global_model(inp).squeeze().numpy()

        ver_3xV    = predict_dense(param, roi)
        ver_3d_Vx3 = reconstruct_vertex(param, dense=True).T

        vnorm     = compute_vertex_normals(ver_3d_Vx3, global_tri)
        valid_bin = vnorm[:, 2] > 0
        if valid_bin.sum() == 0: return "FAIL", (img_path, "유효 vertex 없음")

        pt2d = ver_3xV[:2, :].copy()
        pt2d[0, :] = np.clip(pt2d[0, :], 0, w - 1)
        pt2d[1, :] = np.clip(pt2d[1, :], 0, h - 1)
        pt2d = np.round(pt2d).astype(np.int32)

        img_pixel = img_rgb[pt2d[1, :], pt2d[0, :], :].T.astype(np.float64)
        msr_full  = retinex_msr(img_rgb, sigmas)
        msr_pixel = msr_full[pt2d[1, :], pt2d[0, :], :].T.astype(np.float64)

        harmonic = sh9_basis_vertex(-vnorm)
        Hv    = harmonic[valid_bin, :]
        img_v = img_pixel[:, valid_bin]
        msr_v = msr_pixel[:, valid_bin]

        nChannels = 3
        gamma_amb = np.zeros(nChannels, dtype=np.float64)
        for c in range(nChannels):
            gamma_amb[c] = np.linalg.lstsq(Hv[:, 0:1] * msr_v[c, :, None], img_v[c, :], rcond=None)[0][0]
        facial_amb = np.outer(gamma_amb, harmonic[:, 0])

        gamma_dir = np.zeros((8, nChannels), dtype=np.float64)
        for c in range(nChannels):
            tex_in_amb = msr_v[c, :] * facial_amb[c, valid_bin]
            gamma_dir[:, c] = np.linalg.lstsq(Hv[:, 1:9] * msr_v[c, :, None], img_v[c, :] - tex_in_amb, rcond=None)[0]
        facial_dir = (harmonic[:, 1:9] @ gamma_dir).T

        spr_numerator = img_pixel - (facial_amb + facial_dir) * msr_pixel
        facial_spr    = spr_numerator / np.maximum(msr_pixel, 1e-6)

        uv_verts = np.zeros((ver_3xV.shape[1], 3), dtype=np.float32)
        uv_verts[:, 0] = global_uv_coords[:, 0] * (uv_size - 1)
        uv_verts[:, 1] = (1.0 - global_uv_coords[:, 1]) * (uv_size - 1)
        uv_verts[:, 2] = ver_3xV[2, :]
        triangles = global_tri.astype(np.int32)

        def to_uv(colors_3xV):
            return cv2.resize(
                crender_colors(uv_verts, triangles, colors_3xV.T.astype(np.float32), uv_size, uv_size, c=3),
                (out_size, out_size)
            )

        np.savez_compressed(final_save_path,
            image  = cv2.resize(img_rgb.astype(np.float32), (out_size, out_size)),
            tex_uv = to_uv(msr_pixel).astype(np.float32),
            dl_uv  = to_uv(facial_dir).astype(np.float32),
            amb_uv = to_uv(facial_amb).astype(np.float32),
            spr_uv = to_uv(facial_spr).astype(np.float32),
            label  = label
        )
        return "SUCCESS", None

    except Exception:
        return "FAIL", (img_path, traceback.format_exc())


# =========================================================
# 2. 데이터셋별 task 로더 (ffpp / df40 / celebdf)
#    — 여기만 데이터셋마다 다름
# =========================================================
FFPP_FAKE_TYPES = ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT']
FFPP_REAL_TYPE  = 'FF-real'

DF40_DATASET_CONFIG = {
    'real':       {'json': 'e4s_ff.json',        'top_key': 'e4s_ff',        'real_key': 'e4s_Real', 'fake_key': None},
    'e4s':        {'json': 'e4s_ff.json',        'top_key': 'e4s_ff',        'real_key': None,        'fake_key': 'e4s_Fake'},
    'facedancer': {'json': 'facedancer_ff.json', 'top_key': 'facedancer_ff', 'real_key': None,        'fake_key': 'facedancer_Fake'},
    'uniface':    {'json': 'uniface_ff.json',    'top_key': 'uniface_ff',    'real_key': None,        'fake_key': 'uniface_Fake'},
    'simswap':    {'json': 'simswap_ff.json',    'top_key': 'simswap_ff',    'real_key': None,        'fake_key': 'simswap_Fake'},
    'inswap':     {'json': 'inswap_ff.json',     'top_key': 'inswap_ff',     'real_key': None,        'fake_key': 'inswap_Fake'},
    'fsgan':      {'json': 'fsgan_ff.json',      'top_key': 'fsgan_ff',      'real_key': None,        'fake_key': 'fsgan_Fake'},
}
DF40_SUBSETS = list(DF40_DATASET_CONFIG.keys())


def load_tasks_ffpp(cfg, split):
    """cfg 필요 키: ffpp.json_path, ffpp.base_img_dir, npz_root"""
    ffpp_cfg = cfg['ffpp']
    with open(ffpp_cfg['json_path']) as f:
        data = json.load(f)
    ff = data['FaceForensics++']
    tasks = []
    real_count = fake_count = 0

    for vid_id, vid_data in ff[FFPP_REAL_TYPE][split]['c23'].items():
        for frame_win in vid_data['frames']:
            img_path = os.path.join(ffpp_cfg['base_img_dir'], frame_win.replace('\\', '/'))
            npz_path = os.path.join(cfg['npz_root'], 'ffpp', split, 'original_sequences', vid_id,
                                     os.path.basename(frame_win).replace('.png', '.npz'))
            tasks.append((img_path, 0, npz_path))
            real_count += 1

    for fake_type in FFPP_FAKE_TYPES:
        for vid_id, vid_data in ff[fake_type][split]['c23'].items():
            for frame_win in vid_data['frames']:
                img_path = os.path.join(ffpp_cfg['base_img_dir'], frame_win.replace('\\', '/'))
                npz_path = os.path.join(cfg['npz_root'], 'ffpp', split, 'manipulated_sequences', fake_type, vid_id,
                                         os.path.basename(frame_win).replace('.png', '.npz'))
                tasks.append((img_path, 1, npz_path))
                fake_count += 1

    print(f"[ffpp/{split}] real: {real_count}, fake: {fake_count}, total: {len(tasks)}")
    return tasks


def _resolve_df40_path(raw_path, path_prefix_map):
    raw_path = raw_path.replace('\\', '/')
    for prefix, real_base in path_prefix_map.items():
        if raw_path.startswith(prefix):
            return real_base + raw_path[len(prefix):]
    return raw_path


def load_tasks_df40(cfg, subset):
    """cfg 필요 키: df40.json_dir, df40.path_prefix_map. test split만 처리."""
    df40_cfg = cfg['df40']
    ds_cfg = DF40_DATASET_CONFIG[subset]
    json_path = os.path.join(df40_cfg['json_dir'], ds_cfg['json'])

    with open(json_path) as f:
        data = json.load(f)
    top = data[ds_cfg['top_key']]

    tasks = []
    real_count = fake_count = 0

    if ds_cfg['real_key']:
        split_data = top[ds_cfg['real_key']].get('test', {})
        for vid_id, vid_data in split_data.items():
            for frame_path in vid_data['frames']:
                img_path = _resolve_df40_path(frame_path, df40_cfg['path_prefix_map'])
                npz_path = os.path.join(cfg['npz_root'], 'df40', 'real', vid_id,
                                         os.path.basename(frame_path).replace('.png', '.npz'))
                tasks.append((img_path, 0, npz_path))
                real_count += 1

    if ds_cfg['fake_key']:
        split_data = top[ds_cfg['fake_key']].get('test', {})
        for vid_id, vid_data in split_data.items():
            for frame_path in vid_data['frames']:
                img_path = _resolve_df40_path(frame_path, df40_cfg['path_prefix_map'])
                npz_path = os.path.join(cfg['npz_root'], 'df40', subset, vid_id,
                                         os.path.basename(frame_path).replace('.png', '.npz'))
                tasks.append((img_path, 1, npz_path))
                fake_count += 1

    print(f"[df40/{subset}] real: {real_count}, fake: {fake_count}, total: {len(tasks)}")
    return tasks


def load_tasks_celebdf(cfg):
    """cfg 필요 키: celebdf.json_path, celebdf.base_img_dir. test split만 처리."""
    celebdf_cfg = cfg['celebdf']
    with open(celebdf_cfg['json_path']) as f:
        data = json.load(f)['Celeb-DF-v1']

    tasks = []
    for split_key in ['CelebDFv1_real', 'CelebDFv1_fake']:
        is_fake = 1 if split_key == 'CelebDFv1_fake' else 0
        split_data = data[split_key].get('test', {})
        for vid_id, vid_info in split_data.items():
            for frame_rel in vid_info['frames']:
                frame_rel_clean = frame_rel.replace('//', '/')
                img_path = os.path.join(celebdf_cfg['base_img_dir'], frame_rel_clean)
                frame_name = os.path.basename(frame_rel)
                npz_path = os.path.join(cfg['npz_root'], 'celebdf', 'test', str(is_fake), vid_id,
                                         frame_name.replace('.png', '.npz'))
                tasks.append((img_path, is_fake, npz_path))

    print(f"[celebdf/test] total: {len(tasks)}")
    return tasks


# =========================================================
# 3. CLI / 실행 로직
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description='FF++/DF40/Celeb-DF -> NPZ 통합 전처리')
    p.add_argument('--config', type=str, required=True, help='YAML config 파일 경로')
    p.add_argument('--dataset', type=str, default='all', choices=['ffpp', 'df40', 'celebdf', 'all', 'stats'],
                   help='처리할 데이터셋 (기본: all). stats는 FF++ meanstd.json만 별도로 계산')
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for key in ['npz_root', 'model_path', 'tri_path', 'uv_path']:
        if not cfg.get(key):
            raise ValueError(f"config.yaml에 '{key}' 값이 필요합니다.")

    cfg.setdefault('uv_size', 256)
    cfg.setdefault('out_size', 256)
    cfg.setdefault('sigmas', [15.0, 80.0, 120.0])
    cfg.setdefault('num_workers', max(1, cpu_count() // 2))
    return cfg


def worker_cfg(cfg):
    return {
        'model_path': cfg['model_path'],
        'tri_path':   cfg['tri_path'],
        'uv_path':    cfg['uv_path'],
        'uv_size':    cfg['uv_size'],
        'out_size':   cfg['out_size'],
        'sigmas':     tuple(cfg['sigmas']),
    }


def run_tasks(tasks, cfg, tag):
    if not tasks:
        print(f"[{tag}] 태스크 없음, 스킵")
        return
    print(f"\n[{tag}] 전처리 시작 | {cfg['num_workers']}개 코어 | {len(tasks)}개 태스크")
    success = skip = fail = 0
    with Pool(processes=cfg['num_workers'], initializer=init_worker, initargs=(worker_cfg(cfg),)) as pool:
        for status, _ in tqdm(pool.imap_unordered(process_single_row, tasks), total=len(tasks)):
            if status == "SUCCESS": success += 1
            elif status == "SKIP":  skip += 1
            elif status == "FAIL":  fail += 1
    print(f"[{tag}] 완료: 성공 {success}, 스킵 {skip}, 실패 {fail}")


def compute_ffpp_stats(cfg):
    """FF++ train split NPZ 기준 채널별 mean/std 계산 -> meanstd.json.
    DF40/Celeb-DF는 이 통계를 재사용 (별도 계산 안 함).
    numpy 벡터화 연산으로 계산 (파일당 픽셀 단위 Python 루프 없음 -> 대폭 빠름)."""
    npz_dir = os.path.join(cfg['npz_root'], 'ffpp', 'train')
    keys = ['image', 'tex_uv', 'dl_uv', 'spr_uv']

    # 채널별 합, 제곱합, 픽셀 수를 파일 단위로 누적 (단일 패스, 분산 병합 공식 사용)
    sum_    = {k: np.zeros(3, dtype=np.float64) for k in keys}
    sumsq   = {k: np.zeros(3, dtype=np.float64) for k in keys}
    count   = {k: 0 for k in keys}

    files = [os.path.join(root, f) for root, _, fnames in os.walk(npz_dir) for f in fnames if f.endswith('.npz')]
    print(f"\ntrain {len(files)}개로 stats 계산 중 (벡터화)...")
    for fpath in tqdm(files):
        try:
            d = np.load(fpath)
            for k in keys:
                if k in d:
                    arr = d[k].reshape(-1, 3).astype(np.float64)   # (H*W, 3)
                    sum_[k]   += arr.sum(axis=0)
                    sumsq[k]  += (arr ** 2).sum(axis=0)
                    count[k]  += arr.shape[0]
        except Exception:
            pass

    result = {}
    print("\n===== mean / std (FF++ train) =====")
    for k in keys:
        n = count[k]
        mean = sum_[k] / n if n > 0 else np.zeros(3)
        # var = E[x^2] - (E[x])^2
        var  = sumsq[k] / n - mean ** 2 if n > 0 else np.zeros(3)
        std  = np.sqrt(np.maximum(var, 0))
        result[k] = {"mean": mean.tolist(), "std": std.tolist()}
        print(f"{k}: mean={mean.tolist()}, std={std.tolist()}")

    out_path = os.path.join(cfg['npz_root'], 'meanstd.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"meanstd.json 저장 완료: {out_path}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    targets = ['ffpp', 'df40', 'celebdf'] if args.dataset == 'all' else [args.dataset]

    if args.dataset == 'stats':
        compute_ffpp_stats(cfg)
        return

    if 'ffpp' in targets:
        for split in ['train', 'val', 'test']:
            run_tasks(load_tasks_ffpp(cfg, split), cfg, f'ffpp/{split}')

    if 'df40' in targets:
        for subset in DF40_SUBSETS:
            run_tasks(load_tasks_df40(cfg, subset), cfg, f'df40/{subset}')

    if 'celebdf' in targets:
        run_tasks(load_tasks_celebdf(cfg), cfg, 'celebdf/test')


if __name__ == "__main__":
    main()