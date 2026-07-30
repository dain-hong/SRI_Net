#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPZ에서 채널별 mean/std를 계산해 meanstd.json 생성.

주의:
  - 반드시 "학습에 쓰는 split"만 대상으로 계산할 것 (test 포함 시 data leakage).
  - streaming 방식(sum, sum of squares)이라 파일이 많아도 메모리 안 터짐.
  - 손상된 NPZ는 건너뛰고 마지막에 목록 출력.

사용:
    python compute_meanstd.py --npz_base /path/to/npz/ffpp --out meanstd.json
    python compute_meanstd.py --npz_base ... --max_files 5000   # 서브샘플로 빠르게
    python compute_meanstd.py --npz_base ... --mask_zero       # UV 빈 영역(0) 제외
"""
import os
import json
import glob
import random
import argparse
import numpy as np

KEYS = ['tex_uv', 'dl_uv', 'spr_uv']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz_base', required=True, help='NPZ 루트 (재귀 탐색)')
    ap.add_argument('--out', default='meanstd.json')
    ap.add_argument('--max_files', type=int, default=0,
                    help='0이면 전체. 양수면 랜덤 서브샘플 개수')
    ap.add_argument('--mask_zero', action='store_true',
                    help='세 채널이 모두 0인 픽셀(UV 바깥)을 통계에서 제외')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.npz_base, '**/*.npz'), recursive=True))
    if not files:
        raise SystemExit(f'NPZ를 못 찾음: {args.npz_base}')
    print(f'[scan] 총 {len(files)}개 NPZ 발견')

    if args.max_files and len(files) > args.max_files:
        random.seed(args.seed)
        files = random.sample(files, args.max_files)
        print(f'[scan] 랜덤 서브샘플 {len(files)}개로 계산')

    # 채널별 누적: 합, 제곱합, 픽셀 수
    acc = {k: {'s': np.zeros(3, np.float64),
               'sq': np.zeros(3, np.float64),
               'n': 0} for k in KEYS}
    bad_files = []
    missing_key_count = {k: 0 for k in KEYS}

    for i, p in enumerate(files, 1):
        try:
            npz = np.load(p)
            for k in KEYS:
                if k not in npz:
                    missing_key_count[k] += 1
                    continue
                arr = npz[k].astype(np.float64)          # (H, W, 3)
                flat = arr.reshape(-1, arr.shape[-1])    # (H*W, 3)

                if args.mask_zero:
                    keep = ~np.all(flat == 0, axis=1)
                    flat = flat[keep]
                    if flat.size == 0:
                        continue

                acc[k]['s']  += flat.sum(axis=0)
                acc[k]['sq'] += (flat ** 2).sum(axis=0)
                acc[k]['n']  += flat.shape[0]
        except Exception as e:
            bad_files.append((p, f'{type(e).__name__}: {e}'))

        if i % 1000 == 0:
            print(f'  ... {i}/{len(files)} 처리')

    result = {}
    for k in KEYS:
        n = acc[k]['n']
        if n == 0:
            print(f'[warn] {k}: 유효 픽셀 0개 → 스킵')
            continue
        mean = acc[k]['s'] / n
        var = acc[k]['sq'] / n - mean ** 2
        var = np.maximum(var, 0.0)          # 부동소수 오차로 음수 되는 것 방지
        std = np.sqrt(var)
        std = np.where(std < 1e-6, 1.0, std)  # std=0인 죽은 채널 방어
        result[k] = {'mean': mean.tolist(), 'std': std.tolist()}

        print(f'\n[{k}]  (픽셀 {n:,}개)')
        print(f'  mean = {[round(float(v), 6) for v in mean]}')
        print(f'  std  = {[round(float(v), 6) for v in std]}')
        if missing_key_count[k]:
            print(f'  ⚠️ 이 키가 없는 파일 {missing_key_count[k]}개')

    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\n✅ 저장 완료: {args.out}')

    if bad_files:
        print(f'\n⚠️ 손상/읽기실패 NPZ {len(bad_files)}개 (앞 10개만 표시):')
        for p, e in bad_files[:10]:
            print(f'  {p}  →  {e}')
        with open(args.out + '.badfiles.txt', 'w') as f:
            for p, e in bad_files:
                f.write(f'{p}\t{e}\n')
        print(f'  전체 목록: {args.out}.badfiles.txt')


if __name__ == '__main__':
    main()