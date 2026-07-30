#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전처리된 NPZ (image, tex_uv, dl_uv, spr_uv, amb_uv) 시각화.

- image / tex_uv / amb_uv: 원래 양수 위주라 min-max stretch로 표시
- dl_uv / spr_uv: signed(부호 있는) 값이므로 min-max stretch로 보면 왜곡됨
  -> RdBu_r diverging colormap (빨강=양수, 파랑=음수, 흰색=0)으로 부호 보존
  -> vmax는 절댓값의 percentile로 잡아서 이상치(occlusion 등) 하나가
     스케일 전체를 지배해 진짜 신호가 뭉개지는 것을 방지

사용:
    python visualize_npz.py --npz_path /path/to/xxx.npz --out preview.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIGNED_COMPONENTS = {'dl_uv', 'spr_uv'}
ORDER = ['image', 'tex_uv', 'amb_uv', 'dl_uv', 'spr_uv']
TITLES = {
    'image': 'Image', 'tex_uv': 'Retinex Texture', 'amb_uv': 'Ambient Light',
    'dl_uv': 'Direct Light', 'spr_uv': 'Specular Reflection',
}


def normalize_for_display(arr):
    """비-signed 컴포넌트(image, tex_uv, amb_uv)용: min-max stretch"""
    a = arr.astype(np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return np.clip(a, 0, 1)


def render_signed(ax, arr, pct=99):
    """signed 컴포넌트(dl_uv, spr_uv)용: 채널 평균 후 0 중심 diverging colormap"""
    a = arr.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    vmax = np.percentile(np.abs(a), pct) + 1e-8
    im = ax.imshow(a, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    return im, vmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz_path', required=True)
    ap.add_argument('--out', default='preview.png')
    ap.add_argument('--dpi', type=int, default=180)
    ap.add_argument('--pct', type=float, default=99, help='signed 컴포넌트 vmax 계산에 쓸 percentile')
    args = ap.parse_args()

    npz = np.load(args.npz_path)
    print(f"[Info] NPZ keys: {list(npz.keys())}")

    present = [k for k in ORDER if k in npz]
    if not present:
        raise SystemExit(f"알려진 키가 하나도 없음. 실제 키: {list(npz.keys())}")

    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.0))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor('white')

    for ax, k in zip(axes, present):
        arr = npz[k]
        neg_ratio = (arr < 0).mean() * 100

        if k in SIGNED_COMPONENTS:
            im, vmax = render_signed(ax, arr, pct=args.pct)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            range_info = f"±{vmax:.3g} (p{args.pct:.0f})"
        else:
            ax.imshow(normalize_for_display(arr))
            range_info = f"[{arr.min():.3g}, {arr.max():.3g}]"

        stat = f"min={arr.min():.3g} max={arr.max():.3g} mean={arr.mean():.3g}\nneg%={neg_ratio:.1f}"
        ax.set_title(f"{TITLES[k]}\n{stat}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        print(f"[{k}] shape={arr.shape} dtype={arr.dtype} display_range={range_info} neg%={neg_ratio:.1f}")

    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi, facecolor='white')
    print(f"[Saved] {args.out}")


if __name__ == '__main__':
    main()