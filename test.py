#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
학습된 SRI-Net 체크포인트로 FF++ val / Celeb-DF-v1 / DF40 전 서브셋 평가.
결과는 콘솔 출력 + (설정 시) JSON 저장.
"""
import os
import json
import argparse

import yaml
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score

from model.sri_net import SRINet
from data.dataset import get_dataloader


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_batch_keys(cfg_model):
    keys = []
    if cfg_model['use_image']: keys.append('image')
    if cfg_model['use_tex']:   keys.append('tex')
    if cfg_model['use_dl']:    keys.append('dl')
    if cfg_model['use_spr']:   keys.append('spr')
    return keys


@torch.no_grad()
def evaluate(model, dataloader, device, keys):
    model.eval()
    all_labels, all_probs = [], []
    for batch in dataloader:
        labels = batch['label'].to(device)
        model_input = {k: batch[k].to(device, non_blocking=True) for k in keys}
        output = model(model_input)
        probs = torch.softmax(output['logits'], dim=1)[:, 1]
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    if not all_labels:
        return {'auc': float('nan'), 'acc': float('nan'), 'n': 0}
    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')
    acc = accuracy_score(all_labels, (all_probs >= 0.5).astype(int))
    return {'auc': float(auc), 'acc': float(acc), 'n': int(len(all_labels))}


def main(cfg_path, ckpt_override=None):
    cfg = load_config(cfg_path)
    torch.backends.cudnn.enabled = cfg['train']['cudnn_enabled']
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')
    keys = build_batch_keys(cfg['model'])

    ckpt_path = ckpt_override or cfg['test']['ckpt_path']
    model = SRINet(cfg['model']).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"[Test] 체크포인트 로드: {ckpt_path}"
          + (f" (epoch {ckpt.get('epoch')})" if isinstance(ckpt, dict) and 'epoch' in ckpt else ""))

    results = {}

    val_loader = get_dataloader(
        meanstd_path=cfg['data']['meanstd_path'], mode='val',
        batch_size=cfg['test']['batch_size'], num_workers=cfg['test']['num_workers'],
        ff_npz_base=cfg['data']['ff_npz_base'],
        use_image=cfg['model']['use_image'], use_spr=cfg['model']['use_spr'],
        use_tex=cfg['model']['use_tex'], use_dl=cfg['model']['use_dl'], aug_use=False)
    results['ffpp_val'] = evaluate(model, val_loader, device, keys)

    celeb_loader = get_dataloader(
        meanstd_path=cfg['data']['meanstd_path'], mode='celeb',
        batch_size=cfg['test']['batch_size'], num_workers=cfg['test']['num_workers'],
        celeb_json_path=cfg['data']['celeb_json_path'],
        celeb_npz_root=cfg['data']['celeb_npz_root'],
        use_image=cfg['model']['use_image'], use_spr=cfg['model']['use_spr'],
        use_tex=cfg['model']['use_tex'], use_dl=cfg['model']['use_dl'], aug_use=False)
    results['celeb'] = evaluate(model, celeb_loader, device, keys)

    for ds in cfg['data']['df40_datasets']:
        loader = get_dataloader(
            meanstd_path=cfg['data']['meanstd_path'], mode='df40',
            batch_size=cfg['test']['batch_size'], num_workers=cfg['test']['num_workers'],
            df40_json_dir=cfg['data']['df40_json_dir'],
            df40_npz_root=cfg['data']['df40_npz_root'], df40_datasets=[ds],
            use_image=cfg['model']['use_image'], use_spr=cfg['model']['use_spr'],
            use_tex=cfg['model']['use_tex'], use_dl=cfg['model']['use_dl'], aug_use=False)
        results[f'df40_{ds}'] = evaluate(model, loader, device, keys)

    print("\n===== Test Results =====")
    for name, r in results.items():
        print(f"  {name:14s}  AUC={r['auc']:.4f}  ACC={r['acc']:.4f}  (n={r['n']})")
    print("=" * 30)

    df40_aucs = [results[f'df40_{ds}']['auc'] for ds in cfg['data']['df40_datasets']]
    results['df40_mean_auc'] = float(np.nanmean(df40_aucs))
    print(f"  {'df40_mean':14s}  AUC={results['df40_mean_auc']:.4f}")

    save_path = cfg['test'].get('save_json')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[Test] 결과 저장: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--ckpt', type=str, default=None, help='config의 test.ckpt_path 대신 사용할 체크포인트 경로')
    args = parser.parse_args()
    main(args.config, args.ckpt)
