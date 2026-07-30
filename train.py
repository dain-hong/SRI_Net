#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import random

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from model.sri_net import SRINet
from data.dataset import SRINetDataset, get_dataloader


# =========================================================
# 설정 / 시드
# =========================================================
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def branch_kwargs(cfg_model):
    """use_image/use_spr/use_tex/use_dl 묶음 — dataset/model 양쪽에서 재사용."""
    return {
        'use_image': cfg_model['use_image'],
        'use_spr': cfg_model['use_spr'],
        'use_tex': cfg_model['use_tex'],
        'use_dl': cfg_model['use_dl'],
    }


def batch_keys(cfg_model):
    """활성화된 브랜치에 대응하는 배치 키 목록 ('image'/'tex'/'dl'/'spr')."""
    name_map = [('image', 'use_image'), ('tex', 'use_tex'), ('dl', 'use_dl'), ('spr', 'use_spr')]
    return [key for key, flag in name_map if cfg_model[flag]]


# =========================================================
# 데이터로더
# =========================================================
def build_train_loader(cfg):
    dataset = SRINetDataset(
        meanstd_path=cfg['data']['meanstd_path'], mode='train',
        ff_npz_base=cfg['data']['ff_npz_base'],
        aug_use=cfg['data']['aug_use'], aug_hflip_p=cfg['data']['aug_hflip_p'],
        aug_cutout_p=cfg['data']['aug_cutout_p'], aug_noise_p=cfg['data']['aug_noise_p'],
        **branch_kwargs(cfg['model']))

    common = dict(batch_size=cfg['data']['batch_size'], num_workers=cfg['data']['num_workers'],
                   pin_memory=True, drop_last=True)

    if cfg['data']['use_weighted_sampler']:
        return DataLoader(dataset, sampler=dataset.get_sampler(), **common)
    return DataLoader(dataset, shuffle=True, **common)


def build_eval_loader(cfg, mode, **extra):
    return get_dataloader(
        meanstd_path=cfg['data']['meanstd_path'], mode=mode,
        batch_size=cfg['test']['batch_size'], num_workers=cfg['data']['num_workers'],
        aug_use=False,
        **branch_kwargs(cfg['model']), **extra)


# =========================================================
# 배치 처리 / 손실
# =========================================================
def compute_losses(output, labels, cfg_train):
    ce = nn.CrossEntropyLoss()
    loss = ce(output['logits'], labels)
    loss_dict = {'ce_main': loss.item()}

    aux_w = cfg_train['aux_loss_weight']
    aux_logits = {k: output[k] for k in ('uv_logits', 'image_logits') if k in output}
    aux_logits.update({f'pre_{k}': v for k, v in output.get('pre_aux_logits', {}).items()})
    for name, logit in aux_logits.items():
        l = ce(logit, labels)
        loss = loss + aux_w * l
        loss_dict[f'ce_{name}'] = l.item()

    loss_dict['total'] = loss.item()
    return loss, loss_dict


# =========================================================
# 최적화
# =========================================================
def build_optimizer(model, cfg_train):
    params = [p for p in model.parameters() if p.requires_grad]
    opt_cls = torch.optim.AdamW if cfg_train['optimizer'].lower() == 'adamw' else torch.optim.Adam
    return opt_cls(params, lr=cfg_train['lr'], weight_decay=cfg_train['weight_decay'])


def build_scheduler(optimizer, cfg_train, steps_per_epoch):
    if cfg_train['scheduler'] != 'cosine':
        return None

    total_steps = cfg_train['epochs'] * steps_per_epoch
    warmup_steps = cfg_train['warmup_epochs'] * steps_per_epoch
    min_ratio = cfg_train['min_lr'] / cfg_train['lr']

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_ratio, 0.5 * (1 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =========================================================
# 평가
# =========================================================
@torch.no_grad()
def evaluate(model, dataloader, device, keys):
    model.eval()
    all_labels, all_probs = [], []
    for batch in dataloader:
        labels = batch['label'].to(device)
        model_input = {k: batch[k].to(device, non_blocking=True) for k in keys}
        probs = torch.softmax(model(model_input)['logits'], dim=1)[:, 1]
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    model.train()

    if not all_labels:
        return {'auc': float('nan'), 'acc': float('nan'), 'n': 0}

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')
    acc = accuracy_score(all_labels, (all_probs >= 0.5).astype(int))
    return {'auc': auc, 'acc': acc, 'n': len(all_labels)}


def run_full_eval(model, cfg, device, keys):
    """FF++ val + Celeb-DF-v1 + DF40 서브셋별 AUC. df40_datasets가 비어있으면 DF40은 건너뜀."""
    results = {
        'ffpp_val': evaluate(model, build_eval_loader(
            cfg, 'val', ff_npz_base=cfg['data']['ff_npz_base']), device, keys),
        'celeb': evaluate(model, build_eval_loader(
            cfg, 'celeb', celeb_json_path=cfg['data']['celeb_json_path'],
            celeb_npz_root=cfg['data']['celeb_npz_root']), device, keys),
    }
    for ds in cfg['data']['df40_datasets']:
        loader = build_eval_loader(
            cfg, 'df40', df40_json_dir=cfg['data']['df40_json_dir'],
            df40_npz_root=cfg['data']['df40_npz_root'], df40_datasets=[ds])
        results[f'df40_{ds}'] = evaluate(model, loader, device, keys)

    results['ffpp_val_auc'] = results['ffpp_val']['auc']
    results['celeb_auc'] = results['celeb']['auc']
    return results


def print_eval(epoch, results):
    print(f"\n===== Epoch {epoch} Evaluation =====")
    for name, r in results.items():
        if isinstance(r, dict):
            print(f"  {name:14s}  AUC={r['auc']:.4f}  ACC={r['acc']:.4f}  (n={r['n']})")
    print("=" * 40)


# =========================================================
# 체크포인트
# =========================================================
def save_checkpoint(model, epoch, cfg, metric, path):
    torch.save({'model': model.state_dict(), 'epoch': epoch, 'config': cfg, 'metric': metric}, path)


def load_checkpoint_if_resume(model, cfg_train, device):
    if not cfg_train.get('resume'):
        return 1
    ckpt = torch.load(cfg_train['resume'], map_location=device)
    model.load_state_dict(ckpt['model'])
    start_epoch = ckpt.get('epoch', 0) + 1
    print(f"[Resume] {cfg_train['resume']} 에서 재시작 (epoch {start_epoch})")
    return start_epoch


# =========================================================
# 학습 루프
# =========================================================
def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, cfg, device, keys, epoch):
    model.train()
    running = {}
    n_steps = len(train_loader)

    for step, batch in enumerate(train_loader, 1):
        optimizer.zero_grad()

        labels = batch['label'].to(device)
        model_input = {k: batch[k].to(device, non_blocking=True) for k in keys}

        with torch.amp.autocast('cuda', enabled=cfg['train']['amp']):
            output = model(model_input)
            loss, loss_dict = compute_losses(output, labels, cfg['train'])

        scaler.scale(loss).backward()
        if cfg['train']['grad_clip_norm']:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip_norm'])
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        for k, v in loss_dict.items():
            running[k] = running.get(k, 0.0) + v

        if step % cfg['train']['log_interval'] == 0:
            avg = {k: v / step for k, v in running.items()}
            lr_now = optimizer.param_groups[0]['lr']
            msg = " ".join(f"{k}={v:.4f}" for k, v in avg.items())
            print(f"[Epoch {epoch}][{step}/{n_steps}] lr={lr_now:.2e} {msg}")


def maybe_eval_and_checkpoint(model, cfg, device, keys, epoch, history, best_metric):
    if epoch % cfg['train']['eval_interval'] != 0:
        return best_metric

    results = run_full_eval(model, cfg, device, keys)
    print_eval(epoch, results)
    history.append({'epoch': epoch, **{k: v for k, v in results.items() if isinstance(v, dict)}})

    save_dir = cfg['train']['save_dir']
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    metric = results.get(cfg['train']['select_metric'], -1.0)
    save_checkpoint(model, epoch, cfg, metric, os.path.join(save_dir, 'last.pth'))
    if metric > best_metric:
        best_metric = metric
        save_checkpoint(model, epoch, cfg, metric, os.path.join(save_dir, 'best.pth'))
        print(f"[Best] {cfg['train']['select_metric']}={metric:.4f} → best.pth 저장")

    return best_metric


def main(cfg_path):
    cfg = load_config(cfg_path)
    set_seed(cfg['train']['seed'])
    torch.backends.cudnn.enabled = cfg['train']['cudnn_enabled']
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg['train']['save_dir'], exist_ok=True)

    keys = batch_keys(cfg['model'])
    train_loader = build_train_loader(cfg)

    model = SRINet(cfg['model']).to(device)
    start_epoch = load_checkpoint_if_resume(model, cfg['train'], device)

    optimizer = build_optimizer(model, cfg['train'])
    scheduler = build_scheduler(optimizer, cfg['train'], steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['train']['amp'])

    best_metric = -1.0
    history = []

    for epoch in range(start_epoch, cfg['train']['epochs'] + 1):
        train_one_epoch(model, train_loader, optimizer, scheduler, scaler, cfg, device, keys, epoch)
        best_metric = maybe_eval_and_checkpoint(model, cfg, device, keys, epoch, history, best_metric)

    print(f"학습 완료. best {cfg['train']['select_metric']} = {best_metric:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    main(args.config)
