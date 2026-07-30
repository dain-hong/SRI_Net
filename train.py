#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRI-Net 학습 스크립트.
- use_pair=True: real/fake 페어 배치를 concat해서 한 번의 forward로 처리
  (모델의 pre_aux_logits/uv_logits/image_logits는 페어 concat 상태 그대로 계산됨)
- 매 epoch마다 FF++ val, Celeb-DF-v1, DF40 서브셋(e4s/facedancer/uniface/simswap/inswap/fsgan) 각각 AUC 평가
- select_metric 기준으로 best.pth 저장 (기본: celeb_auc, cross-domain 목표치 91.3% AUC)
"""
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
# 유틸
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_optimizer(model, cfg_train):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg_train['optimizer'].lower() == 'adamw':
        return torch.optim.AdamW(params, lr=cfg_train['lr'], weight_decay=cfg_train['weight_decay'])
    return torch.optim.Adam(params, lr=cfg_train['lr'], weight_decay=cfg_train['weight_decay'])


def build_scheduler(optimizer, cfg_train, steps_per_epoch):
    total_steps  = cfg_train['epochs'] * steps_per_epoch
    warmup_steps = cfg_train['warmup_epochs'] * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        min_ratio = cfg_train['min_lr'] / cfg_train['lr']
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return max(min_ratio, cos)

    if cfg_train['scheduler'] == 'cosine':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return None


def move_to_device(sample_dict, device):
    out = {}
    for k, v in sample_dict.items():
        out[k] = v.to(device, non_blocking=True)
    return out


def merge_pair_batch(batch, device):
    """{'real': {...}, 'fake': {...}} -> concat batch (real 앞, fake 뒤), 반환: batch_dict, half_size"""
    real = move_to_device(batch['real'], device)
    fake = move_to_device(batch['fake'], device)
    half = real['label'].shape[0]

    merged = {}
    for k in real.keys():
        if k == 'label':
            continue
        merged[k] = torch.cat([real[k], fake[k]], dim=0)
    labels = torch.cat([real['label'], fake['label']], dim=0)
    return merged, labels, half


def contrastive_loss(embedding, half, margin, device):
    """real/fake embedding 간 cosine similarity 기반 마진 손실. real, fake는 다른 클래스이므로 유사도를 낮추도록 유도."""
    emb_real = embedding[:half]
    emb_fake = embedding[half:]
    if emb_real.shape[0] != emb_fake.shape[0]:
        n = min(emb_real.shape[0], emb_fake.shape[0])
        emb_real, emb_fake = emb_real[:n], emb_fake[:n]
    sim = nn.functional.cosine_similarity(emb_real, emb_fake, dim=1)
    loss = torch.relu(sim - margin).mean()
    return loss


def compute_losses(output, labels, cfg_train, half=None, device=None):
    ce = nn.CrossEntropyLoss()
    logits = output['logits']
    loss = ce(logits, labels)
    loss_dict = {'ce_main': loss.item()}

    aux_w = cfg_train['aux_loss_weight']
    if 'uv_logits' in output:
        l = ce(output['uv_logits'], labels)
        loss = loss + aux_w * l
        loss_dict['ce_uv'] = l.item()
    if 'image_logits' in output:
        l = ce(output['image_logits'], labels)
        loss = loss + aux_w * l
        loss_dict['ce_image'] = l.item()
    for name, logit in output.get('pre_aux_logits', {}).items():
        l = ce(logit, labels)
        loss = loss + aux_w * l
        loss_dict[f'ce_{name}'] = l.item()

    if half is not None and cfg_train['contrastive_weight'] > 0:
        c_loss = contrastive_loss(output['embedding'], half, cfg_train['contrastive_margin'], device)
        loss = loss + cfg_train['contrastive_weight'] * c_loss
        loss_dict['contrastive'] = c_loss.item()

    loss_dict['total'] = loss.item()
    return loss, loss_dict


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
    model.train()
    if not all_labels:
        return {'auc': float('nan'), 'acc': float('nan'), 'n': 0}
    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')
    acc = accuracy_score(all_labels, (all_probs >= 0.5).astype(int))
    return {'auc': auc, 'acc': acc, 'n': len(all_labels)}


def run_full_eval(model, cfg, device, keys):
    """FF++ val + Celeb-DF-v1 + DF40 서브셋별 AUC"""
    results = {}

    val_loader = get_dataloader(
        meanstd_path=cfg['data']['meanstd_path'], mode='val',
        batch_size=cfg['test']['batch_size'], num_workers=cfg['data']['num_workers'],
        use_pair=False, ff_npz_base=cfg['data']['ff_npz_base'],
        use_image=cfg['model']['use_image'], use_spr=cfg['model']['use_spr'],
        use_tex=cfg['model']['use_tex'], use_dl=cfg['model']['use_dl'], aug_use=False)
    results['ffpp_val'] = evaluate(model, val_loader, device, keys)

    celeb_loader = get_dataloader(
        meanstd_path=cfg['data']['meanstd_path'], mode='celeb',
        batch_size=cfg['test']['batch_size'], num_workers=cfg['data']['num_workers'],
        use_pair=False, celeb_json_path=cfg['data']['celeb_json_path'],
        celeb_npz_root=cfg['data']['celeb_npz_root'],
        use_image=cfg['model']['use_image'], use_spr=cfg['model']['use_spr'],
        use_tex=cfg['model']['use_tex'], use_dl=cfg['model']['use_dl'], aug_use=False)
    results['celeb'] = evaluate(model, celeb_loader, device, keys)

    for ds in cfg['data']['df40_datasets']:
        loader = get_dataloader(
            meanstd_path=cfg['data']['meanstd_path'], mode='df40',
            batch_size=cfg['test']['batch_size'], num_workers=cfg['data']['num_workers'],
            use_pair=False, df40_json_dir=cfg['data']['df40_json_dir'],
            df40_npz_root=cfg['data']['df40_npz_root'], df40_datasets=[ds],
            use_image=cfg['model']['use_image'], use_spr=cfg['model']['use_spr'],
            use_tex=cfg['model']['use_tex'], use_dl=cfg['model']['use_dl'], aug_use=False)
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
# 메인
# =========================================================
def main(cfg_path):
    cfg = load_config(cfg_path)
    set_seed(cfg['train']['seed'])

    torch.backends.cudnn.enabled = cfg['train']['cudnn_enabled']
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')

    os.makedirs(cfg['train']['save_dir'], exist_ok=True)
    keys = build_batch_keys(cfg['model'])

    # ---- 데이터 ----
    train_dataset = SRINetDataset(
        meanstd_path=cfg['data']['meanstd_path'], mode='train',
        use_pair=cfg['data']['use_pair'], ff_npz_base=cfg['data']['ff_npz_base'],
        use_image=cfg['model']['use_image'], use_spr=cfg['model']['use_spr'],
        use_tex=cfg['model']['use_tex'], use_dl=cfg['model']['use_dl'],
        aug_use=cfg['data']['aug_use'], aug_hflip_p=cfg['data']['aug_hflip_p'],
        aug_cutout_p=cfg['data']['aug_cutout_p'], aug_noise_p=cfg['data']['aug_noise_p'])

    if cfg['data']['use_pair']:
        train_loader = DataLoader(train_dataset, batch_size=cfg['data']['batch_size'],
                                   shuffle=True, num_workers=cfg['data']['num_workers'],
                                   pin_memory=True, drop_last=True)
    else:
        if cfg['data']['use_weighted_sampler']:
            sampler = train_dataset.get_sampler()
            train_loader = DataLoader(train_dataset, batch_size=cfg['data']['batch_size'],
                                       sampler=sampler, num_workers=cfg['data']['num_workers'],
                                       pin_memory=True, drop_last=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=cfg['data']['batch_size'],
                                       shuffle=True, num_workers=cfg['data']['num_workers'],
                                       pin_memory=True, drop_last=True)

    # ---- 모델 ----
    model = SRINet(cfg['model']).to(device)

    start_epoch = 1
    if cfg['train'].get('resume'):
        ckpt = torch.load(cfg['train']['resume'], map_location=device)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"[Resume] {cfg['train']['resume']} 에서 재시작 (epoch {start_epoch})")

    optimizer = build_optimizer(model, cfg['train'])
    scheduler = build_scheduler(optimizer, cfg['train'], steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['train']['amp'])

    best_metric = -1.0
    history = []

    for epoch in range(start_epoch, cfg['train']['epochs'] + 1):
        model.train()
        running = {}
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()

            if cfg['data']['use_pair']:
                merged, labels, half = merge_pair_batch(batch, device)
                model_input = {k: merged[k] for k in keys}
            else:
                labels = batch['label'].to(device)
                model_input = {k: batch[k].to(device, non_blocking=True) for k in keys}
                half = None

            with torch.cuda.amp.autocast(enabled=cfg['train']['amp']):
                output = model(model_input)
                loss, loss_dict = compute_losses(output, labels, cfg['train'], half=half, device=device)

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
                print(f"[Epoch {epoch}][{step}/{len(train_loader)}] lr={lr_now:.2e} {msg}")

        if epoch % cfg['train']['eval_interval'] == 0:
            results = run_full_eval(model, cfg, device, keys)
            print_eval(epoch, results)
            history.append({'epoch': epoch, **{k: v for k, v in results.items() if isinstance(v, dict)}})

            with open(os.path.join(cfg['train']['save_dir'], 'history.json'), 'w') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

            metric = results.get(cfg['train']['select_metric'], -1.0)
            ckpt = {'model': model.state_dict(), 'epoch': epoch, 'config': cfg, 'metric': metric}
            torch.save(ckpt, os.path.join(cfg['train']['save_dir'], 'last.pth'))
            if metric > best_metric:
                best_metric = metric
                torch.save(ckpt, os.path.join(cfg['train']['save_dir'], 'best.pth'))
                print(f"[Best] {cfg['train']['select_metric']}={metric:.4f} → best.pth 저장")

    print(f"학습 완료. best {cfg['train']['select_metric']} = {best_metric:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    main(args.config)