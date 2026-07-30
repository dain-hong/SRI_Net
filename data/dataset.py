#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import glob
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

def df40_rel_to_npz(frame_rel, npz_root, dataset_name):
    clean_path = frame_rel.replace('\\', '/')
    parts = clean_path.split('frames/')
    sub_path = parts[-1] if len(parts) > 1 else clean_path.split('/')[-1]
    sub_path = sub_path.replace('.png', '.npz')
    return os.path.join(npz_root, dataset_name, sub_path)

def celeb_rel_to_npz(frame_rel, npz_root, is_fake):
    clean = frame_rel.replace('//', '/').replace('\\', '/')
    vid_id     = clean.split('/')[-2]
    frame_name = os.path.basename(clean).replace('.png', '.npz')
    return os.path.join(npz_root, 'test', str(is_fake), vid_id, frame_name)

def apply_jpeg_compression(img_np, quality_range=(30, 95)):
    import cv2
    quality = random.randint(*quality_range)
    img_u8 = np.clip(img_np, 0, 255).astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR), encode_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32)

class SRINetDataset(Dataset):
    DF40_CONFIG = {
        'e4s':        {'json': 'e4s_ff.json',        'top_key': 'e4s_ff',        'real_key': 'e4s_Real',        'fake_key': 'e4s_Fake'},
        'facedancer': {'json': 'facedancer_ff.json', 'top_key': 'facedancer_ff', 'real_key': 'facedancer_Real', 'fake_key': 'facedancer_Fake'},
        'uniface':    {'json': 'uniface_ff.json',    'top_key': 'uniface_ff',    'real_key': 'uniface_Real',    'fake_key': 'uniface_Fake'},
        'simswap':    {'json': 'simswap_ff.json',    'top_key': 'simswap_ff',    'real_key': 'simswap_Real',    'fake_key': 'simswap_Fake'},
        'inswap':     {'json': 'inswap_ff.json',     'top_key': 'inswap_ff',     'real_key': 'inswap_Real',     'fake_key': 'inswap_Fake'},
        'fsgan':      {'json': 'fsgan_ff.json',      'top_key': 'fsgan_ff',      'real_key': 'fsgan_Real',      'fake_key': 'fsgan_Fake'},
    }

    def __init__(self, meanstd_path, mode='train', use_pair=True, ff_json_path=None, ff_npz_base=None,
                 df40_json_dir=None, df40_npz_root=None, df40_datasets=None,
                 celeb_json_path=None, celeb_npz_root=None,
                 use_image=True, use_spr=True, use_tex=True, use_dl=True,
                 aug_use=True, aug_hflip_p=0.5, aug_cutout_p=0.5, aug_noise_p=0.3):

        self.mode         = mode
        self.use_pair     = use_pair
        self.use_image    = use_image
        self.use_spr      = use_spr
        self.use_tex      = use_tex
        self.use_dl       = use_dl
        self.aug_use      = aug_use
        self.aug_hflip_p  = aug_hflip_p
        self.aug_cutout_p = aug_cutout_p
        self.aug_noise_p  = aug_noise_p

        with open(meanstd_path, 'r') as f:
            ms = json.load(f)

        self.normalizers = {}
        if use_image:
            self.normalizers['image'] = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        if use_tex:
            self.normalizers['tex']   = transforms.Normalize(mean=ms['tex_uv']['mean'], std=ms['tex_uv']['std'])
        if use_dl:
            self.normalizers['dl']    = transforms.Normalize(mean=ms['dl_uv']['mean'], std=ms['dl_uv']['std'])
        if use_spr:
            self.normalizers['spr']   = transforms.Normalize(mean=ms['spr_uv']['mean'], std=ms['spr_uv']['std'])

        if mode in ('train', 'val'):
            raw_samples = self._from_folder_structure(ff_npz_base)
        elif mode == 'celeb':
            raw_samples = self._from_celeb_json(celeb_json_path, celeb_npz_root)
        else:
            raw_samples = self._from_df40_json(df40_json_dir, df40_npz_root, df40_datasets)

        if mode == 'train' and self.use_pair:
            self.samples = self._build_pairs(raw_samples)
            print(f"[{mode}] Loaded: {len(self.samples)} Pairs for Contrastive Learning")
            self._pair_example_logged = False
        else:
            self.samples = raw_samples
            print(f"[{mode}] Loaded: Real={sum(1 for _,l in self.samples if l==0)}, "
                  f"Fake={sum(1 for _,l in self.samples if l==1)}, Total={len(self.samples)}"
                  + (" (flat, no pairing)" if mode == 'train' else ""))

    def _build_pairs(self, raw_samples):
        real_dict = {}
        fake_list = []
        paired_samples = []

        print("🔄 진짜-가짜 짝꿍(Pair) 매칭 진행 중...")

        for path, label in raw_samples:
            if label == 0:
                parts = path.replace('\\', '/').split('/')
                vid_id = parts[-2]
                frame_id = parts[-1]
                real_dict[f"{vid_id}_{frame_id}"] = path
            else:
                fake_list.append(path)

        for fake_path in fake_list:
            parts = fake_path.replace('\\', '/').split('/')
            vid_id_pair = parts[-2]
            frame_id = parts[-1]

            if '_' in vid_id_pair:
                source_id, target_id = vid_id_pair.split('_')
                target_key = f"{target_id}_{frame_id}"
                source_key = f"{source_id}_{frame_id}"

                if target_key in real_dict:
                    paired_samples.append((real_dict[target_key], fake_path))
                elif source_key in real_dict:
                    paired_samples.append((real_dict[source_key], fake_path))
            else:
                real_key = f"{vid_id_pair}_{frame_id}"
                if real_key in real_dict:
                    paired_samples.append((real_dict[real_key], fake_path))

        return paired_samples

    def get_sampler(self):
        if self.mode == 'train' and self.use_pair:
            raise ValueError("Pair 학습 시에는 클래스가 1:1로 고정되므로 WeightedRandomSampler를 사용할 필요가 없습니다.")
        labels = [l for _, l in self.samples]
        class_counts = np.bincount(labels)
        weights = 1.0 / class_counts[labels]
        return WeightedRandomSampler(
            weights=torch.from_numpy(weights).float(),
            num_samples=len(self.samples),
            replacement=True
        )

    def _from_folder_structure(self, npz_base):
        cache_path = os.path.join(npz_base, '_file_list_cache.json')

        if os.path.exists(cache_path):
            print(f"[Cache] 파일 목록 캐시 로드: {cache_path}")
            with open(cache_path, 'r') as f:
                samples = [tuple(x) for x in json.load(f)]
            print(f"[Cache] 캐시에서 {len(samples)}개 샘플 로드 완료")
            return samples

        print(f"[Cache] 캐시 없음 → 전체 스캔 시작: {npz_base}")
        all_files = glob.glob(os.path.join(npz_base, '**/*.npz'), recursive=True)
        samples = []
        for p in all_files:
            label = 1 if any(x in p.lower() for x in ['fake', 'swap', 'nt', 'f2f']) else 0
            samples.append((p, label))

        try:
            with open(cache_path, 'w') as f:
                json.dump(samples, f)
            print(f"[Cache] 스캔 완료 및 캐시 저장: {len(samples)}개 → {cache_path}")
        except OSError as e:
            print(f"[Cache] 캐시 저장 실패 (읽기 전용 경로 등): {e} → 캐시 없이 계속 진행")

        return samples

    def _from_celeb_json(self, json_path, npz_root):
        samples = []
        with open(json_path, 'r') as f:
            data = json.load(f)['Celeb-DF-v1']
        for split_key in ['CelebDFv1_real', 'CelebDFv1_fake']:
            label = 1 if split_key == 'CelebDFv1_fake' else 0
            test_data = data[split_key].get('test', {})
            for vid_id, vid_info in test_data.items():
                for frame_rel in vid_info.get('frames', []):
                    p = celeb_rel_to_npz(frame_rel, npz_root, label)
                    if os.path.exists(p):
                        samples.append((p, label))
        return samples

    def _from_df40_json(self, json_dir, npz_root, datasets):
        samples = []
        for ds_name in datasets:
            if ds_name == 'real' or ds_name not in self.DF40_CONFIG: continue
            cfg = self.DF40_CONFIG[ds_name]
            json_path = os.path.join(json_dir, cfg['json'])
            if not os.path.exists(json_path): continue
            with open(json_path, 'r') as f:
                data = json.load(f)[cfg['top_key']]
            for mode_key in [cfg['real_key'], cfg['fake_key']]:
                if mode_key and mode_key in data:
                    label = 0 if 'Real' in mode_key else 1
                    target_data = data[mode_key].get('test', data[mode_key].get('train', {}))
                    for vid_id, vid_data in target_data.items():
                        frames = vid_data.get('frames', []) if 'frames' in vid_data else []
                        if not frames:
                            for sub_vid, sub_data in vid_data.items():
                                frames.extend(sub_data.get('frames', []))
                        for fr in frames:
                            ds_folder = 'real' if label == 0 else ds_name
                            p = df40_rel_to_npz(fr, npz_root, ds_folder)
                            if os.path.exists(p): samples.append((p, label))
        return list(set(samples))

    def _augment_image(self, img_np, do_flip):
        if do_flip:
            img_np = img_np[:, ::-1, :].copy()

        if random.random() < 0.5:
            img_np = apply_jpeg_compression(img_np, quality_range=(30, 95))

        if random.random() < 0.5:
            img_t = torch.from_numpy(img_np.transpose(2, 0, 1) / 255.0).float()
            img_t = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)(img_t)
            img_np = (img_t.numpy().transpose(1, 2, 0) * 255.0).astype(np.float32)

        if random.random() < self.aug_noise_p:
            noise = np.random.randn(*img_np.shape).astype(np.float32) * 5.0
            img_np = np.clip(img_np + noise, 0, 255)

        if random.random() < self.aug_cutout_p:
            h, w = img_np.shape[:2]
            cx, cy = random.randint(0, w), random.randint(0, h)
            bw, bh = random.randint(20, 60), random.randint(20, 60)
            x1, x2 = max(0, cx - bw//2), min(w, cx + bw//2)
            y1, y2 = max(0, cy - bh//2), min(h, cy + bh//2)
            img_np[y1:y2, x1:x2, :] = 0

        return img_np

    def __len__(self):
        return len(self.samples)

    def _load_single_npz(self, npz_path, do_flip):
        npz = np.load(npz_path)
        img_np = None

        if self.use_image and self.mode == 'train' and self.aug_use:
            img_np = self._augment_image(npz['image'].copy(), do_flip=do_flip)

        active = []
        key_map = [('image', 'image', self.use_image), ('tex', 'tex_uv', self.use_tex),
                   ('dl', 'dl_uv', self.use_dl), ('spr', 'spr_uv', self.use_spr)]

        for b_key, n_key, use in key_map:
            if use and n_key in npz:
                if b_key == 'image':
                    arr = img_np if img_np is not None else npz[n_key]
                    t = torch.from_numpy(arr.transpose(2, 0, 1).astype(np.float32)) / 255.0
                else:
                    arr = npz[n_key].astype(np.float32)
                    if do_flip:
                        arr = arr[:, ::-1, :].copy()
                    t = torch.from_numpy(arr.transpose(2, 0, 1))
                active.append((b_key, t))

        sample = {k: self.normalizers[k](t) for k, t in active}
        return sample

    def __getitem__(self, idx):
        try:
            if self.mode == 'train' and self.use_pair:
                real_path, fake_path = self.samples[idx]
                do_flip = self.aug_use and random.random() < self.aug_hflip_p

                real_sample = self._load_single_npz(real_path, do_flip)
                fake_sample = self._load_single_npz(fake_path, do_flip)

                real_sample['label'] = torch.tensor(0, dtype=torch.long)
                fake_sample['label'] = torch.tensor(1, dtype=torch.long)

                if not self._pair_example_logged:
                    print(f"[Pair Sample Check] REAL: {real_path}  <->  FAKE: {fake_path}")
                    self._pair_example_logged = True

                return {'real': real_sample, 'fake': fake_sample}
            else:
                npz_path, label = self.samples[idx]
                do_flip = self.aug_use and self.mode == 'train' and random.random() < self.aug_hflip_p
                sample = self._load_single_npz(npz_path, do_flip=do_flip)
                sample['label'] = torch.tensor(label, dtype=torch.long)
                return sample

        except Exception as e:
            import traceback
            print(f"[SRINetDataset] 샘플 로드 실패 (idx={idx})")
            print(f"  에러: {type(e).__name__}: {e}")
            if not hasattr(self, '_fallback_count'): self._fallback_count = 0
            self._fallback_count += 1
            if self._fallback_count > 20:
                raise RuntimeError("샘플 로드 연속 실패")
            return self.__getitem__((idx + 1) % len(self.samples))


def get_dataloader(meanstd_path, mode='train', batch_size=32, num_workers=4,
                   use_weighted_sampler=False, use_pair=True, **kwargs):
    dataset = SRINetDataset(meanstd_path=meanstd_path, mode=mode, use_pair=use_pair, **kwargs)

    if mode == 'train' and use_pair:
        if use_weighted_sampler:
            print("⚠️ Train 모드에서는 Pair 구성으로 인해 완벽한 클래스 밸런스(50:50)가 보장되므로, WeightedRandomSampler를 무시하고 일반 Shuffle을 사용합니다.")
        return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)

    if mode == 'train' and not use_pair:
        if use_weighted_sampler:
            sampler = dataset.get_sampler()
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)

    if use_weighted_sampler:
        sampler = dataset.get_sampler()
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=True, drop_last=False)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=False)