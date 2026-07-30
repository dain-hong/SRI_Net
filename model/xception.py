#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepfakeBench Xception 백본에서 가중치를 안전하게 잘라내어
Entry / Mid / Exit Flow로 분할 배정하는 모듈입니다.
분할 기준 (Image Branch DeepfakeBench Xception 깊이 맞춤):
  Entry : conv1~act2 + block1~3   → 728ch (Cross Attention 적용 지점)
  Mid   : block4~11               → 728ch 유지
  Exit  : block12 + conv3~act4    → 2048ch

get_deepfakebench_xception_layers(): DeepfakeBench pretrained checkpoint
(xception-b5690688.pth, Image branch가 쓰는 것과 동일한 파일)를 UV 브랜치
(tex/dl/spr)에도 이식할 수 있게 하는 함수. ImageNet pretrained(자연 이미지용
필터)가 SPR/UV 도메인과 맞지 않을 수 있다는 가설을 검증하기 위해,
"이미 딥페이크 탐지에 fine-tuning된 weight"로 시작하게 함.
"""
import torch
import torch.nn as nn


# =========================================================
# DeepfakeBench Xception 구성요소 (ImageBranchXception과 동일 구조)
# sri_net.py의 정의와 완전히 동일 - 순환 import를 피하기 위해 여기서도 정의.
# =========================================================
class _SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation,
                               groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.conv1(x))


class _Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1,
                 start_with_relu=True, grow_first=True):
        super().__init__()
        if out_filters != in_filters or strides != 1:
            self.skip   = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        relu = nn.ReLU(inplace=True)
        rep, filters = [], in_filters

        if grow_first:
            rep += [nn.ReLU(inplace=False),
                    _SeparableConv2d(in_filters, out_filters, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_filters)]
            filters = out_filters

        for _ in range(reps - 1):
            rep += [relu,
                    _SeparableConv2d(filters, filters, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(filters)]

        if not grow_first:
            rep += [relu,
                    _SeparableConv2d(in_filters, out_filters, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_filters)]

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x    = self.rep(inp)
        skip = self.skip(inp) if self.skip is not None else inp
        if self.skip is not None:
            skip = self.skipbn(skip)
        return x + skip


class _ImageBranchXception(nn.Module):
    """sri_net.py의 ImageBranchXception과 완전히 동일한 구조 (순환 import 회피용 복제)"""
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1     = nn.BatchNorm2d(32)
        self.relu    = nn.ReLU(inplace=True)
        self.conv2   = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2     = nn.BatchNorm2d(64)
        self.block1  = _Block(64,  128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2  = _Block(128, 256, 2, 2, start_with_relu=True,  grow_first=True)
        self.block3  = _Block(256, 728, 2, 2, start_with_relu=True,  grow_first=True)
        self.block4  = _Block(728, 728, 3, 1, start_with_relu=True,  grow_first=True)
        self.block5  = _Block(728, 728, 3, 1, start_with_relu=True,  grow_first=True)
        self.block6  = _Block(728, 728, 3, 1, start_with_relu=True,  grow_first=True)
        self.block7  = _Block(728, 728, 3, 1, start_with_relu=True,  grow_first=True)
        self.block8  = _Block(728, 728, 3, 1, start_with_relu=True,  grow_first=True)
        self.block9  = _Block(728, 728, 3, 1, start_with_relu=True,  grow_first=True)
        self.block10 = _Block(728, 728, 3, 1, start_with_relu=True,  grow_first=True)
        self.block11 = _Block(728, 728, 3, 1, start_with_relu=True,  grow_first=True)
        self.block12 = _Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.conv3   = _SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3     = nn.BatchNorm2d(1536)
        self.conv4   = _SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4     = nn.BatchNorm2d(2048)


def get_deepfakebench_xception_layers(pretrained_path: str):
    """
    DeepfakeBench pretrained weight(xception-b5690688.pth, Image branch와 동일 파일)를
    ImageBranchXception과 동일한 구조에 로드한 뒤,
    entry(conv1~block3) / mid(block4~11) / exit(block12~conv4)로 잘라서 반환.
    """
    model = _ImageBranchXception()

    if pretrained_path:
        ckpt = torch.load(pretrained_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        model_state  = model.state_dict()
        model_keys   = set(model_state.keys())
        filtered = {}
        for k, v in ckpt.items():
            if k not in model_keys:
                continue
            target_shape = model_state[k].shape
            if v.shape != target_shape:
                # pointwise conv 등, 체크포인트에 2D(out,in)로 저장된 weight를
                # 표준 Conv2d 4D(out,in,1,1)로 변환 (load_image_branch와 동일 처리)
                if v.dim() == 2 and target_shape == torch.Size([*v.shape, 1, 1]):
                    v = v.unsqueeze(-1).unsqueeze(-1)
                else:
                    print(f"[SPR-DeepfakeBench] shape 불일치 스킵: {k} {v.shape} vs {target_shape}")
                    continue
            filtered[k] = v
        missing = model_keys - set(filtered.keys())
        model.load_state_dict(filtered, strict=False)
        print(f"[SPR-DeepfakeBench] 로드: {len(filtered)}개 / 누락(랜덤초기화): {len(missing)}개")
        if missing:
            print(f"[SPR-DeepfakeBench] 누락 키: {sorted(missing)}")
    else:
        print("[SPR-DeepfakeBench] pretrained_path 미지정 → 랜덤 초기화")

    entry_layers = nn.Sequential(
        model.conv1, model.bn1, model.relu,
        model.conv2, model.bn2, model.relu,
        model.block1, model.block2, model.block3
    )
    mid_layers = nn.Sequential(
        model.block4, model.block5, model.block6, model.block7,
        model.block8, model.block9, model.block10, model.block11
    )
    exit_layers = nn.Sequential(
        model.block12,
        model.conv3, model.bn3, model.relu,
        model.conv4, model.bn4, model.relu
    )
    return entry_layers, mid_layers, exit_layers