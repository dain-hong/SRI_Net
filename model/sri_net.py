#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from .xception import get_deepfakebench_xception_layers


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation,
                               groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.conv1(x))


class Block(nn.Module):
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
                    SeparableConv2d(in_filters, out_filters, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_filters)]
            filters = out_filters

        for _ in range(reps - 1):
            rep += [relu,
                    SeparableConv2d(filters, filters, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(filters)]

        if not grow_first:
            rep += [relu,
                    SeparableConv2d(in_filters, out_filters, 3, 1, 1, bias=False),
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


class ImageBranchXception(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1     = nn.BatchNorm2d(32)
        self.relu    = nn.ReLU(inplace=True)
        self.conv2   = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2     = nn.BatchNorm2d(64)
        self.block1  = Block(64,  128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2  = Block(128, 256, 2, 2, start_with_relu=True,  grow_first=True)
        self.block3  = Block(256, 728, 2, 2, start_with_relu=True,  grow_first=True)
        self.block4  = Block(728, 728, 3, 1, start_with_relu=True,  grow_first=True)
        self.block5  = Block(728, 728, 3, 1, start_with_relu=True,  grow_first=True)
        self.block6  = Block(728, 728, 3, 1, start_with_relu=True,  grow_first=True)
        self.block7  = Block(728, 728, 3, 1, start_with_relu=True,  grow_first=True)
        self.block8  = Block(728, 728, 3, 1, start_with_relu=True,  grow_first=True)
        self.block9  = Block(728, 728, 3, 1, start_with_relu=True,  grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True,  grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True,  grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.conv3   = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3     = nn.BatchNorm2d(1536)
        self.conv4   = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4     = nn.BatchNorm2d(2048)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.block1(x);  x = self.block2(x);  x = self.block3(x)
        x = self.block4(x);  x = self.block5(x);  x = self.block6(x)
        x = self.block7(x);  x = self.block8(x);  x = self.block9(x)
        x = self.block10(x); x = self.block11(x); x = self.block12(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = self.relu(x)
        return x


def load_image_branch(pretrained_path: str) -> ImageBranchXception:
    branch = ImageBranchXception()
    if pretrained_path:
        ckpt = torch.load(pretrained_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        branch_keys   = set(branch.state_dict().keys())
        branch_shapes = {k: v.shape for k, v in branch.state_dict().items()}
        filtered = {}
        for k, v in ckpt.items():
            if k not in branch_keys:
                continue
            target_shape = branch_shapes[k]
            if v.shape != target_shape:
                if v.dim() == 2 and target_shape == torch.Size([*v.shape, 1, 1]):
                    v = v.unsqueeze(-1).unsqueeze(-1)
                else:
                    print(f"[ImageBranch] shape 불일치 스킵: {k} {v.shape} vs {target_shape}")
                    continue
            filtered[k] = v
        missing = branch_keys - set(filtered.keys())
        branch.load_state_dict(filtered, strict=False)
        print(f"[ImageBranch] 로드: {len(filtered)}개 / 누락(랜덤초기화): {len(missing)}개")
        if missing:
            print(f"[ImageBranch] 누락 키 목록: {sorted(missing)}")
    else:
        print("[ImageBranch] pretrained_path 미지정 → 랜덤 초기화")
    return branch


class CrossAttention(nn.Module):
    def __init__(self, dim=728):
        super().__init__()
        self.scale = math.sqrt(dim)

    @staticmethod
    def _flatten_hw(x):
        b, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2)

    @staticmethod
    def _restore_hw(x, h, w):
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], h, w)

    def forward(self, q_feat, kv_feat):
        b, c, h, w = q_feat.shape
        q    = self._flatten_hw(q_feat)
        k    = self._flatten_hw(kv_feat)
        v    = self._flatten_hw(kv_feat)
        attn = torch.softmax(
            torch.bmm(q, k.transpose(1, 2)) / self.scale, dim=-1
        )
        out  = torch.bmm(attn, v)
        return self._restore_hw(out, h, w)


class SRINet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.use_image = cfg['use_image']
        self.use_spr   = cfg['use_spr']
        self.use_tex   = cfg['use_tex']
        self.use_dl    = cfg['use_dl']
        num_classes    = cfg['num_classes']
        pretrained     = cfg.get('pretrained', False)
        image_ckpt     = cfg.get('image_pretrained', None)
        dropout_p      = cfg.get('dropout', 0.5)

        self.spr_skip_mid = cfg.get('spr_skip_mid', False)

        assert self.use_image or self.use_spr or self.use_tex or self.use_dl, \
            "적어도 하나의 브랜치는 True여야 합니다."

        if self.use_image:
            self.image_branch = load_image_branch(image_ckpt)

        if self.use_tex:
            entry_tex, _, _ = get_deepfakebench_xception_layers(image_ckpt)
            self.entry_tex  = entry_tex

        if self.use_dl:
            entry_dl, _, _ = get_deepfakebench_xception_layers(image_ckpt)
            self.entry_dl  = entry_dl

        if self.use_spr:
            entry_spr, mid_spr, _ = get_deepfakebench_xception_layers(image_ckpt)
            self.entry_spr    = entry_spr
            self.mid_flow_spr = mid_spr

        self.use_cross_attn = self.use_tex and self.use_dl
        if self.use_cross_attn:
            _, mid_td, _ = get_deepfakebench_xception_layers(image_ckpt)
            self.cross_attn_1 = CrossAttention(dim=728)
            self.mid_flow_td  = mid_td

        self.use_spr_attn = self.use_spr and (self.use_tex or self.use_dl)
        if self.use_spr_attn:
            self.cross_attn_2 = CrossAttention(dim=728)

        if self.use_spr or self.use_tex or self.use_dl:
            _, _, exit_spec = get_deepfakebench_xception_layers(image_ckpt)
            self.exit_spec  = exit_spec

        self.gap = nn.AdaptiveAvgPool2d(1)

        in_dim = 0
        if self.use_image:
            in_dim += 2048
        if self.use_spr or self.use_tex or self.use_dl:
            in_dim += 2048

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes),
        )

        if self.use_spr or self.use_tex or self.use_dl:
            self.uv_aux_classifier = nn.Linear(2048, num_classes)
        if self.use_image:
            self.image_aux_classifier = nn.Linear(2048, num_classes)

        if self.use_tex:
            self.tex_pre_aux_classifier = nn.Linear(728, num_classes)
        if self.use_dl:
            self.dl_pre_aux_classifier = nn.Linear(728, num_classes)
        if self.use_spr:
            self.spr_pre_aux_classifier = nn.Linear(728, num_classes)

        if self.use_cross_attn:
            self.td_mid_pre_aux_classifier = nn.Linear(728, num_classes)

        if self.use_spr_attn:
            self.std_pre_aux_classifier = nn.Linear(728, num_classes)

    def forward(self, batch):
        feats = []
        pre_aux_logits = {}
        branch_feats = {}

        if self.use_tex and self.use_dl:
            f_tex = self.entry_tex(batch['tex'])
            f_dl  = self.entry_dl(batch['dl'])

            tex_feat_detached = self.gap(f_tex).flatten(1).detach()
            dl_feat_detached  = self.gap(f_dl).flatten(1).detach()
            pre_aux_logits['tex'] = self.tex_pre_aux_classifier(tex_feat_detached)
            pre_aux_logits['dl']  = self.dl_pre_aux_classifier(dl_feat_detached)
            branch_feats['tex'] = tex_feat_detached
            branch_feats['dl']  = dl_feat_detached

            attn_td  = self.cross_attn_1(f_tex, f_dl)
            f_td     = attn_td + f_tex + f_dl
            f_td_mid = self.mid_flow_td(f_td)

            td_mid_detached = self.gap(f_td_mid).flatten(1).detach()
            pre_aux_logits['td_mid_post_attn'] = self.td_mid_pre_aux_classifier(td_mid_detached)
            branch_feats['td_mid_post_attn'] = td_mid_detached

            if self.use_spr:
                f_spr_entry = self.entry_spr(batch['spr'])
                spr_feat_detached = self.gap(f_spr_entry).flatten(1).detach()
                pre_aux_logits['spr'] = self.spr_pre_aux_classifier(spr_feat_detached)
                branch_feats['spr'] = spr_feat_detached

                f_spr    = self.mid_flow_spr(f_spr_entry)
                attn_std = self.cross_attn_2(f_td_mid, f_spr)
                f_std    = attn_std + f_spr

                std_detached = self.gap(f_std).flatten(1).detach()
                pre_aux_logits['std_post_attn'] = self.std_pre_aux_classifier(std_detached)
                branch_feats['std_post_attn'] = std_detached
            else:
                f_std = f_td_mid

        elif self.use_spr and (self.use_tex or self.use_dl):
            if self.use_tex:
                f_ctx = self.entry_tex(batch['tex'])
                ctx_feat_detached = self.gap(f_ctx).flatten(1).detach()
                pre_aux_logits['tex'] = self.tex_pre_aux_classifier(ctx_feat_detached)
                branch_feats['tex'] = ctx_feat_detached
            else:
                f_ctx = self.entry_dl(batch['dl'])
                ctx_feat_detached = self.gap(f_ctx).flatten(1).detach()
                pre_aux_logits['dl'] = self.dl_pre_aux_classifier(ctx_feat_detached)
                branch_feats['dl'] = ctx_feat_detached

            f_spr_entry = self.entry_spr(batch['spr'])
            spr_feat_detached = self.gap(f_spr_entry).flatten(1).detach()
            pre_aux_logits['spr'] = self.spr_pre_aux_classifier(spr_feat_detached)
            branch_feats['spr'] = spr_feat_detached

            f_spr    = self.mid_flow_spr(f_spr_entry)
            attn_std = self.cross_attn_2(f_ctx, f_spr)
            f_std    = attn_std + f_spr

            std_detached = self.gap(f_std).flatten(1).detach()
            pre_aux_logits['std_post_attn'] = self.std_pre_aux_classifier(std_detached)
            branch_feats['std_post_attn'] = std_detached

        elif self.use_spr:
            f_spr_entry = self.entry_spr(batch['spr'])
            spr_feat_detached = self.gap(f_spr_entry).flatten(1).detach()
            pre_aux_logits['spr'] = self.spr_pre_aux_classifier(spr_feat_detached)
            branch_feats['spr'] = spr_feat_detached
            if self.spr_skip_mid:
                f_std = f_spr_entry
            else:
                f_std = self.mid_flow_spr(f_spr_entry)

        elif self.use_tex or self.use_dl:
            if self.use_tex:
                f_std = self.entry_tex(batch['tex'])
                std_feat_detached = self.gap(f_std).flatten(1).detach()
                pre_aux_logits['tex'] = self.tex_pre_aux_classifier(std_feat_detached)
                branch_feats['tex'] = std_feat_detached
            else:
                f_std = self.entry_dl(batch['dl'])
                std_feat_detached = self.gap(f_std).flatten(1).detach()
                pre_aux_logits['dl'] = self.dl_pre_aux_classifier(std_feat_detached)
                branch_feats['dl'] = std_feat_detached

        uv_feat_raw = None
        if self.use_spr or self.use_tex or self.use_dl:
            uv_feat_raw = torch.flatten(self.gap(self.exit_spec(f_std)), 1)
            feats.append(nn.functional.normalize(uv_feat_raw, dim=1))
            branch_feats['uv_final'] = uv_feat_raw.detach()

        image_feat_raw = None
        if self.use_image:
            image_feat_raw = torch.flatten(self.gap(self.image_branch(batch['image'])), 1)
            feats.append(nn.functional.normalize(image_feat_raw, dim=1))
            branch_feats['image_final'] = image_feat_raw.detach()

        fused = torch.cat(feats, dim=1)
        x = self.classifier[0](fused)
        x = self.classifier[1](x)
        embedding = x
        x = self.classifier[2](x)
        logits = self.classifier[3](x)

        output = {'logits': logits, 'embedding': embedding}
        if uv_feat_raw is not None:
            output['uv_logits'] = self.uv_aux_classifier(uv_feat_raw)
        if image_feat_raw is not None:
            output['image_logits'] = self.image_aux_classifier(image_feat_raw)
        output['pre_aux_logits'] = pre_aux_logits
        output['branch_feats'] = branch_feats

        return output