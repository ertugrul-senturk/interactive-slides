"""Visual Prompt Tuning (Jia et al., ECCV 2022) on a frozen timm ViT.

One class covers three settings, so training and inference share the same code:
  num_prompts = 0            -> linear probe (frozen backbone + linear head)
  num_prompts > 0, deep=False -> VPT-shallow (prompts before layer 1 only)
  num_prompts > 0, deep=True  -> VPT-deep (fresh prompts before every layer)
"""
import math

import timm
import torch
import torch.nn as nn

BACKBONE = "vit_base_patch16_224.orig_in21k"  # ViT-B/16, ImageNet-21k supervised, as in the paper


class VPTViT(nn.Module):
    def __init__(self, num_classes, num_prompts=50, deep=True, prompt_dropout=0.1,
                 backbone=BACKBONE, pretrained=True, vit=None):
        super().__init__()
        # Pass `vit` to share one frozen backbone between several tasks or methods.
        self.vit = vit if vit is not None else timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        for p in self.vit.parameters():
            p.requires_grad = False

        d = self.vit.embed_dim
        n_layers = len(self.vit.blocks)
        self.num_prompts, self.deep = num_prompts, deep
        self.prompt_dropout = nn.Dropout(prompt_dropout)

        if num_prompts > 0:
            # Xavier-uniform as in the official code: fan_in = 3*patch*patch, fan_out = d
            patch = self.vit.patch_embed.patch_size[0]
            bound = math.sqrt(6.0 / (3 * patch * patch + d))
            self.prompts = nn.Parameter(torch.empty(n_layers if deep else 1, num_prompts, d).uniform_(-bound, bound))
        else:
            self.register_parameter("prompts", None)
        self.head = nn.Linear(d, num_classes)

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()  # the backbone stays in eval mode: no dropout or drop-path inside it
        return self

    def trainable_state_dict(self):
        """Just what a task needs to store: the prompts and the head."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("vit.")}

    def forward_features(self, x):
        v, p = self.vit, self.num_prompts
        x = v.patch_embed(x)
        x = v._pos_embed(x)          # prepends [CLS] and adds position embeddings
        x = v.norm_pre(x)
        B = x.shape[0]
        for i, blk in enumerate(v.blocks):
            if p and (i == 0 or self.deep):
                P = self.prompt_dropout(self.prompts[i if self.deep else 0]).expand(B, -1, -1)
                # Layer 1: [CLS, E] -> [CLS, P, E].  Deeper layers: replace the previous prompt outputs.
                rest = x[:, 1:] if i == 0 else x[:, 1 + p:]
                x = torch.cat([x[:, :1], P, rest], dim=1)
            x = blk(x)
        x = v.norm(x)
        return x[:, 0]               # final [CLS] embedding

    def forward(self, x):
        return self.head(self.forward_features(x))


def count(model):
    backbone = sum(p.numel() for p in model.vit.parameters())
    prompts = model.prompts.numel() if model.prompts is not None else 0
    head = sum(p.numel() for p in model.head.parameters())
    return {"backbone": backbone, "prompts": prompts, "head": head,
            "trainable": prompts + head, "trainable_pct": 100 * (prompts + head) / backbone}


class MultiTaskVPT(nn.Module):
    """One frozen ViT shared by several VPT-deep tasks. Each image in a batch carries its own task's prompts,
    so a mixed batch (say a skin lesion, a blood smear and an X-ray) runs in a single forward pass."""

    def __init__(self, vit):
        super().__init__()
        self.vit = vit
        self.prompts = nn.ParameterDict()
        self.heads = nn.ModuleDict()

    def add_task(self, name, state, num_classes):
        """`state` is a VPTViT.trainable_state_dict(): {'prompts': (N, p, d), 'head.weight', 'head.bias'}."""
        self.prompts[name] = nn.Parameter(state["prompts"], requires_grad=False)
        head = nn.Linear(self.vit.embed_dim, num_classes)
        head.load_state_dict({"weight": state["head.weight"], "bias": state["head.bias"]})
        self.heads[name] = head

    def forward(self, x, tasks):
        v = self.vit
        P = torch.stack([self.prompts[t] for t in tasks])        # (B, N, p, d)
        p = P.shape[2]
        x = v.norm_pre(v._pos_embed(v.patch_embed(x)))
        for i, blk in enumerate(v.blocks):
            rest = x[:, 1:] if i == 0 else x[:, 1 + p:]
            x = blk(torch.cat([x[:, :1], P[:, i], rest], dim=1))
        cls = v.norm(x)[:, 0]
        return [self.heads[t](cls[b]) for b, t in enumerate(tasks)]
