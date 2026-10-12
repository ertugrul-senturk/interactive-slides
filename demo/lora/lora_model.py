"""LoRA (Hu et al., ICLR 2022) on a frozen timm ViT.

Every attention block of ViT-B/16 computes query, key and value with one fused 768 -> 2304 matrix (`qkv`).
LoRA leaves that matrix frozen and adds a trained low-rank bypass beside the query part and the value part,
as in the paper's default (W_q and W_v):

    q = W_q x + (alpha / r) * B_q A_q x        A: r x 768 (random init),  B: 768 x r (zeros at start)
    v = W_v x + (alpha / r) * B_v A_v x

  rank = 0  -> linear probe (frozen backbone + linear head), so baselines share the code
  rank = 8  -> LoRA on W_q and W_v in all 12 blocks: 12 x 2 x 2 x 768 x 8 = 294,912 numbers + the head

After training, B A is a 768 x 768 matrix that can be added into W once (`merge`), and the model is again a
plain ViT of the same size and speed.
"""
import math

import timm
import torch
import torch.nn as nn

BACKBONE = "vit_base_patch16_224.orig_in21k"  # ViT-B/16, ImageNet-21k supervised: the same base model as the VPT demo


class LoRAQKV(nn.Module):
    """Wraps a frozen fused qkv Linear and adds low-rank updates to its query and value outputs.

    Two ways to supply the pairs:
      * its own trained parameters (training, a single task), or
      * `ext = (A, B)` set from outside, per image: A (batch, 2, r, d), B (batch, 2, d, r), already scaled.
        The browser graph and the multi-task model use this, so every image in a batch can carry its own task.
    """

    def __init__(self, qkv, rank, alpha, dropout=0.0):
        super().__init__()
        self.qkv, self.rank = qkv, rank
        d = qkv.in_features
        self.scale = alpha / rank if rank else 0.0
        self.dropout = nn.Dropout(dropout)
        self.ext = None
        if rank:
            # Same initialisation as the official loralib: A Kaiming-uniform, B zero, so training starts from W exactly
            self.A = nn.Parameter(torch.empty(2, rank, d))      # [query, value]
            self.B = nn.Parameter(torch.zeros(2, d, rank))
            for i in range(2):
                nn.init.kaiming_uniform_(self.A.data[i], a=math.sqrt(5))

    def forward(self, x):
        out = self.qkv(x)                                       # (batch, tokens, 3d): [q | k | v]
        d = self.qkv.in_features
        if self.ext is not None:
            A, B = self.ext
            h = self.dropout(x)
            # plain batched MatMuls (not einsum), so the exported graph uses the most widely supported ONNX op
            dq = (h @ A[:, 0].transpose(1, 2)) @ B[:, 0].transpose(1, 2)
            dv = (h @ A[:, 1].transpose(1, 2)) @ B[:, 1].transpose(1, 2)
        elif self.rank:
            h = self.dropout(x)
            dq = (h @ self.A[0].T) @ self.B[0].T * self.scale
            dv = (h @ self.A[1].T) @ self.B[1].T * self.scale
        else:
            return out
        return torch.cat([out[..., :d] + dq, out[..., d:2 * d], out[..., 2 * d:] + dv], dim=-1)

    def delta(self):
        """The trained change to the fused qkv weight, (3d, d): zero rows for the key."""
        d = self.qkv.in_features
        z = torch.zeros(d, d, device=self.A.device, dtype=self.A.dtype)
        return torch.cat([self.B[0] @ self.A[0], z, self.B[1] @ self.A[1]], dim=0) * self.scale


class LoRAViT(nn.Module):
    def __init__(self, num_classes, rank=8, alpha=8, dropout=0.0, backbone=BACKBONE, pretrained=True, vit=None):
        super().__init__()
        # Pass `vit` to share one frozen backbone between several tasks or methods.
        self.vit = vit if vit is not None else timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.rank, self.alpha = rank, alpha
        for blk in self.vit.blocks:
            qkv = blk.attn.qkv.qkv if isinstance(blk.attn.qkv, LoRAQKV) else blk.attn.qkv   # a shared vit may be wrapped already
            blk.attn.qkv = LoRAQKV(qkv, rank, alpha, dropout)
        self.head = nn.Linear(self.vit.embed_dim, num_classes)

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()   # the backbone stays in eval mode: no dropout or drop-path inside it
        for blk in self.vit.blocks:
            if isinstance(blk.attn.qkv, LoRAQKV):                # after merge() the bypass is gone
                blk.attn.qkv.dropout.train(mode)
        return self

    def lora_layers(self):
        return [blk.attn.qkv for blk in self.vit.blocks]

    def trainable_state_dict(self):
        """Just what a task needs to store: the low-rank pairs and the head."""
        state = {k: v for k, v in self.state_dict().items() if k.endswith((".A", ".B")) or k.startswith("head.")}
        state["_alpha"] = torch.tensor(float(self.alpha))       # needed to rebuild alpha / r on loading
        return state

    def stacked(self):
        """All pairs as two arrays, with alpha / r folded into B: A (12, 2, r, d), B (12, 2, d, r)."""
        L = self.lora_layers()
        return torch.stack([m.A for m in L]), torch.stack([m.B * m.scale for m in L])

    @torch.no_grad()
    def merge(self):
        """Adds B A into the frozen weights once and removes the bypass: a plain ViT again, same size and speed."""
        for blk in self.vit.blocks:
            m = blk.attn.qkv
            if m.rank:
                m.qkv.weight += m.delta()
            blk.attn.qkv = m.qkv
        self.rank = 0
        return self

    def forward_features(self, x):
        return self.vit(x)                                       # timm: final [CLS] embedding (num_classes = 0)

    def forward(self, x):
        return self.head(self.forward_features(x))


def count(model):
    backbone = sum(p.numel() for n, p in model.vit.named_parameters() if not (n.endswith(".A") or n.endswith(".B")))
    lora = sum(p.numel() for n, p in model.vit.named_parameters() if n.endswith(".A") or n.endswith(".B"))
    head = sum(p.numel() for p in model.head.parameters())
    return {"backbone": backbone, "lora": lora, "head": head,
            "trainable": lora + head, "trainable_pct": 100 * (lora + head) / backbone}


def load_task(state, num_classes, vit=None, pretrained=True):
    """Rebuilds a LoRAViT from a saved trainable_state_dict (pairs + head)."""
    rank = state["vit.blocks.0.attn.qkv.A"].shape[1]
    m = LoRAViT(num_classes, rank=rank, alpha=state.get("_alpha", torch.tensor(rank)).item(), vit=vit, pretrained=pretrained)
    m.load_state_dict({k: v for k, v in state.items() if not k.startswith("_")}, strict=False)
    return m


class MultiTaskLoRA(nn.Module):
    """One frozen ViT shared by several LoRA tasks. Each image in a batch carries its own task's pairs,
    so a mixed batch (say a skin lesion, a blood smear and an X-ray) runs in a single forward pass."""

    def __init__(self, vit):
        super().__init__()
        self.vit = vit
        for blk in vit.blocks:
            if not isinstance(blk.attn.qkv, LoRAQKV):
                blk.attn.qkv = LoRAQKV(blk.attn.qkv, 0, 0)
        self.A, self.B = {}, {}
        self.heads = nn.ModuleDict()

    def add_task(self, name, model):
        """`model` is a trained LoRAViT; only its pairs and head are kept."""
        A, B = model.stacked()
        self.A[name], self.B[name] = A.detach(), B.detach()
        self.heads[name] = model.head

    def forward(self, x, tasks):
        A = torch.stack([self.A[t] for t in tasks])            # (batch, 12, 2, r, d)
        B = torch.stack([self.B[t] for t in tasks])
        for i, blk in enumerate(self.vit.blocks):
            blk.attn.qkv.ext = (A[:, i], B[:, i])
        try:
            cls = self.vit(x)
        finally:
            for blk in self.vit.blocks:
                blk.attn.qkv.ext = None
        return [self.heads[t](cls[b]) for b, t in enumerate(tasks)]
