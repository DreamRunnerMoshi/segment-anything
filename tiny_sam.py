"""
tiny_sam.py — one SAM training loop, reconstructed step by step.

The official repo is inference-only (no training code ships with it). This
script rebuilds the training procedure from the paper:

    Kirillov et al., "Segment Anything", ICCV 2023, §3.1 "Training"

The recipe it follows:

  Step 1  The image encoder (heavy ViT) runs ONCE per image. Its embedding is
          reused for every prompt iteration — that asymmetry is the core of SAM.
  Step 2  Prompt iteration 1: a single foreground POINT prompt -> MULTI-mask
          output (3 masks: whole / part / subpart). We supervise only the mask
          closest to GT, and train the IoU head to predict that mask's true IoU
          (this is what makes mask selection possible at inference, no GT).
  Step 3  Prompt iterations 2-3: sample new points from the ERROR REGION
          (predicted mask XOR ground truth) — label 1 for false negatives,
          0 for false positives — accumulate them into the prompt, and feed
          the previous low-res logits back in as a dense mask prompt. Now
          SINGLE-mask output. Supervise again.
  Step 4  Losses, per the paper: sigmoid FOCAL loss + DICE loss on masks,
          MSE on the predicted IoU.
  Step 5  One end-to-end backward pass through encoder + prompt encoder +
          decoder, then an Adam step. (The paper trains all of it from scratch.)

Because SA-1B is 11M images / 1.1B masks and ViT-H is 632M params, we shrink
the model (same wiring as segment_anything/build_sam.py, ~250k params, 128x128
images) and use synthetic circle-on-noise images, so the full pipeline —
prompt sampling, multi-mask selection, error-driven re-prompting, mask
feedback, all three losses — runs in seconds on a laptop.

Run:  .venv/bin/python tiny_sam.py
"""

import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim

sys.path.insert(0, str(Path(__file__).resolve().parent))

from segment_anything.modeling import (  # noqa: E402
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)

# ---------------------------------------------------------------------------
# Config — real SAM: 1024x1024 images, batch 256, 665k iterations on ViT-H.
# ---------------------------------------------------------------------------
IMG_SIZE = 128          # real SAM: 1024
EMBED_DIM = 64          # real SAM: 256 (prompt/decoder) and 1280 (ViT-H)
BATCH = 4
N_STEPS = 100
LR = 1e-3
DEVICE = os.environ.get("SAM_DEVICE", "cpu")  # "mps" / "cuda" also work


# ---------------------------------------------------------------------------
# Step 0: build a *tiny* SAM — identical wiring to build_sam._build_sam(),
# just with shrunken dims. 128px input / patch 16 -> 8x8 embedding grid.
# ---------------------------------------------------------------------------
def build_tiny_sam() -> Sam:
    grid = IMG_SIZE // 16
    return Sam(
        image_encoder=ImageEncoderViT(
            depth=2,
            embed_dim=EMBED_DIM,
            img_size=IMG_SIZE,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_heads=4,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[0, 1],  # all layers global -> no windowing
            window_size=8,
            out_chans=EMBED_DIM,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=EMBED_DIM,
            image_embedding_size=(grid, grid),
            input_image_size=(IMG_SIZE, IMG_SIZE),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=EMBED_DIM,
                mlp_dim=2 * EMBED_DIM,
                num_heads=4,
            ),
            transformer_dim=EMBED_DIM,
            iou_head_depth=3,
            iou_head_hidden_dim=EMBED_DIM,
        ),
    )


# ---------------------------------------------------------------------------
# Synthetic stand-in for SA-1B: a colored circle on a noisy background.
# ---------------------------------------------------------------------------
def make_batch(batch: int = BATCH):
    imgs = torch.rand(batch, 3, IMG_SIZE, IMG_SIZE) * 64.0   # dim noise bg
    masks = torch.zeros(batch, IMG_SIZE, IMG_SIZE, dtype=torch.bool)
    yy, xx = torch.meshgrid(
        torch.arange(IMG_SIZE, dtype=torch.float32),
        torch.arange(IMG_SIZE, dtype=torch.float32),
        indexing="ij",
    )
    for b in range(batch):
        cx = torch.randint(IMG_SIZE // 4, 3 * IMG_SIZE // 4, ()).float()
        cy = torch.randint(IMG_SIZE // 4, 3 * IMG_SIZE // 4, ()).float()
        r = torch.randint(IMG_SIZE // 8, IMG_SIZE // 4, ()).float()
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        imgs[b][:, inside] = (torch.rand(3, 1) * 190 + 60).expand(3, inside.sum())
        masks[b] = inside
    return imgs, masks


# ---------------------------------------------------------------------------
# Prompt sampling — mimics SAM's "model-in-the-loop data engine".
# ---------------------------------------------------------------------------
def sample_foreground_points(gt):
    """One random point inside each GT mask: SAM's iteration-1 prompt.

    Returns coords (B,1,2) as (x,y) and labels (B,1) of ones (foreground).
    """
    coords = []
    for m in gt:
        ys, xs = m.nonzero(as_tuple=True)
        i = torch.randint(len(xs), ())
        coords.append(torch.stack([xs[i].float(), ys[i].float()]))
    return torch.stack(coords)[:, None, :], torch.ones(len(gt), 1, device=gt.device)


def sample_error_points(pred, gt):
    """One random point from each sample's error region (pred XOR gt).

    The point's label is the GT value at that pixel: 1 means the model missed
    object there (false negative), 0 means it hallucinated object (false
    positive). Accumulating these is how training sharpens ambiguous prompts.
    """
    coords, labels = [], []
    for p, g in zip(pred, gt):
        err = p != g
        ys, xs = (err if err.any() else g).nonzero(as_tuple=True)
        i = torch.randint(len(xs), ())
        coords.append(torch.stack([xs[i].float(), ys[i].float()]))
        labels.append(g[ys[i], xs[i]].float())
    return (
        torch.stack(coords)[:, None, :].to(gt.device),
        torch.stack(labels)[:, None].to(gt.device),
    )


# ---------------------------------------------------------------------------
# Losses — paper §3.1: focal + dice on masks, MSE for the IoU head.
# ---------------------------------------------------------------------------
def sigmoid_focal_loss(logits, targets, alpha=0.8, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t) ** gamma * ce).mean()


def dice_loss(logits, targets, smooth=1.0):
    p = torch.sigmoid(logits).flatten(1)
    t = targets.flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + smooth) / (p.sum(1) + t.sum(1) + smooth)).mean()


def iou_mse_loss(pred_iou, true_iou):
    """Trains the IoU head to rank masks — at inference the best of the 3
    multi-mask outputs is picked by this head alone, with no GT available."""
    return F.mse_loss(pred_iou, true_iou)


@torch.no_grad()
def hard_iou(logits, gt):
    """Actual IoU of each predicted mask vs GT — the 'teacher' signal used to
    select which of the 3 masks to supervise, and as the IoU-head target."""
    pred = logits > 0.0
    inter = (pred & gt[:, None]).flatten(2).sum(2)
    union = (pred | gt[:, None]).flatten(2).sum(2)
    return inter.float() / union.clamp(min=1).float()  # (B, C)


# ---------------------------------------------------------------------------
# One training step: the full SAM loop with 3 iterative prompt rounds.
# ---------------------------------------------------------------------------
def decode(sam, img_emb, img_pe, sparse, dense, multimask_output):
    """MaskDecoder batches over PROMPT SETS for a *single* image — inside it,
    `repeat_interleave(image_embeddings, n_prompts)` assumes the image batch
    dim is 1. So we loop over images, exactly like Sam.forward() does
    (sam.py lines 101-117), and stack the results back into a batch."""
    lows, ious = [], []
    for i in range(img_emb.shape[0]):
        lr, ip = sam.mask_decoder(
            img_emb[i : i + 1], img_pe, sparse[i : i + 1], dense[i : i + 1],
            multimask_output=multimask_output,
        )
        lows.append(lr)
        ious.append(ip)
    return torch.cat(lows), torch.cat(ious)


def train_step(sam, opt, imgs, gt, trace=False):
    opt.zero_grad()
    gt = gt.to(DEVICE)
    gt_f = gt[:, None].float()                 # (B,1,H,W) for the pixel losses
    B = gt.shape[0]
    pick = torch.arange(B, device=DEVICE)

    # --- Step 1: image encoder runs ONCE; embedding reused below -----------
    x = sam.preprocess(imgs.to(DEVICE))        # normalize + pad -> (B,3,128,128)
    img_emb = sam.image_encoder(x)             # (B, EMBED_DIM, 8, 8)
    img_pe = sam.prompt_encoder.get_dense_pe() # (1, EMBED_DIM, 8, 8)
    if trace:
        print(f"  image          {tuple(imgs.shape)} -> encoder ONCE -> {tuple(img_emb.shape)}")

    # --- Step 2: iteration 1 — one fg point, MULTI-mask output -------------
    coords, labels = sample_foreground_points(gt)
    sparse, dense = sam.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
    low_res, iou_pred = decode(sam, img_emb, img_pe, sparse, dense, True)
                                                                  # (B,3,32,32)
    logits = F.interpolate(low_res, size=IMG_SIZE, mode="bilinear", align_corners=False)

    ious = hard_iou(logits, gt)                                   # (B,3)
    best = ious.argmax(dim=1)                # teacher: supervise best of the 3
    sel_logits = logits[pick, best][:, None]                      # (B,1,H,W)
    sel_low = low_res[pick, best][:, None]                        # (B,1,32,32)
    loss = (
        sigmoid_focal_loss(sel_logits, gt_f)
        + dice_loss(sel_logits, gt_f)
        + iou_mse_loss(iou_pred[pick, best], ious[pick, best])
    )
    pred_mask = sel_logits[:, 0] > 0.0
    if trace:
        print(f"  it1 prompt     {tuple(coords.shape)} xy -> sparse {tuple(sparse.shape)}"
              f"  (2nd token is the auto-padded 'no-point' token, no box given)")
        print(f"  it1 decoder    multi-mask: low_res {tuple(low_res.shape)},"
              f" logits {tuple(logits.shape)}, iou_pred {tuple(iou_pred.shape)}")
        print(f"  it1 selected-mask mean IoU: {ious[pick, best].mean():.3f}")

    # --- Step 3: iterations 2-3 — error-region points + mask feedback ------
    true_iou = None
    for it in (2, 3):
        err_coords, err_labels = sample_error_points(pred_mask, gt)
        coords = torch.cat([coords, err_coords], dim=1)
        labels = torch.cat([labels, err_labels], dim=1)
        sparse, dense = sam.prompt_encoder(
            points=(coords, labels), boxes=None, masks=sel_low    # logits fed back
        )
        low_res, iou_pred = decode(sam, img_emb, img_pe, sparse, dense, False)
                                                                   # (B,1,32,32)
        logits = F.interpolate(low_res, size=IMG_SIZE, mode="bilinear", align_corners=False)
        true_iou = hard_iou(logits, gt)[:, 0]
        loss = loss + (
            sigmoid_focal_loss(logits, gt_f)
            + dice_loss(logits, gt_f)
            + iou_mse_loss(iou_pred[:, 0], true_iou)
        )
        sel_low = low_res                      # feedback into the next round
        pred_mask = logits[:, 0] > 0.0
        if trace:
            print(f"  it{it} prompt     {it} accumulated pts -> sparse {tuple(sparse.shape)},"
                  f" dense mask feedback {tuple(dense.shape)} -> single mask,"
                  f" mean IoU {true_iou.mean():.3f}")

    # --- Steps 4-5: end-to-end backward + Adam step -------------------------
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(sam.parameters(), 5.0))
    opt.step()
    return {
        "loss": loss.item(),
        "iou_it1": ious[pick, best].mean().item(),
        "iou_it3": true_iou.mean().item(),
        "grad_norm": grad_norm,
    }


def main():
    torch.manual_seed(7)
    sam = build_tiny_sam().to(DEVICE).train()
    n_enc = sum(p.numel() for p in sam.image_encoder.parameters())
    n_prompt = sum(p.numel() for p in sam.prompt_encoder.parameters())
    n_dec = sum(p.numel() for p in sam.mask_decoder.parameters())
    print(f"tiny SAM on {DEVICE}: {n_enc + n_prompt + n_dec:,} params"
          f"  (encoder {n_enc:,} | prompt {n_prompt:,} | decoder {n_dec:,})")
    print("training all of it end-to-end, as in the paper\n")
    opt = optim.Adam(sam.parameters(), lr=LR)

    print("=== one training step, traced ===")
    imgs, gt = make_batch()
    d = train_step(sam, opt, imgs, gt, trace=True)
    print(f"  => loss {d['loss']:.3f} | IoU it1 {d['iou_it1']:.3f} -> it3 {d['iou_it3']:.3f}\n")

    print("=== fresh synthetic batch every step (like streaming from SA-1B) ===")
    for step in range(N_STEPS):
        imgs, gt = make_batch()
        d = train_step(sam, opt, imgs, gt)
        print(f"step {step:2d}  loss {d['loss']:7.3f}   "
              f"IoU it1 {d['iou_it1']:.3f} -> it3 {d['iou_it3']:.3f}   "
              f"|grad| {d['grad_norm']:.2f}")


if __name__ == "__main__":
    main()
