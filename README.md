# 🪄 tiny‑SAM — watch *Segment Anything* learn, on a laptop

[![built on Segment Anything](https://img.shields.io/badge/built_on-Segment_Anything-6f42c1)](https://github.com/facebookresearch/segment-anything)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.7-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![runs on](https://img.shields.io/badge/runs_on-CPU_%C2%B7_MPS_%C2%B7_CUDA-success)](#quickstart)
[![tiny model](https://img.shields.io/badge/tiny_model-355K_params-lightgrey)](#tiny-sam-vs-real-sam)
[![paper](https://img.shields.io/badge/paper-arXiv%3A2304.02643-b31b1b)](https://arxiv.org/abs/2304.02643)

**Meta's SAM ships inference code only — there is no training script.** tiny‑SAM is a
self‑contained reconstruction of SAM's *training loop* from the paper, shrunk to **355K
parameters** so the whole thing — image encoder, prompt encoder, mask decoder, multi‑mask
selection, error‑driven re‑prompting, mask feedback, focal + dice + IoU losses, end‑to‑end
backprop — runs in a few seconds on a CPU and **visibly learns to segment**:

```
step  0  loss  2.652  IoU 0.12 → 0.08     step 49  loss  0.286  IoU 0.86 → 0.87
step 19  loss  1.529  IoU 0.82 → 0.82     step 99  loss  0.096  IoU 0.96 → 0.96   ✅
```

If you've ever wanted to *see* how a promptable segmenter is actually trained — not just
call `predictor.predict()` — this is the repo for it.

![tiny-SAM training loop](assets/training_loop.png)

> One optimizer step, end‑to‑end. The **green dashed rail** is the image embedding,
> computed *once* and reused by all three prompt iterations; the **orange / blue elbows**
> feed each iteration's predicted mask and low‑res logits back into the next prompt; the
> **red lines** on the right merge the three losses into a single backward pass.
> Editable source: [`assets/training_loop.excalidraw`](assets/training_loop.excalidraw).

---

## Why this exists

The official [`segment_anything/`](segment_anything/) package gives you a frozen model and
a beautiful inference API — but it never shows *how the model got good*. The training
procedure lives only in §3.1 of the paper (Kirillov et al., *Segment Anything*, ICCV 2023).
tiny‑SAM turns that prose into runnable, annotated code, line‑for‑line mapped to the paper,
so you can read it, break it, and watch the numbers move.

It's a great way to understand, concretely:

- why the **image encoder runs once** and the tiny decoder runs many times,
- how a **single ambiguous click** becomes 3 candidate masks, and how the **IoU head**
  learns to pick the right one *without ground truth* at inference time,
- how **mistakes become the next prompt** (sample points from `pred XOR gt`),
- and a subtle API gotcha: the decoder's batch dim is *prompt hypotheses per image*, not
  images — which is exactly why `Sam.forward()` loops over the batch.

---

## Quickstart

No checkpoint download, no GPU, no dataset. Just PyTorch:

```bash
git clone https://github.com/<you>/segment-anything.git   # this repo
cd segment-anything
python3 -m venv .venv
.venv/bin/pip install torch numpy
.venv/bin/python tiny_sam.py
```

(`tiny_sam.py` only needs the vendored `segment_anything/modeling` package — no
`torchvision`, no extra deps. Set `SAM_DEVICE=mps` or `cuda` to use a GPU.)

You'll see one fully‑traced step (every tensor shape printed), then 100 streaming steps on
fresh synthetic images — a tiny stand‑in for SA‑1B:

```
tiny SAM on cpu: 355,528 params  (encoder 195,456 | prompt 1,804 | decoder 158,268)
training all of it end-to-end, as in the paper

=== one training step, traced ===
  image          (4, 3, 128, 128) -> encoder ONCE -> (4, 64, 8, 8)
  it1 prompt     (4, 1, 2) xy -> sparse (4, 2, 64)  (2nd token is the auto-padded 'no-point' token)
  it1 decoder    multi-mask: low_res (4, 3, 32, 32), logits (4, 3, 128, 128), iou_pred (4, 3)
  it1 selected-mask mean IoU: 0.082
  it2 prompt     2 accumulated pts -> sparse (4, 3, 64), dense mask feedback (4, 64, 8, 8) -> single mask
  it3 prompt     3 accumulated pts -> sparse (4, 4, 64), dense mask feedback (4, 64, 8, 8) -> single mask

=== fresh synthetic batch every step (like streaming from SA-1B) ===
step  0  loss   2.652   IoU it1 0.119 -> it3 0.080   |grad| 1.74
step  8  loss   2.748   IoU it1 0.419 -> it3 0.393   |grad| 1.49
step 19  loss   1.529   IoU it1 0.818 -> it3 0.816   |grad| 3.39
step 49  loss   0.286   IoU it1 0.859 -> it3 0.865   |grad| 1.00
step 69  loss   0.162   IoU it1 0.919 -> it3 0.927   |grad| 1.15
step 99  loss   0.096   IoU it1 0.958 -> it3 0.960   |grad| 0.89
```

Mean mask IoU on *unseen* images goes **0.08 → 0.96** — the loop genuinely trains
segmentation, from scratch, end‑to‑end.

---

## How the training loop works

`tiny_sam.py` follows the paper's recipe in five steps per optimizer step:

| # | What happens | Where |
|---|---|---|
| 1 | **Encode once.** `ImageEncoderViT` → embedding `(B,64,8,8)`, cached & reused. | `train_step` |
| 2 | **Iteration 1 — ambiguous.** One foreground point → `PromptEncoder` → decoder with `multimask=True` → 3 masks; a *teacher* (hard IoU vs GT) picks the best; the **IoU head** is trained to predict that IoU. | `sample_foreground_points` |
| 3 | **Iterations 2–3 — refine.** Sample a point from `pred ⊕ gt` (label 1 = missed object, 0 = hallucinated), accumulate it, feed the previous logits back as a dense mask prompt, decode a single refined mask. | `sample_error_points` |
| 4 | **Losses.** sigmoid **focal** + **dice** on masks, **MSE** on predicted IoU. | `sigmoid_focal_loss` / `dice_loss` / `iou_mse_loss` |
| 5 | **Backward, end‑to‑end.** Gradients from all 9 mask losses flow through decoder + prompt encoder + encoder; one Adam step. | `main` |

The full, annotated walkthrough — with the diagram, every tensor shape, the decoder‑batch
gotcha, and a faithful‑vs‑simplified table — is in
**[`TRAINING_WALKTHROUGH.md`](TRAINING_WALKTHROUGH.md)**.

---

## tiny‑SAM vs real SAM

Same wiring as `segment_anything/build_sam.py`, just shrunken so it trains interactively:

|  | tiny‑SAM (this demo) | SAM ViT‑H (paper) |
|---|---|---|
| Image size | 128 × 128 | 1024 × 1024 |
| Embedding grid | 8 × 8 | 64 × 64 |
| Prompt / decoder dim | 64 | 256 |
| ViT embed / depth / heads | 64 / 2 / 4 | 1280 / 32 / 16 |
| Parameters | **355 K** | ~636 M |
| Data | synthetic circles | SA‑1B (11 M images, 1.1 B masks) |
| Train batch / steps | 4 / 100 | 256 / 665 k |
| Hardware | laptop CPU | many GPUs |

Everything that matters for *understanding* is preserved: 3‑iteration error‑driven prompting,
multi‑mask + IoU‑head supervision, mask‑logit feedback, and the three losses. What's dropped
for size is listed honestly in the walkthrough.

---

## Things to try

Tiny‑SAM is meant to be poked. In `tiny_sam.py`:

- **`N_STEPS = 300`** — watch IoU saturate near 1.0; confirms it really learns, not memorizes
  (every step is a *fresh* random image).
- **Freeze the encoder** — reproduces SAM's late‑training stage and makes steps much faster:
  ```python
  for p in sam.image_encoder.parameters():
      p.requires_grad_(False)
  ```
- **Box prompts** — unambiguous, so they skip the multi‑mask stage:
  ```python
  from segment_anything.utils.amg import batched_mask_to_box
  boxes = batched_mask_to_box(gt)
  sparse, dense = sam.prompt_encoder(points=None, boxes=boxes, masks=None)
  ```
- **Drop the IoU loss** — training loss still falls, but mask *selection* at inference (no GT)
  breaks. This isolates exactly *why* the IoU head exists.

---

## Repo layout

```
tiny_sam.py                 ← the training loop (start here)
TRAINING_WALKTHROUGH.md     ← step-by-step explanation + diagram
assets/training_loop.png    ← rendered training-loop diagram
assets/training_loop.excalidraw   ← editable diagram source
segment_anything/           ← upstream SAM model code (unchanged, used as the backbone)
```

---

## Using the upstream model (unchanged)

The vendored `segment_anything/` package is Meta's original, unmodified — so all of its
inference tooling still works if you point it at a checkpoint:

```python
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)
predictor.set_image(<your_image>)
masks, scores, logits = predictor.predict(point_coords=..., point_labels=...)
```

Checkpoints: [`vit_h`](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) ·
[`vit_l`](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) ·
[`vit_b`](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth).
Automatic "segment everything" (`SamAutomaticMaskGenerator`), ONNX export, and the browser
demo are all still in `scripts/`, `notebooks/`, and `demo/`. For video, see the follow‑up
[SAM 2](https://github.com/facebookresearch/segment-anything-2).

---

## Built on Segment Anything

tiny‑SAM is an **independent, educational reconstruction** built on top of Meta AI Research's
open‑source Segment Anything — it is **not affiliated with or endorsed by Meta**. The
`segment_anything/` model code is reproduced verbatim under its original license; the
training‑loop script, walkthrough, and diagrams are original additions for learning purposes.

- Upstream repo: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
- Paper & project: [segment-anything.com](https://segment-anything.com/) · [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- License: the upstream model code is [Apache‑2.0](LICENSE). The tiny‑SAM training
  reconstruction, walkthrough, and diagrams are provided for educational use — please add
  your own `LICENSE` for your fork (e.g. MIT / Apache‑2.0) before publishing.

If you use SAM or SA‑1B in research, please cite the original work:

<details>
<summary>BibTeX</summary>

```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and
          Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and
          Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

</details>
