# EDGAR

**Efficient Diffusion-Driven Gaussian Avatar Refinement via Synthetic Failure**  
Kwan Yun · Min-Hyeok Bang · Colette Suhjung Koo · Junyong Noh  
Accepted to **ACM Transactions on Graphics**

[Project page](https://kwanyun.github.io/edgar/) · [Training data](docs/dataset.md) · [Paper](https://arxiv.org/abs/2611.07909)

EDGAR refines degraded Gaussian-avatar renderings using an identity reference and
a single-step diffusion model. Refined views then supervise the existing 3D avatar,
so the resulting avatar renders through its original pipeline without per-frame
diffusion inference.

This repository includes refiner training, image inference, and avatar refinement
adapters for RGBAvatar and CAP4D. The full model uses six experts in each of 11 MoE
modules. Ablation variants, experiment logs, baseline repositories, and third-party
model assets are excluded.

## Installation

Use Linux, Python 3.10, and an NVIDIA CUDA GPU. Install in a separate environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[train,test]'
```

For image inference alone, install with `pip install -e .`. Native avatar refinement
also requires the selected baseline's renderer, dependencies, and licensed assets;
see [the setup guide](docs/refinement.md). The first EDGAR run downloads SD-Turbo.
For offline operation, populate the model caches. Training/image inference also
accepts `--base-model /path/to/sd-turbo`, overriding the checkpoint's saved location.
Training additionally uses torchvision's VGG16 and LPIPS weights; cache these
before running without network access.

## Model assets

Place model files outside version control:

```text
checkpoints/
├── edgar_exp6.pt
├── sapiens_0.3b_epoch_1600_torchscript.pt2
└── affectnet8-sd.pth                         # training only
```

| Asset | Source | Used by |
| --- | --- | --- |
| EDGAR six-expert checkpoint | [Private checkpoint ZIP](https://github.com/kwanyun/edgar-checkpoint/releases/download/v1.0/edgar_exp6.zip) → `edgar_exp6.pt` | Inference and refinement |
| Sapiens 0.3B feature backbone | [Official TorchScript checkpoint](https://huggingface.co/facebook/sapiens-pretrain-0.3b-torchscript) | All stages |
| POSTER V2, AffectNet 8 classes | [Official checkpoints](https://github.com/Talented-Q/POSTER_V2#checkpoints) | Training |
| SD-Turbo | [Official model](https://huggingface.co/stabilityai/sd-turbo) | All stages; downloaded automatically |

Use the Sapiens **pretraining feature** checkpoint, not a depth or segmentation
checkpoint. POSTER must include the complete feature-network weights. Model files
are packaged separately from the code and retain their applicable licenses.
The checkpoint and datasets are hosted in separate private GitHub repositories.
Sign in with an account that has access; private links may show 404 otherwise.
See [download instructions](docs/downloads.md) for authenticated downloads and
checksum verification. The checkpoint was trained with both data sources.

## Refine images

```bash
edgar-infer \
  --checkpoint checkpoints/edgar_exp6.pt \
  --sapiens checkpoints/sapiens_0.3b_epoch_1600_torchscript.pt2 \
  --input renders/ --reference reference.png \
  --output outputs/refined_images \
  --preprocessing application --seed 2026
```

The input may be one image or a directory. Use a white-background avatar rendering
and a reference image of the same identity. Outputs are raw 512×512 RGB images;
input subdirectories are retained. For example, `frame.jpg` becomes `frame.jpg.png`.
Choose a new output directory for each run.

`--preprocessing application` preserves the original avatar application's white
padding and reference normalization. For a checkpoint newly trained with this
repository, use `--preprocessing training`. See [implementation details](docs/implementation.md)
for this distinction. `--precision bf16` enables optional lower-precision inference.

## Train EDGAR

Extract an EDGAR-SET package and use its manifest:

```bash
python tools/unpack_dataset.py --package /path/to/CAFCA-only --output data/CAFCA

edgar-train \
  --manifest data/CAFCA/manifest_identity_disjoint.json \
  --sapiens checkpoints/sapiens_0.3b_epoch_1600_torchscript.pt2 \
  --poster checkpoints/affectnet8-sd.pth \
  --output outputs/train \
  --steps 40000 --batch-size 4 --accumulation 2 \
  --learning-rate 2e-5 --warmup 1000 --precision bf16
```

The trainer combines pixel, LPIPS, Gram, expression-feature, and MoE balancing
losses. It writes local JSONL metrics and complete checkpoints. No W&B account is
required. Add `--initialize checkpoints/edgar_exp6.pt` to fine-tune, or
`--resume outputs/train/step_005000.pt` with a new output directory to continue a run.
`--steps` always specifies the total target update count.

The example uses a release-defined identity-disjoint split for new training.
The accompanying `manifest.json` preserves the recovered training loader's sorted
99/1 split. CAFCA-only training uses the synthetic subset; it does not reproduce
training with the combined dataset. See [dataset documentation](docs/dataset.md).

Export a training checkpoint without optimizer state:

```bash
python -m edgar.export --checkpoint outputs/train/step_040000.pt --output checkpoints/edgar_trained.pt
```

## Refine a Gaussian avatar

Start from a trained native avatar. Edit `configs/refine_rgbavatar.json` to point to
the native repository, checkpoint, tracked training sequence, and FLAME model:

```bash
edgar-refine-avatar \
  --pipeline rgbavatar --config configs/refine_rgbavatar.json \
  --checkpoint checkpoints/edgar_exp6.pt \
  --sapiens checkpoints/sapiens_0.3b_epoch_1600_torchscript.pt2 \
  --reference reference.png --output outputs/refined_avatar \
  --steps 500 --preprocessing application
```

For CAP4D, use `--pipeline cap4d --config configs/refine_cap4d.json`. The adapters
sample training states, generate EDGAR pseudo-targets, optimize the existing avatar,
and save a new native checkpoint. The default sampling uses 50 states and four yaw
offsets, producing 200 target views. No densification or pruning is performed.
RGBAvatar retains its native fast-forward color update.

The RGBAvatar result is `outputs/refined_avatar/avatar/model.ply`; CAP4D writes a
native `chkpnt*.pth` under the same `avatar/` directory. Render these with the
corresponding baseline renderer. Setup, configuration, and validation scope are
described in [the refinement guide](docs/refinement.md).

## EDGAR-SET packages

| Package | Triplets | Identities | Availability |
| --- | ---: | ---: | --- |
| CAFCA-only | 22,420 | 1,500 synthetic | [Private download](https://github.com/kwanyun/edgar-set-cafca/releases/tag/v1.0); CC BY-NC-SA 4.0 |
| Both | 132,915 | 1,599 | [Private download](https://github.com/kwanyun/edgar-set-both/releases/tag/v1.0); public redistribution permission pending |

Each triplet is a 1536×512 image ordered **reference · ground truth · degraded
rendering**. Packages include approximately 1 GiB TAR shards, SHA256 checksums,
sample indexes, and explicit training/validation manifests. See
[the dataset card](docs/dataset.md) for licenses and split definitions.

## Validation

```bash
python -m pytest -q
```

Tests cover data/preprocessing, checkpoints, native-adapter hooks, and dataset
packaging. The recovered six-expert refiner has been compared against the original
application on an A6000 with identical FP32 output on the checked image.
[VALIDATION.md](VALIDATION.md) records the completed runtime checks and their scope.

## Citation

```bibtex
@misc{yun2026edgar,
      title={EDGAR: Efficient Diffusion-Driven Gaussian Avatar Refinement via Synthetic Failure}, 
      author={Kwan Yun and Minhyeok Bang and Colette Suhjung Koo and Junyong Noh},
      year={2026},
      eprint={2611.07909},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2611.07909}, 
}
```

## Acknowledgments and license

EDGAR builds on SD-Turbo, Difix3D+, Diffusers, Sapiens, POSTER V2, and the native
avatar pipelines. See [THIRD_PARTY.md](THIRD_PARTY.md), [NOTICE](NOTICE), and
[LICENSE.txt](LICENSE.txt). The supplied NVIDIA-derived code is for noncommercial
research/evaluation under the retained license; datasets and model assets have
their own terms. **Powered by Stability AI.**
