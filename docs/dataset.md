# EDGAR-SET

EDGAR-SET pairs a degraded avatar rendering with ground truth and a different
reference view/expression of the same identity. The recovered distribution contains
132,915 triplets (398,745 image panels), matching the paper's rounded 400K images.

| Source | Triplets | Identities | TAR bytes |
| --- | ---: | ---: | ---: |
| CAFCA-derived synthetic | 22,420 | 1,500 | 11,685,611,520 |
| NeRSemble-derived real capture | 110,495 | 99 | 42,941,440,000 |
| Combined | 132,915 | 1,599 | 54,627,051,520 |

These counts describe the recovered, filtered training triplets, not the complete
upstream CAFCA or NeRSemble datasets. Each image is 1536×512 with three 512×512 panels:

```text
[ reference image | ground-truth image | degraded avatar rendering ]
```

## Packages and terms

**CAFCA-only** contains 11 TAR shards and is an adaptation of
[CAFCA](https://github.com/syntec-research/Cafca/blob/main/Dataset.md), distributed
under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The adaptation constructs degraded Gaussian renderings, pairs them with ground
truth and reference views, filters samples, and packs the triplet panels. Retain
the included attribution, change notice, and license when redistributing it.

**Both** contains those CAFCA shards plus 40 NeRSemble-derived shards. It is hosted
privately while permission to redistribute the NeRSemble-derived
images is pending. It is not covered wholesale by the CAFCA license. Researchers
can request upstream access through the [official NeRSemble data repository](https://github.com/tobias-kirschstein/nersemble-data).

CAFCA-only is publicly available; Both remains private. See their separate
[download links and instructions](downloads.md).

The packages exclude FLAME models, original videos, tracking meshes, model weights,
and personal contact information. Training images remain subject to their source terms.

## Extract and verify

```bash
python tools/unpack_dataset.py --package /path/to/CAFCA-only --output data/CAFCA
```

The script verifies every archive's SHA256 before extracting. It rejects unsafe
archive members, duplicate paths, and existing output directories. The result is:

```text
data/CAFCA/
├── manifest.json
├── manifest_identity_disjoint.json
├── index.jsonl
└── triplets/CAFCA/...
```

Use the same command with an authorized `Both` package to extract the combined set.
The sample index records identity, source, relative path, image byte size, SHA256,
and shard membership. Every recovered triplet passed image-format, dimensions,
and PNG integrity checks before packaging.

## Splits

- `manifest.json` reproduces the recovered loader's **sorted 99/1 file split**.
  It can share identities across train and validation and must not be presented as
  an identity-disjoint evaluation.
- `manifest_identity_disjoint.json` is a **new release split** for training new
  models: approximately 2% of identities from each source are held out using stable
  SHA256 ordering with seed 2026. It is not the original paper split.

The package preserves canonical recovered files without re-filtering, recompression,
or new augmentation. Ten byte-identical Jupyter image backups in the NeRSemble
tree are excluded; their canonical originals remain included. The 99/1 split is
applied to these cleaned paths, so it is not an exact replay of a loader that also
enumerated those editor backups. Training a CAFCA-only model does
not reproduce the combined-data experiment.

## Use your own triplets

Create explicit train and validation lists. Paths are relative to the manifest:

```json
{
  "train": [{"triplet": "train/packed.png"}],
  "validation": [{"triplet": "validation/packed.png"}]
}
```

Alternatively, each row may contain `input`, `target`, and `reference` image paths.
Input and target must be aligned and the reference must depict the same identity.
Keep held-out evaluation identities out of the training set.

## Rebuild packages from the recovered layout

```bash
python tools/package_dataset.py --root /path/to/data --output /path/to/new_staging
```

The input layout is `GS/<range>/<identity>/*.png` and
`RGB/<identity>/<setting>/<camera>/*.png`. This packaging utility preserves existing
triplets; it does not recreate the upstream avatar-generation and filtering process.
