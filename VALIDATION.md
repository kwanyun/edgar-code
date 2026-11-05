# Validation

Validated on 2026-11-04–05 using an NVIDIA RTX A6000 (48 GB). Original research
repositories and data were left unchanged; tests used isolated copies and outputs.

## Environment and package

- A fresh Python 3.10 virtual environment, without shared site packages, with
  PyTorch 2.3.1+cu121, torchvision 0.18.1, Diffusers 0.25.1, Transformers 4.38.0,
  PEFT 0.9.0, and the declared training dependencies.
- Native refinement used NumPy 1.23.5, SciPy 1.13.1, Chumpy 0.70, nvdiffrast 0.4.0,
  gsplat 1.5.3, and PyTorch3D 0.7.9. RGBAvatar's custom CUDA rasterizer was rebuilt
  against this environment. Native extensions were compiled with CUDA toolkit 12.4.
- Source parsing, CLI help, local documentation links, and example JSON checks.
- Source distribution and wheel builds succeeded. The wheel installed into a
  separate target directory; all three installed CLI entry points passed help checks.
- All **20 automated tests passed**, covering preprocessing, triplet order, validation behavior,
  checkpoint completeness, import isolation, POSTER feature loading, avatar
  optimization hooks, geometry-pickle compatibility, and archive integrity.
- Checkpoint-location regression tests confirm that an explicit local base-model
  path takes precedence, while omitted overrides preserve checkpoint defaults.
  The override case failed before the correction and passed afterward. This
  correction was checked on CPU; neural computation and model weights were unchanged.
- Source-package inspection for common secret patterns, private server paths,
  debugger calls, unwanted assets, and ZIP/file integrity.

The optional native environment's upstream Decord 0.6.0 wheel contains a CPython
3.6 wheel tag despite its `py3` distribution filename. Consequently `pip check`
reports a platform-tag warning for Decord on Python 3.10. Decord imported in the
completed native runs; it is not a dependency of EDGAR training or image inference.

## Image refiner and training

- Checkpoint inspection confirms 11 MoE modules with six experts each in exp6.
  The exp4 checkpoint uses the same module placement with four experts each.
- The exported `edgar_exp6.pt` retains the recovered 40,001-step exp6 inference
  tensors and excludes optimizer state. The inspected main RGBAvatar and CAP4D
  launchers select these weights.
- FP32 comparison with the original application, using exp6, identical seed,
  application reference normalization, and a white-padded rectangular render:
  maximum and mean absolute image differences **0.0** on the checked image.
- A separate comparison with the complete six-expert 10,001-step training
  checkpoint also produced maximum and mean absolute differences **0.0**.
- Fresh-environment BF16 training completed two updates at batch size 1 with
  gradient accumulation 2, including pixel, LPIPS, Gram, expression-feature,
  and balancing losses. Validation metrics were finite; the reference-projection
  gradient norm at update 2 was approximately **0.0858**.
- A separate Python 3.9 environment completed training, checkpoint save, resume
  from update 2 to 3, optimizer/scheduler restoration, inference-weight export,
  and FP32/BF16 image inference from the exported checkpoint.

## Native avatar refinement

Each end-to-end smoke run loaded a real trained avatar and tracked state, generated
EDGAR targets at −30° and +30°, performed three optimization updates, and wrote a
new native checkpoint. Saved checkpoints were reloaded and rendered. Baseline
checkpoint hashes remained unchanged; parameters and rendered images were finite.

| Pipeline | Gaussian count before / after | Saved output |
| --- | ---: | --- |
| RGBAvatar | 82,555 / 82,555 | Native `model.ply` |
| CAP4D | 72,058 / 72,058 | Native `chkpnt*.pth` |

RGBAvatar's three sampled-view losses were 0.06156, 0.05530, and 0.05170.
CAP4D's were 0.01725, 0.05637, and 0.05304; these use different sampled views
and are not a convergence comparison.

An additional CAP4D save/reload check verified exact preservation of all optimized
parameters, including nonzero per-frame neck offsets. Reloaded render differences
were at most **4.45 × 10⁻⁶** in the final check, with mean differences below
**3.64 × 10⁻⁹**. Native GPU rendering is not asserted to be bitwise deterministic.
The neck-offset loader rejects a different tracked sequence; native reenactment
can ignore the optional training-state field.

## Dataset packages

The recovered canonical set contains 22,420 CAFCA-derived and 110,495
NeRSemble-derived triplets. All images passed dimension, color-mode, and PNG
integrity checks before packaging. All 51 TAR shard hashes and archive member
counts were verified against the sample indexes. Both split policies were checked.
Ten excluded Jupyter backup images were byte-identical to included originals.
The combined package is private pending NeRSemble redistribution permission.

## Scope

These checks establish the tested execution paths, not convergence at 40,000
updates or reproduction of all paper metrics. Native tests use the supplied
research checkouts; unmodified public baseline repositories were not independently
validated. No separate GaussianAvatars adapter is included. Original model assets,
private test inputs, and generated examples are excluded from the source archive.
