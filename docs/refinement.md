# Gaussian-avatar refinement

Refinement starts from an existing trained avatar and its tracked training states.
EDGAR does not replace the baseline's video tracking or initial reconstruction.
The built-in adapters use native rendering and deformation code from separately
installed RGBAvatar and CAP4D repositories.

## Native dependencies

Use Python 3.10. First install the selected baseline and obtain its model assets:

- [RGBAvatar](https://github.com/gapszju/RGBAvatar): its custom
  `submodules/diff-gaussian-rasterization`, nvdiffrast, FLAME assets, and the
  NeRSemble tracking format used by its multi-view loader.
- [CAP4D](https://github.com/felixtaubner/cap4d): its native avatar dependencies,
  including gsplat, PyTorch3D, FlowFace/FLAME assets, and an optimized avatar checkpoint.

The adapter API was taken from the supplied research checkouts. The public
repositories currently resolve to RGBAvatar commit
`903653b689a8b860df30f165e242ccd6ed3f5583` and CAP4D commit
`1f25e99702ab5c0b38aa5e195fdd6cd963fa528c`; the supplied checkouts have local
modifications, so these commits are references rather than a claim of identical trees.
Keep the native repository and EDGAR package in separate directories.

Install EDGAR and its additional Python helpers in the native environment:

```bash
python -m pip install -e '/path/to/EDGAR[avatar]'
python -m pip install --no-build-isolation chumpy==0.70
```

Use the pinned PyTorch 2.3.1/torchvision 0.18.1 stack for the checked EDGAR runtime.
The avatar extra pins NumPy 1.23.5 for compatibility with the legacy Chumpy objects
in FLAME model files. Chumpy's older build requires the explicit command above.
Compile the native CUDA extensions against that same PyTorch installation. CUDA
extensions compiled for a different Python or PyTorch version must be rebuilt.
Do not install another baseline's package under the same
`diff_gaussian_rasterization` module name in this environment.

FLAME model files are not included. Obtain them through the upstream process and
retain their original terms. The code package does not redistribute native avatar
checkpoints, tracking data, or the baseline repositories.

## RGBAvatar

Edit `configs/refine_rgbavatar.json`. All paths are resolved relative to the JSON
file before entering the native repository.

| Field | Meaning |
| --- | --- |
| repository | RGBAvatar checkout |
| native_config | YAML matching the trained avatar's model dimensions |
| baseline | Existing native `model.ply` |
| transforms | Tracked `transforms_train.json` in the NeRSemble/GaussianAvatars format |
| flame_model | Licensed FLAME geometry model, e.g. FLAME 2023 |
| camera_indices | Source camera indices used to train the avatar; empty means all listed training cameras |
| camera_count | Number of sampled camera/frame states; default 50 |
| angles | Yaw offsets in degrees; default -60, -30, 30, 60 |
| raster_context | `cuda` for headless use, or `gl` when OpenGL is available |

```bash
edgar-refine-avatar \
  --pipeline rgbavatar --config configs/refine_rgbavatar.json \
  --checkpoint checkpoints/edgar_exp6.pt \
  --sapiens checkpoints/sapiens_0.3b_epoch_1600_torchscript.pt2 \
  --reference reference.png --output outputs/rgbavatar_refined \
  --steps 500 --seed 2026 --preprocessing application
```

The adapter loads the native PLY, initializes native binding state, and samples
training states with replacement. It retains the native optimizer parameter groups
and fast-forward color update. The direct refinement objective is 0.6 L1 + 0.4 SSIM;
the original direct path does not reach the iteration threshold that enables LPIPS.

The output is `outputs/rgbavatar_refined/avatar/model.ply`. Use RGBAvatar's renderer
and the original tracked sequence to render the refined avatar. The saved
`refinement.json` records the sampled states, settings, and optimization losses.

## CAP4D

Edit `configs/refine_cap4d.json` with the native YAML, baseline `chkpnt*.pth`, and
the reference/generated image directories used by the baseline avatar.

```bash
edgar-refine-avatar \
  --pipeline cap4d --config configs/refine_cap4d.json \
  --checkpoint checkpoints/edgar_exp6.pt \
  --sapiens checkpoints/sapiens_0.3b_epoch_1600_torchscript.pt2 \
  --reference reference.png --output outputs/cap4d_refined \
  --steps 500 --preprocessing application
```

CAP4D samples camera/frame states without replacement. The adapter retains its
native deformation model and both Gaussian and neck optimizer steps. It saves a
native checkpoint under `outputs/cap4d_refined/avatar/`. Use CAP4D's animation
scripts with the original source data. The run's `native/data/assets/` contains
a private copy of the native assets; NumPy-2-authored FLAME geometry is reserialized
compatibly for NumPy 1.x, preserving its arrays. Run native animation from this
`native/` directory with the CAP4D repository on PYTHONPATH if the original assets
require that conversion.

The native checkpoint format omits per-frame neck offsets. EDGAR saves these in
an additional `edgar_training_state` field, which native CAP4D restore safely
ignores for reenactment with a new driving sequence. The EDGAR adapter restores
them when continuing refinement on the same tracked sequence. To reproduce the
optimized training views with a native script, call
`edgar.adapters.cap4d.restore_training_offsets(model, checkpoint_state)` after
native restore. This helper checks the tracked sequence before applying offsets.

Both adapters have passed short GPU refinement and native-checkpoint reload
checks; see [VALIDATION.md](../VALIDATION.md) for their scope.

## Other renderers

`--adapter module:factory` supports an additional pipeline without changing the
EDGAR network. The factory receives the JSON configuration and returns
`(avatar, views, optimizer)`. Required avatar members:

- `render(view)`: differentiable RGB CHW tensor on a calibrated, white-padded 512² canvas;
- `parameters()`: existing trainable parameters;
- `gaussian_count`: fixed Gaussian count;
- `save(path)`: save a new native model;
- optional `begin_optimization()` and `after_step()` for native mode/color-update behavior.

The optimizer must expose `zero_grad(set_to_none=True)` and `step()`. The driver
rejects changes to Gaussian count or parameter identities. No native GaussianAvatars
adapter is bundled: its separate refinement source was not recovered from the
inspected server. The CAP4D `gaussianavatars/` directory is CAP4D's implementation,
not a standalone GaussianAvatars integration.
