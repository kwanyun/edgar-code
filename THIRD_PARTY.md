# Third-party notices

- **Difix3D+**: https://github.com/nv-tlabs/Difix3D — the source snapshot retains
  a Difix3D README and a combined NVIDIA/Stability AI license. That license is
  preserved verbatim in `LICENSE.txt`; the old README is replaced with EDGAR usage.
- **Hugging Face Diffusers**: modified UNet/block implementations in
  `src/edgar/unet.py` and `unet_blocks.py` retain their Apache-2.0 copyright headers;
  the full license is in `licenses/Diffusers-Apache-2.0.txt`.
- **POSTER V2**: the required model subset in `src/edgar/vendor/poster/` comes from
  the supplied POSTER V2 checkout. Its MIT notice is retained in `licenses/POSTER-MIT.txt`.
  Standalone checkpoint loads and unused classifier/profiling paths were removed.
- **SD-Turbo**, **Sapiens**, and POSTER pretrained weights are external assets,
  not included or relicensed by this source package. Use the upstream providers'
  applicable terms and supply complete compatible weights.
- **RGBAvatar**: https://github.com/gapszju/RGBAvatar — the small native rendering
  helper and camera/optimizer conventions in `src/edgar/adapters/` are adapted from
  the supplied application. The MIT license is retained in `licenses/RGBAvatar-MIT.txt`.
  The native repository, custom CUDA rasterizer, and FLAME assets are external and
  retain their own terms.
- **CAP4D**: https://github.com/felixtaubner/cap4d — the adapter calls its native
  avatar, scene, renderer, and optimizers. CAP4D code, checkpoints, and third-party
  assets are not redistributed in this package.
- **CAFCA**: https://github.com/syntec-research/Cafca/blob/main/Dataset.md — the
  separately packaged synthetic triplets are adaptations under CC BY-NC-SA 4.0.
- **NeRSemble**: https://github.com/tobias-kirschstein/nersemble-data — its derived
  triplets are staged separately and are not authorized here for public redistribution.

This is an attribution inventory, not a replacement for the included licenses.
