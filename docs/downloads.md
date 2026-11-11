# Downloads

The model and the two datasets are hosted separately from the code. **The exp6
checkpoint and CAFCA-only dataset are public and can be downloaded in a browser
without signing in.** The Both dataset remains private; sign in with a GitHub account that has access.
An unauthorized request can return 404 even when a private release exists.

| Package | Download | Contents |
| --- | --- | --- |
| EDGAR checkpoint | [Public release](https://github.com/kwanyun/edgar-checkpoint/releases/tag/v1.0) · [ZIP](https://github.com/kwanyun/edgar-checkpoint/releases/download/v1.0/edgar_exp6.zip) | Six-expert inference weights, model card, license, notices, checksum |
| CAFCA-only | [Public release](https://github.com/kwanyun/edgar-set-cafca/releases/tag/v1.0) | 22,420 triplets; 11 TAR shards and metadata ZIP |
| Both | [Release](https://github.com/kwanyun/edgar-set-both/releases/tag/v1.0) | 132,915 triplets; 51 TAR shards and metadata ZIP |

The checkpoint uses both data sources; a separate CAFCA-only checkpoint is not
provided. The Both dataset must remain private until permission to redistribute
NeRSemble-derived images is obtained. Retain each package's licenses and notices.

## Download with GitHub CLI

Install the [GitHub CLI](https://cli.github.com/) and authenticate; the private
Both dataset requires an account with repository access. The checkpoint and CAFCA-only can also be
downloaded without authentication through the browser links below.
Run the following commands from the EDGAR code checkout. Use new
destination directories so previous downloads or extracted files are preserved.

```bash
gh auth login
```

### Checkpoint

```bash
gh release download v1.0 --repo kwanyun/edgar-checkpoint \
  --pattern 'edgar_exp6.zip*' --dir packages/checkpoint
(cd packages/checkpoint && sha256sum -c edgar_exp6.zip.sha256)
python -m zipfile -e packages/checkpoint/edgar_exp6.zip checkpoints
(cd checkpoints && sha256sum -c SHA256SUMS)
```

ZIP compression preserves the original checkpoint bytes. The extracted file is
`checkpoints/edgar_exp6.pt`. Obtain Sapiens and the other external model assets
through the links in the main README.

### CAFCA-only

```bash
gh release download v1.0 --repo kwanyun/edgar-set-cafca \
  --pattern 'EDGAR-SET-CAFCA-only-metadata.zip*' --dir packages/CAFCA-only
(cd packages/CAFCA-only && sha256sum -c EDGAR-SET-CAFCA-only-metadata.zip.sha256)
python -m zipfile -e packages/CAFCA-only/EDGAR-SET-CAFCA-only-metadata.zip packages/CAFCA-only
gh release download v1.0 --repo kwanyun/edgar-set-cafca \
  --pattern '*.tar' --dir packages/CAFCA-only/shards
python tools/unpack_dataset.py --package packages/CAFCA-only --output data/CAFCA
```

### Both

```bash
gh release download v1.0 --repo kwanyun/edgar-set-both \
  --pattern 'EDGAR-SET-Both-metadata.zip*' --dir packages/Both
(cd packages/Both && sha256sum -c EDGAR-SET-Both-metadata.zip.sha256)
python -m zipfile -e packages/Both/EDGAR-SET-Both-metadata.zip packages/Both
gh release download v1.0 --repo kwanyun/edgar-set-both \
  --pattern '*.tar' --dir packages/Both/shards
python tools/unpack_dataset.py --package packages/Both --output data/Both
```

The unpacker verifies every TAR's SHA256 before extracting images. The two dataset
packages are self-contained; Both includes its own CAFCA shards. Allow room for
both the archives and the extracted images: about 12 GB each for CAFCA-only, and
55 GB each for Both.

## Download in a browser

Open the corresponding release and expand **Assets**. The checkpoint and CAFCA-only require no login.
For the checkpoint,
download the ZIP and its `.sha256` file. For a dataset, download its metadata ZIP,
the matching `.sha256` file, and every `.tar` asset. Extract the metadata ZIP into
the package directory and place all TAR files in its `shards/` subdirectory, then
use `tools/unpack_dataset.py` as above.

GitHub's automatically generated **Source code (zip)** and **Source code (tar.gz)**
contain repository documentation only. They do not contain model weights or dataset
shards; choose the named release assets listed above.
