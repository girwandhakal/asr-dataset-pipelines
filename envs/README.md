# Environment

One-step conda setup for this repository. This is the supported way to
install the pipeline's dependencies.

```powershell
conda env create -f envs/environment.yml
conda activate asr-dataset-pipelines
```

This installs everything the pipeline needs, including `ffmpeg` and Python
3.10 -- you don't need either already on your system for this path. Update
the environment after this file changes with:

```powershell
conda env update -f envs/environment.yml --prune
```

Two things conda can't set up for you, per the root
[README.md](../README.md#requirements):

- **Google Chrome**, needed for the interactive TalkBank login used by
  `download_media.py`.
- **A TalkBank account** with access to the CHILDES data you intend to use.
