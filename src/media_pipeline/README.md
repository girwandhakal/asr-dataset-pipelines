# Media pipeline

The complete dataset workflow is orchestrated from `src/main.py`. It runs the
preprocessing pipeline first and then this media pipeline. Run only this
folder's stages with `python src/media_pipeline/main.py`.

This folder contains the scripts that prepare transcripts and download
TalkBank media:

`main.py` is the pipeline entry point. It runs the three stages below in
order.

1. `extract_transcripts.py` extracts transcript ZIP archives into
   `data/transcripts`. Existing extracted folders are skipped.
2. `create_manifest.py` reads the selected workbook and local `.cha` files,
   then writes `data/processed/utterance_manifest.csv`,
   `data/out/manifest.log`, and `data/out/manifest_summary.txt`.
3. `download_media.py` reads `data/processed/utterance_manifest.csv`, downloads the required TalkBank
   media, creates audio clips, and writes them under `data/media`. It learns
   which TalkBank collection each corpus belongs to by reading the first
   column of `data/childes_talkbank_master_file.xlsx`, where a path such as
   `childes/Clinical-Eng/EllisWeismer/LT/66conv/11005` gives the collection
   (`Clinical-Eng`) and the corpus (`EllisWeismer`). Corpus names are not
   hardcoded, so pasting an updated master file is all that is needed to add a
   corpus. Override the location with `--master-file`, or merge extra entries
   over the derived mapping with a JSON `--collection-map`.

The extraction step is also available as a function for use by the pipeline:

```python
from src.media_pipeline.extract_transcripts import extract_archives

extract_archives()
```

The other steps can still be run from the repository root:

```powershell
python src/media_pipeline/create_manifest.py
python src/media_pipeline/download_media.py --cookies-file path/to/cookies.json
```

Run the complete pipeline with:

```powershell
python src/media_pipeline/main.py --cookies-file path/to/cookies.json
```

If `--cookies-file` is omitted, the downloader opens Chrome for interactive
TalkBank login. Use `--dry-run` to prepare the manifest and plan downloads
without creating clips.

Use `--help` on any script to see available path and authentication options.
