# Media processing

Run `python src/media_pipeline/main.py` after preprocessing, or use the full
runner `python src/main.py`.

1. Extract missing transcript ZIP directories.
2. Export the exact rows from `all_utterances_clean_mlu_balanced.csv` to
   `utterance_manifest.csv`, preserving child / transcript / utterance IDs.
3. Reuse existing clips at their existing paths, then download and cut
   missing intervals into mono 16-kHz MP3 when TalkBank access is available.
4. Finalize `data/media/media_download_report.csv` as the audio/reference handoff.

The selected CSV is authoritative; the media stage does not add other utterances
from the retained transcripts. Original clips remain intact. Canonical IDs stay
unchanged in metadata; only filesystem directory components are sanitized.

Use `--cookies-file path/to/cookies.json` or interactive browser login for new
downloads. `--reuse-only` completes available local pairs and reports missing
audio without network access (exit 2 if incomplete). `--dry-run` plans downloads
without encoding clips. Download locations come from the master workbook.

The report preserves raw and cleaned text, relative audio paths, child identity,
transcript identity, study design and processing / exclusion status. GenSEC reads
this report and its paired clips. See the root README for the input schema and
methodology.
