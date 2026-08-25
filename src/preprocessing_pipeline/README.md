# Preprocessing pipeline

This folder contains the Child ASR preprocessing workflow. `main.py` is the small
command-line runner; `build_manifest.py` contains the processing functions;
`helpers.py` contains reusable loading and normalization functions; and
`config.py` contains paths, corpus mappings, and column settings.

From the repository root:

```powershell
pip install pandas scipy openpyxl xlsxwriter redivis
python src/preprocessing_pipeline/main.py
```

To run the complete dataset pipeline, including media processing, use the
overall runner instead:

```powershell
python src/main.py
```

Use `--refresh-data` to download fresh Redivis tables instead of using the
cached CSV files in `data/raw/`.

The workflow writes its CSV files and Excel workbook to `data/processed/`.
Running `build_manifest.py` directly remains supported for compatibility:

```powershell
python src/preprocessing_pipeline/build_manifest.py
```

## Helper functions

`helpers.py` contains documented functions for:

- ID and path cleanup: `clean_id`, `path_key`
- Safe column/value selection: `first_column`, `first_value`
- Child metadata fallbacks: `child_id_from_filename`, `fallback_age`
- Workbook and Redivis loading: `load_manifest`, `load_tables`
- Transcript and utterance preparation: `transcript_metadata`, `normalize_utterances`

Each function includes its inputs, output, and important matching or fallback
behavior in its docstring.
