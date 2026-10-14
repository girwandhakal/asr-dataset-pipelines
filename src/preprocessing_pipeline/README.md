# Preprocessing

Run `python src/preprocessing_pipeline/main.py` from the repository root.
The full runner is `python src/main.py`.

- `helpers.py` reads the master and cached Redivis transcript metadata.
- `selection.py` resolves corpus-specific participant IDs, records exclusions,
  selects complete longitudinal transcripts under a cross-sectional-derived cap,
  and creates diagnostics / validation checks.
- `build_manifest.py` pools word counts per child, balances LT / TD candidates,
  and writes the final tables and workbook.
- `config.py` holds paths, exclusions and default cap settings.

CHAT files supply the exact usable utterances, text and timestamps; the old
Redivis utterance cache is not loaded. IDs are namespaced strings. MacWhinney
and unresolved participants are excluded with reasons, without deleting sources.

Options: `--refresh-data`, `--cap-center mean|median`, `--cap-sd 3`, `--seed 42`.
See the root README for sampling policy, outputs and GenSEC handoff.
