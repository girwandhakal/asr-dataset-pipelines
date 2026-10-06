# CHILDES ASR Dataset Pipeline

This repository builds a complete child-speech dataset for automatic speech
recognition (ASR) experiments from
[TalkBank CHILDES](https://talkbank.org/childes/). It handles the dataset
compilation workflow end to end: selecting TalkBank records, matching
transcript and utterance data, calculating child-level statistics, balancing
the TD and language-disordered groups by mean length of utterance (MLU),
creating a timestamped utterance manifest, downloading linked TalkBank media,
and cutting the final audio clips used by ASR experiments.

The goal is not only to create a transcript manifest. The final result is a
reproducible, MLU-balanced child-speech dataset containing the metadata,
utterance selections, timestamps, and audio clips needed for ASR training,
validation, or evaluation.

The complete dataset workflow is:

```text
TalkBank search export
        |
        v
Preprocessing: clean, match, filter, summarize, and balance candidates
        |
        v
Media: extract CHAT archives, build utterance manifest, download media,
       and create timestamped 16-kHz mono MP3 clips
```

The overall runner is [`src/main.py`](src/main.py). It runs preprocessing
before media processing, so a new master file can be used to compile a new
dataset with the same processing and balancing rules.

## What the pipeline produces

The final ASR dataset is assembled from four connected components:

1. **Selected TalkBank records** from the master search export.
2. **Cleaned and MLU-balanced candidate and utterance tables** for choosing
   the participants and utterances included in the dataset.
3. **Timestamped utterance-manifest rows** linking each selected utterance to
   its source transcript and media interval.
4. **Audio clips** extracted from TalkBank media for the selected utterances.

The preprocessing stage:

- reads the TalkBank search export, called the **master file** in this project;
- matches selected transcripts to CHILDES transcript and utterance tables;
- keeps child-speaker utterances and attaches transcript and participant
  metadata;
- calculates utterance and MLU-related statistics;
- creates one candidate row per child;
- applies the MacWhinney age restriction used by this project;
- balances the LT and TD candidate groups by mean length of utterance so the
  ASR dataset is less confounded by group-level language-production
  differences; and
- writes cleaned tables, balancing diagnostics, and an Excel workbook.

The media stage completes the ASR dataset:

- extracts downloaded TalkBank transcript ZIP archives;
- reads local CHAT (`.cha`) transcripts;
- finds timestamped child utterances;
- writes an utterance-level manifest;
- derives each corpus's TalkBank collection from the master file, so no corpus
  names are hardcoded anywhere in the code;
- locates and downloads the linked TalkBank media; and
- cuts the requested intervals into mono 16-kHz MP3 clips.

## Requirements

### Software

- Python 3.10 or newer
- Google Chrome, needed for interactive TalkBank login
- `ffmpeg`, available on your system PATH
- A TalkBank account with access to the CHILDES transcript and media data you
  intend to use

Python dependencies and `ffmpeg` are installed together with conda; see
[`envs/`](envs/README.md).

TalkBank states that CHILDES transcript and media data generally require users
to sign in or register before downloading or browsing the data. Some corpora
or media may have additional access restrictions. Review the
[CHILDES access levels](https://talkbank.org/childes/access.html) before using
restricted data.

## Installation

From the repository root, create and activate the conda environment. This
installs Python 3.10, all Python dependencies, and `ffmpeg` in one step --
see [`envs/`](envs/README.md) for details.

```powershell
conda env create -f envs/environment.yml
conda activate asr-dataset-pipelines
```

Update the environment after `envs/environment.yml` changes with:

```powershell
conda env update -f envs/environment.yml --prune
```

### Verify ffmpeg

The media downloader invokes the `ffmpeg` command to create the final clips.
The conda environment installs it for you; verify it is on PATH:

```powershell
ffmpeg -version
```

If you need to use a different ffmpeg build, pass its executable path when
running the standalone media downloader with `--ffmpeg`.

## Prepare your TalkBank data

### 1. Create or log in to a TalkBank account

Go to [TalkBank](https://talkbank.org/) and log in. If you do not have an
account, register and confirm the registration email. Then open the
[CHILDES database](https://talkbank.org/childes/).

### 2. Search the CHILDES database

Open the [TalkBankDB database search](https://talkbank.org/0info/DB/) and enter
the fields for the population you want to include. The search criteria used
for the dataset associated with this repository were:

| Field | Value |
| --- | --- |
| Corpora | `clinical-eng`, `eng-na`, `eng-uk` |
| Language | `eng` |
| Media | `audio` and/or `video` |
| Age | `60` to `108` months |
| Group type | `TD`, `SLI`, `LT` |


### 3. Download the search results

Run the search and download/export the resulting transcript list. In this
project, that downloaded spreadsheet is the **master file**. Place it at:

```text
data/childes_talkbank_master_file.xlsx
```

If your downloaded file has a different name, rename or copy it to that exact
path. The preprocessing code supports the original positional workbook layout
and a named-column layout. The selection must provide transcript paths and
child/group identifiers. The important fields are:

```text
transcript_path
manifest_new_id   (or new_id / child_id)
language
media_type
recording_date_raw
pid
design
activity
source_group      (or group)
```

The workbook is an input selection file; it is not the final utterance
manifest.

#### How the corpus names are read from the master file

Both stages take the corpus names from the transcript-path column rather than
from a hardcoded list, so pasting an updated master file is all that is needed
to add or remove a corpus. TalkBank writes those paths as
`childes/<collection>/<corpus>/<subfolders>/<record>`, for example:

```text
childes/Clinical-Eng/EllisWeismer/LT/66conv/11005
childes/Eng-NA/OCSC/8/8040
```

The third component is the corpus (`EllisWeismer`, `OCSC`) and the second is
the TalkBank collection it lives in (`Clinical-Eng`, `Eng-NA`). Preprocessing
uses the corpus to group and match records; the media stage additionally uses
the collection to build the download URL
`https://media.talkbank.org/childes/<collection>/<corpus>/...`.

A corpus that appears in the manifest but not in the master file has no known
collection, so its clips are skipped and reported. A leading `childes/`
component is optional, and a header row is ignored if the export has one.

### 4. Download the transcript archives

The search export identifies the records, but the media pipeline also needs
the corresponding CHAT transcript archives. Download the transcript ZIP files
from the relevant TalkBank/CHILDES corpus pages and place the ZIP files in:

```text
data/transcripts/
```

For example:

```text
data/transcripts/ENNI.zip
data/transcripts/Forrester.zip
data/transcripts/MacWhinney.zip
```

The archive name should identify the corpus. The pipeline extracts each ZIP
into a folder with the same stem, for example `ENNI.zip` into
`data/transcripts/ENNI/`. Existing extracted folders are skipped on later
runs.

TalkBank's CLAN documentation describes the same general process: choose a
corpus, use its transcript-download link, and unzip the downloaded archive if
it was not automatically extracted. See the
[CLAN download instructions](https://talkbank.org/0info/manuals/CLAN.pdf).

### 5. Sign in to Redivis for the CHILDES tables

The preprocessing stage retrieves the CHILDES transcript and utterance tables
from Redivis. The code uses this Redivis resource directly:

```text
Organization: datapages
Dataset:     childes-db
Tables:      transcript, utterance
```

Create a Redivis account at [Redivis](https://redivis.com/) using an
institutional account, Google account, or passwordless login. Then search for
the `childes-db` dataset under the `datapages` organization and open it. If
Redivis shows an access or membership request, complete that request and make
sure your account has **Data** access, not only Metadata access.

If you want to verify access in the Redivis web interface, click **Analyze in
workflow** or **Add to workflow** on the `childes-db` dataset page and inspect
the `transcript` and `utterance` tables. This workflow step is optional for
this repository: the local code directly references `datapages/childes-db` and
does not require a particular Redivis workflow. Do not upload the tables to a
new dataset and do not add CSV files to this repository.

When the pipeline is run from a local computer, the Redivis Python client uses
an interactive browser/OAuth login the first time it needs the tables. Follow
the printed Redivis authentication instructions and authorize the client.
Redivis documents API tokens for unattended services; an API token is not
needed for this interactive pipeline workflow.

After the first successful retrieval, the pipeline automatically writes local
cache files to:

```text
data/raw/redivis_transcripts.csv
data/raw/redivis_utterances.csv
```

These caches are generated by the pipeline; users do not need to prepare them
manually. On later runs, the cached CSV files are reused automatically.

Use `--refresh-data` when you want the pipeline to authenticate with Redivis
again and replace the cached tables with fresh copies:

```powershell
python src/main.py --refresh-data
```

The repository's `.env.local` file is intentionally not documented with its
secret values and should not be committed or shared. The current pipeline
does not require a Redivis API token or read `TALKBANK_PASSWORD` directly;
TalkBank media authentication is handled through an interactive browser login
or an exported cookie file.

## Run the complete pipeline

From the repository root:

```powershell
python src/main.py
```

When media downloads are needed and no cookie file is supplied, the pipeline
opens Chrome. Log in to TalkBank in that browser window, return to the
terminal, and press Enter when prompted.

### Common run options

Refresh the automatically generated Redivis caches:

```powershell
python src/main.py --refresh-data
```

Use an exported browser cookie file instead of interactive login:

```powershell
python src/main.py --cookies-file path/to/talkbank-cookies.json
```

Plan and validate media work without creating audio clips:

```powershell
python src/main.py --dry-run
```

Options can be combined:

```powershell
python src/main.py --refresh-data --cookies-file path/to/talkbank-cookies.json
```

Both runners read the master file from its standard location,
`data/childes_talkbank_master_file.xlsx`. To read it from somewhere else, or
to add collections for corpora the master file does not cover, run the
standalone downloader described below with `--master-file` or
`--collection-map`.

Run the overall pipeline as a module:

```powershell
python -m src.main
```

## Run one stage only

Preprocessing only:

```powershell
python src/preprocessing_pipeline/main.py
```

Preprocessing with fresh Redivis tables:

```powershell
python src/preprocessing_pipeline/main.py --refresh-data
```

Media processing only:

```powershell
python src/media_pipeline/main.py
```

Media processing with cookies or dry-run mode:

```powershell
python src/media_pipeline/main.py --cookies-file path/to/talkbank-cookies.json
python src/media_pipeline/main.py --dry-run
```

The media downloader can also be run on its own, which is the only way to
reach its full option set:

```powershell
python src/media_pipeline/download_media.py --master-file path/to/master.xlsx
python src/media_pipeline/download_media.py --collection-map path/to/collections.json
python src/media_pipeline/download_media.py --ffmpeg path/to/ffmpeg.exe
```

`--collection-map` takes a JSON object of `{"Corpus": "Collection"}` entries
that are merged over the mapping derived from the master file. It is only
needed when the master file itself cannot supply a corpus's collection.

Each script supports `--help`.

## Output layout

After a successful run, the important outputs are:

```text
data/
|-- childes_talkbank_master_file.xlsx        # TalkBank search export (input)
|-- media/
|   |-- Corpus/GROUP/child_id/*.mp3           # Final timestamped MP3 clips
|   `-- media_download_report.csv             # Per-clip download outcomes
|-- out/
|   |-- manifest.log                          # Transcript/utterance skip log
|   `-- manifest_summary.txt                  # Human-readable manifest report
|-- processed/
|   |-- all_candidate_table.csv               # Candidates before balancing
|   |-- all_candidate_table_mlu_balanced.csv  # Candidates after balancing
|   |-- all_utterances_clean.csv              # Cleaned utterances before balancing
|   |-- all_utterances_clean_mlu_balanced.csv # Utterances for retained candidates
|   |-- mlu_balance_removal_log.csv           # TD removals during balancing
|   |-- mlu_balance_summary.csv               # Before/after balance statistics
|   |-- mlu_balancing_results.xlsx            # Combined balancing workbook
|   `-- utterance_manifest.csv                # Timestamped utterance manifest
|-- raw/
|   |-- redivis_transcripts.csv               # Cached transcript table
|   `-- redivis_utterances.csv                # Cached utterance table
`-- transcripts/
    |-- CorpusName.zip                        # Downloaded TalkBank archive
    `-- CorpusName/                           # Extracted CHAT files
```

The timestamped manifest contains the rows used to request media clips. The
balancing workbook contains candidate and utterance tables used to determine
which records are retained.

### Reading the media download report

Every requested clip gets one row in `data/media/media_download_report.csv`,
and the run prints the same tally as `Status counts`:

| Status | Meaning |
| --- | --- |
| `created` | The clip was downloaded and cut on this run. |
| `already_exists` | The clip was already on disk and was left alone. |
| `planned` | `--dry-run` only: the media was located but no clip was cut. |
| `missing_media` | The media file could not be found or reached on TalkBank. |
| `skipped` | The corpus is not in the master file, or the row has no match in the balancing workbook. |
| `error` | Download or ffmpeg conversion failed for that clip. |

The media stage exits with status `2` when any row is `missing_media`,
`skipped`, or `error`, and `0` otherwise. Reruns are safe and resumable:
`already_exists` rows are detected before any network request, so a rerun only
retries the clips that did not complete.

## How balancing works

The preprocessing code maps source groups as follows:

```text
TD  -> TD
LT  -> LT
SLI -> LT
```

It summarizes child utterances into candidate-level statistics and uses MLU
to reduce the difference between LT and TD. The balancing procedure removes
TD candidates at the most extreme MLU values while the removal improves the
absolute group mean difference. It stops when the groups are no longer
significantly different at the configured threshold, when the minimum TD
group size would be reached, when another removal would not improve the
difference, or when the removal limit is reached.

MacWhinney candidates are additionally retained only when their derived mean
age is between 5 and 9 years, inclusive. Review
`data/processed/mlu_balance_summary.csv` and
`data/processed/mlu_balance_removal_log.csv` before using the final candidate
set in an experiment.

## Reproducibility and data-use notes

TalkBank content can change. Record the following for every dataset release:

- the date the TalkBank search was run;
- the exact search criteria;
- the downloaded master-file name and checksum;
- the transcript ZIP archives used and their checksums;
- whether cached or refreshed Redivis tables were used;
- the pipeline command and options; and
- the resulting balancing summary.

TalkBank recommends citing the specific corpus references listed in each
corpus's documentation, in addition to the general CHILDES/TalkBank
references. Review the official
[TalkBank citation rules](https://talkbank.org/0share/citation.html) and
[ground rules](https://talkbank.org/0share/rules.html) before distributing
results. The ground rules also describe licensing and restrictions on use of
TalkBank materials.

For historical reproducibility, TalkBank documents database versioning and
date-based retrieval of previous corpus states in its
[database versioning guidance](https://talkbank.org/0info/versions.html).

## Troubleshooting

### `Manifest is missing columns`

The master file does not match one of the supported TalkBank export layouts.
Inspect the spreadsheet headers and ensure it contains `transcript_path`, an
ID field (`manifest_new_id`, `new_id`, or `child_id`), and `source_group`
or `group`, along with the remaining metadata columns listed above.

### Redivis authentication or access fails

Install the dependencies, create or log in to a Redivis account, and confirm
that the account has Data access to `datapages/childes-db`. Then run the
pipeline. The Redivis client will prompt for interactive authentication and
the pipeline will create the local caches automatically:

```powershell
python src/main.py
```

Use `--refresh-data` to replace existing caches with fresh Redivis tables.

### `No corpus produced usable utterances`

Check that the selected corpus names in the master file match the corpus names
in the downloaded Redivis tables and that the required transcript and utterance
columns are present.

### `Corpus not listed in childes_talkbank_master_file.xlsx`

The media stage derives each corpus's TalkBank collection from the first
column of the master file, so a corpus that appears in the manifest but not in
the master file has no download location and is skipped. Re-export the master
file so it covers every selected corpus, then rerun the pipeline. The skipped
rows are listed in `data/media/media_download_report.csv`.

### Transcript files are missing

Confirm that the relevant corpus ZIP archives are in `data/transcripts/` and
that their extracted folders contain `.cha` files. Run the media stage with
`--help` to inspect its options.



### TalkBank returns a login or HTML page instead of media

Authenticate again in the Chrome window, or export fresh TalkBank cookies and
pass them with `--cookies-file`. Some corpora or media may require additional
TalkBank approval.

### `ffmpeg` is not recognized

Install ffmpeg and ensure `ffmpeg -version` works in the same terminal used to
run the pipeline. For a nonstandard installation path, run the standalone
downloader with `--ffmpeg path/to/ffmpeg`.

## Official resources

- [TalkBank](https://talkbank.org/)
- [CHILDES](https://talkbank.org/childes/)
- [TalkBankDB database search](https://talkbank.org/0info/DB/)
- [CHILDES access levels](https://talkbank.org/childes/access.html)
- [CHILDES corpus index](https://talkbank.org/childes/index.html)
- [CLAN manual](https://talkbank.org/0info/manuals/CLAN.pdf)
- [TalkBank citation rules](https://talkbank.org/0share/citation.html)
- [TalkBank ground rules](https://talkbank.org/0share/rules.html)


## Final ground-truth handoff

`data/media/media_download_report.csv` is the authoritative clip/reference CSV.
`utterance` is final cleaned ground truth; `raw_utterance` preserves CHAT text.
`utterance_id` identifies the clip and `audio_path` is relative to `data/media`,
so the file works on both Windows and HPC. GenSEC consumes only rows with
a nonempty `utterance` and `audio_path`, without editing the reference.
Rows with empty cleaned references remain visible and are skipped by GenSEC. Cleaning preserves scoped words, clipped forms,
compound words and repetitions, and excludes CHAT fillers/fragments/events,
unintelligibility markers, and unspoken forms. Nasal hum spellings use `mm`.

To finalize an existing report without downloading or changing audio, run
`python src/media_pipeline/ground_truth.py`. A one-time
`media_download_report.before_ground_truth_v1.csv` backup preserves the old CSV.
The media downloader finalizes its report automatically on future runs.

MLU balancing selects children and transcripts; timestamp extraction rereads
selected CHAT transcripts. It does not establish an utterance-level bijection
with the Redivis balancing table. Final MLU balance should be checked on the
retained reference/audio rows before reporting it as an analysis-sample property.
