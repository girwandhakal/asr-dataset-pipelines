"""Download only the TalkBank media segments present in a timestamp manifest.

The script processes one transcript at a time:

1. Read timestamped rows from ``data/processed/utterance_manifest.csv``.
2. Use ``mlu_balancing_results.xlsx`` to recover group and child ID metadata.
3. Use ``childes_talkbank_master_file.xlsx`` to learn which TalkBank
   collection each corpus lives in. Corpus names are never hardcoded: they are
   read from the master file's path column, so adding a corpus to the study
   only requires pasting an updated master file before running the pipeline.
4. Resolve the transcript's TalkBank media file.
5. Download that one source file into a temporary directory.
6. Convert only the requested ranges to mono 16-kHz MP3 clips with ffmpeg.
7. Delete the temporary source file before moving to the next transcript.

TalkBank access normally requires an authenticated session.  By default the
script opens Chrome for a manual TalkBank login, copies the Selenium cookies
into a requests session, and then downloads media with requests.  An exported
browser cookie file can be supplied with ``--cookies-file`` for repeatable
runs.  The script never stores the source media under the output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import pandas as pd
import requests
from tqdm.auto import tqdm

try:
    from .ground_truth import finalize_rows, REPORT_COLUMNS
except ImportError:
    from ground_truth import finalize_rows, REPORT_COLUMNS


DEFAULT_MASTER_FILE = Path(__file__).resolve().parents[2] / "data" / "childes_talkbank_master_file.xlsx"
DEFAULT_MEDIA_ROOT = "https://media.talkbank.org/childes"

MEDIA_EXTENSIONS = (".wav", ".mp3", ".m4a", ".mp4", ".mov", ".avi", ".webm", ".ogg")
MEDIA_CONTENT_TYPES = (
    "audio/",
    "video/",
    "application/octet-stream",
    "application/ogg",
    "application/force-download",
    "application/x-download",
)


def clean_component(value: object, fallback: str = "unknown") -> str:
    """Make a value safe for use as one filesystem path component."""
    text = str(value).strip() if value is not None else ""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return text or fallback


def path_key(path: object, corpus: str) -> str:
    """Normalize a transcript path for corpus-relative matching."""
    parts = [part for part in str(path).replace("\\", "/").split("/") if part]
    parts = [re.sub(r"\.(cha|xml)$", "", part, flags=re.IGNORECASE) for part in parts]
    corpus_lower = corpus.casefold()
    start = next((i for i, part in enumerate(parts) if part.casefold() == corpus_lower), 0)
    return "/".join(part.casefold() for part in parts[start:])


def filename_stem(path: object) -> str:
    """Return a case-insensitive filename stem from a path-like value."""
    return Path(str(path).replace("\\", "/")).stem.casefold()


def read_cookies(path: Path) -> dict[str, str]:
    """Read either browser-exported JSON cookies or Netscape cookies."""
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        cookies = {}
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) >= 7:
                cookies[fields[5]] = fields[6]
        if not cookies:
            raise ValueError(f"Could not parse cookies file: {path}")
        return cookies

    if isinstance(payload, dict):
        if "cookies" in payload:
            payload = payload["cookies"]
        else:
            return {str(key): str(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return {
            str(item["name"]): str(item["value"])
            for item in payload
            if isinstance(item, dict) and item.get("name")
        }
    raise ValueError(f"Unsupported cookies format: {path}")


def browser_login(login_url: str, chrome_user_data: Path | None) -> dict[str, str]:
    """Follow the Selenium login flow used by reference/file.py."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError as error:
        raise RuntimeError(
            "Browser login requires selenium. Install requirements.txt "
            "or pass --cookies-file instead."
        ) from error

    options = Options()
    if chrome_user_data is not None:
        options.add_argument(f"--user-data-dir={chrome_user_data.resolve()}")
    print(f"Opening TalkBank login page: {login_url}")
    driver = webdriver.Chrome(options=options)
    try:
        driver.get(login_url)
        print("Chrome is open. Complete the TalkBank login in Chrome.")
        input("Log in to TalkBank in the browser, then press Enter here... ")
        cookies = {
            cookie["name"]: cookie["value"]
            for cookie in driver.get_cookies()
            if cookie.get("name")
        }
        print(f"Collected {len(cookies)} browser cookies; closing Chrome.")
        return cookies
    finally:
        driver.quit()


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        """Initialize an empty collection of discovered HTML links."""
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Collect ``href`` values from HTML anchor tags."""
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def split_master_path(value: object, media_root: str = DEFAULT_MEDIA_ROOT) -> tuple[str, str] | None:
    """Split one master-file path into its ``(collection, corpus)`` pair.

    Master-file paths look like
    ``childes/Clinical-Eng/EllisWeismer/LT/66conv/11005``: the leading
    component repeats the media root's own last segment, the next component
    is the TalkBank collection, and the one after it is the corpus. Paths that
    already start at the collection are accepted too, so a header cell or any
    other non-path value simply returns ``None``.
    """
    parts = [part for part in str(value).strip().replace("\\", "/").split("/") if part]
    root_tail = urlparse(media_root).path.strip("/").rsplit("/", 1)[-1]
    if parts and root_tail and parts[0].casefold() == root_tail.casefold():
        parts = parts[1:]
    if len(parts) < 3:
        return None
    return parts[0], parts[1]


def load_collection_map(
    master_file: Path = DEFAULT_MASTER_FILE,
    override: Path | None = None,
    media_root: str = DEFAULT_MEDIA_ROOT,
) -> dict[str, str]:
    """Derive corpus to TalkBank collection names from the master workbook.

    The corpus names are read from the workbook rather than hardcoded, so
    adding a corpus to the study only requires pasting an updated master file.

    Args:
        master_file: Workbook whose first column holds TalkBank media paths.
        override: Optional JSON file merged over the derived mapping.
        media_root: Media root URL, used to recognize the leading path
            component the workbook repeats.

    Returns:
        A mapping of corpus name to TalkBank collection name.

    Raises:
        FileNotFoundError: If the master file does not exist.
        ValueError: If no usable TalkBank paths are found in it.
    """
    if not master_file.exists():
        raise FileNotFoundError(
            f"TalkBank master file not found: {master_file}. Place the "
            "childes_talkbank_master_file.xlsx export there, or pass "
            "--master-file."
        )
    try:
        frame = pd.read_excel(master_file, header=None, usecols=[0], dtype=object)
    except Exception as error:
        raise RuntimeError(
            f"Could not read {master_file}: {type(error).__name__}: {error}. "
            "Reading it requires openpyxl, and the workbook must not be open "
            "in Excel."
        ) from error

    mapping: dict[str, str] = {}
    conflicts: dict[str, set[str]] = defaultdict(set)
    for value in frame.iloc[:, 0].tolist():
        pair = split_master_path(value, media_root)
        if pair is None:
            continue  # Header cells and short paths carry no corpus name.
        collection, corpus = pair
        conflicts[corpus].add(collection)
        mapping.setdefault(corpus, collection)
    for corpus, found in sorted(conflicts.items()):
        if len(found) > 1:
            raise ValueError(
                f"Master file maps {corpus} to more than one collection: {sorted(found)}"
            )
    if not mapping:
        raise ValueError(
            f"No TalkBank paths found in the first column of {master_file}."
        )

    if override is not None:
        mapping.update(json.loads(override.read_text(encoding="utf-8")))
    return mapping


def load_manifest(path: Path) -> list[dict[str, object]]:
    """Read and validate timestamped manifest rows from CSV."""
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                start = float(row["start_seconds"])
                end = float(row["end_seconds"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid timestamps on manifest row {row_number}") from error
            if start < 0 or end <= start:
                raise ValueError(f"Invalid interval on manifest row {row_number}: {start}, {end}")
            rows.append(
                {
                    "manifest_row": row_number,
                    "corpus": row["corpus"].strip(),
                    "filepath": row["filepath"].strip().replace("\\", "/"),
                    "start_seconds": start,
                    "end_seconds": end,
                    "utterance": row.get("utterance", ""),
                    "raw_utterance": row.get("raw_utterance", row.get("utterance", "")),
                }
            )
    return rows


def load_selection_metadata(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Map a corpus/transcript path to group and child ID from the workbook."""
    try:
        frame = pd.read_excel(path, sheet_name="Balanced Utterances", dtype=object)
    except Exception as error:
        raise RuntimeError(
            f"Could not read {path}. The selection workbook must contain the "
            "'Balanced Utterances' sheet and requires pandas/openpyxl."
        ) from error

    required = {"corpus", "filename", "group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Selection workbook is missing columns: {sorted(missing)}")

    child_column = next(
        (column for column in ("new_id", "manifest_new_id", "child_id") if column in frame.columns),
        None,
    )
    if child_column is None:
        raise ValueError("Selection workbook needs new_id, manifest_new_id, or child_id")

    metadata: dict[tuple[str, str], dict[str, str]] = {}
    grouped = frame.groupby("corpus", dropna=False)
    for corpus_value, corpus_frame in grouped:
        corpus = str(corpus_value).strip()
        for _, item in corpus_frame.iterrows():
            filename = str(item["filename"]).strip()
            key = (corpus.casefold(), path_key(filename, corpus))
            record = {
                "group": str(item["group"]).strip().upper(),
                "child_id": clean_component(item[child_column]),
            }
            previous = metadata.get(key)
            if previous is not None and previous != record:
                raise ValueError(f"Conflicting selection metadata for {corpus}: {filename}")
            metadata[key] = record

            # A basename fallback is useful when one source uses Corpus/file.cha
            # and the other uses only file.cha. It is used only when unique.
            basename_key = (corpus.casefold(), filename_stem(filename))
            if basename_key not in metadata:
                metadata[basename_key] = record
            elif metadata[basename_key] != record:
                metadata.pop(basename_key, None)
    return metadata


def transcript_media_name(transcript_path: Path | None) -> str | None:
    """Read @Media when the local CHAT transcript is available."""
    if transcript_path is None or not transcript_path.exists():
        return None
    pattern = re.compile(r"^@Media:\s*([^,\s]+)", re.IGNORECASE)
    for line in transcript_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if match:
            return match.group(1)
    return None


def media_candidates(media_name: str, media_directory: str) -> list[str]:
    """Build likely TalkBank media URLs for a name and known extensions."""
    name = Path(media_name).name
    stem = Path(name).stem if Path(name).suffix else name
    if Path(name).suffix:
        names = [name]
    else:
        names = [stem + extension for extension in MEDIA_EXTENSIONS]
    return [urljoin(media_directory.rstrip("/") + "/", name) for name in names]


def is_media_response(response: requests.Response) -> bool:
    """Return whether an HTTP response appears to contain downloadable media."""
    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
    # TalkBank's downloadable links may use application/force-download or
    # another generic content type. reference/file.py accepts HTTP 200 and
    # writes the response, so reject only HTML login/listing pages here.
    return response.status_code == 200 and not content_type.startswith(
        ("text/html", "application/xhtml", "application/json")
    )


def add_save_parameter(url: str) -> str:
    """Add TalkBank's download parameter when it is not already present."""
    if re.search(r"(?:[?&])f=save(?:&|$)", url, flags=re.IGNORECASE):
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}f=save"


def url_is_under(url: str, root: str) -> bool:
    """Check that a URL belongs to a permitted root URL."""
    url_parts = urlparse(url)
    root_parts = urlparse(root)
    if url_parts.netloc.casefold() != root_parts.netloc.casefold():
        return False
    url_path = url_parts.path.rstrip("/")
    root_path = root_parts.path.rstrip("/")
    return url_path == root_path or url_path.startswith(root_path + "/")


def link_name(url: str) -> str:
    """Return the decoded final path component of a URL."""
    return Path(unquote(urlparse(url).path)).name


def probe_media_url(session: requests.Session, url: str) -> str | None:
    """Probe one candidate URL and return it only when it serves media."""
    candidate = add_save_parameter(url)
    try:
        response = session.get(candidate, stream=True, timeout=45)
    except requests.RequestException:
        return None
    try:
        return candidate if is_media_response(response) else None
    finally:
        response.close()


def resolve_media_url(
    session: requests.Session,
    corpus_directory: str,
    media_name: str,
    preferred_directory: str | None = None,
    max_directory_pages: int = 2000,
) -> tuple[str | None, int, str]:
    """Traverse a corpus directory tree and resolve ``@Media``.

    TalkBank media is organized as ``collection/corpus/subfolders/file``.
    The transcript path gives us a preferred subfolder, but the corpus root is
    also traversed so this still works when local transcript folders differ
    from the server layout.
    """
    corpus_directory = corpus_directory.rstrip("/")
    wanted_stem = Path(media_name).stem.casefold()
    queue: list[str] = []
    for directory in (preferred_directory, corpus_directory):
        if directory:
            directory = directory.rstrip("/")
            if directory not in queue and url_is_under(directory, corpus_directory):
                queue.append(directory)

    visited: set[str] = set()
    diagnostic = "no directory pages were inspected"
    matching_links = 0
    while queue and len(visited) < max_directory_pages:
        directory = queue.pop(0)
        if directory in visited:
            continue
        visited.add(directory)

        # Try the common known media extensions before parsing the listing.
        for candidate in media_candidates(media_name, directory):
            resolved = probe_media_url(session, candidate)
            if resolved:
                return resolved, len(visited), f"matched media in {directory}"

        try:
            response = session.get(directory, timeout=45)
            response.raise_for_status()
        except requests.RequestException as error:
            diagnostic = f"could not read {directory}: {type(error).__name__}: {error}"
            continue
        if "text/html" not in response.headers.get("content-type", "").lower():
            diagnostic = f"{directory} did not return an HTML directory listing"
            response.close()
            continue

        parser = LinkParser()
        page_text = response.text
        parser.feed(page_text)
        response.close()
        page_lower = page_text.casefold()
        if not parser.links:
            if any(marker in page_lower for marker in ("authmodals", "sign in", "log in", "login")):
                diagnostic = f"{directory} returned a TalkBank authentication page"
            else:
                diagnostic = f"{directory} returned HTML with no directory links"
            continue
        for href in parser.links:
            absolute = urljoin(directory + "/", href).split("#", 1)[0]
            if not url_is_under(absolute, corpus_directory):
                continue

            basename = link_name(absolute)
            suffix = Path(basename).suffix.casefold()
            if Path(basename).stem.casefold() == wanted_stem and (
                suffix in MEDIA_EXTENSIONS or not suffix
            ):
                matching_links += 1
                resolved = probe_media_url(session, absolute)
                if resolved:
                    return resolved, len(visited), f"matched media link in {directory}"

            # Directory listings generally end directory links with '/'.
            # The suffix check also prevents CHAT/XML files from entering the
            # crawl queue.
            if href.endswith("/") or not suffix:
                child_directory = absolute.rstrip("/")
                if child_directory not in visited and child_directory not in queue:
                    queue.append(child_directory)
        diagnostic = (
            f"scanned {directory}; found {len(parser.links)} links and "
            f"{matching_links} matching media filename link(s)"
        )

    if len(visited) >= max_directory_pages and queue:
        diagnostic = f"reached --max-directory-pages={max_directory_pages}"
    return None, len(visited), diagnostic


def download_source(session: requests.Session, url: str, destination: Path) -> int:
    """Stream source media to a temporary file and return bytes downloaded."""
    with session.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" in content_type:
            raise RuntimeError("TalkBank returned an HTML login/access page instead of media")
        total_bytes = response.headers.get("content-length")
        progress = tqdm(
            total=int(total_bytes) if total_bytes and total_bytes.isdigit() else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"  Download {destination.name}",
            leave=False,
        )
        bytes_downloaded = 0
        try:
            with destination.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
                        bytes_downloaded += len(chunk)
                        progress.update(len(chunk))
        finally:
            progress.close()
        return bytes_downloaded


def run_ffmpeg(
    source: Path,
    output: Path,
    start: float,
    end: float,
    ffmpeg: str,
    timeout: float,
) -> None:
    """Extract one audio interval into a mono 16-kHz MP3 file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(source),
        "-t",
        f"{end - start:.6f}",
        "-map",
        "0:a:0?",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-q:a",
        "2",
        "-y",
        str(output),
    ]
    # Keep the final media extension at the end so ffmpeg can infer the
    # output format. ``clip.mp3.part`` makes ffmpeg report "Unable to choose
    # an output format"; ``clip.part.mp3`` remains temporary but is valid.
    temporary = output.with_name(f"{output.stem}.part{output.suffix}")
    try:
        subprocess.run(command[:-1] + [str(temporary)], check=True, timeout=timeout)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def clip_name(corpus: str, filepath: str, start: float, end: float, ordinal: int) -> str:
    """Create a stable short filename for one transcript audio interval."""
    transcript_hash = hashlib.blake2s(
        f"{corpus}|{filepath}".encode("utf-8"), digest_size=4
    ).hexdigest()
    start_ms = round(start * 1000)
    end_ms = round(end * 1000)
    # The path hash identifies the transcript; integer milliseconds preserve
    # the useful time information while keeping filenames short.
    return f"{transcript_hash}_{start_ms}_{end_ms}_{ordinal}.mp3"


def build_parser() -> argparse.ArgumentParser:
    """Create the standalone downloader command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    repository = Path(__file__).resolve().parents[2]
    data_dir = repository / "data"
    parser.add_argument(
        "--manifest",
        type=Path,
        default=data_dir / "processed" / "utterance_manifest.csv",
    )
    parser.add_argument(
        "--selection-workbook",
        type=Path,
        default=data_dir / "processed" / "mlu_balancing_results.xlsx",
        help="Workbook used to recover group and child ID metadata.",
    )
    parser.add_argument("--transcripts-dir", type=Path, default=data_dir / "transcripts")
    parser.add_argument("--output-dir", type=Path, default=data_dir / "media")
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument(
        "--cookies-file",
        type=Path,
        help="JSON or Netscape cookies exported from an authenticated TalkBank session.",
    )
    parser.add_argument(
        "--login-url",
        default="https://media.talkbank.org/childes/Eng-NA",
        help="TalkBank page opened for browser login.",
    )
    parser.add_argument(
        "--chrome-user-data",
        type=Path,
        default=None,
        help="Optional Chrome profile directory for browser-login sessions.",
    )
    parser.add_argument(
        "--master-file",
        type=Path,
        default=data_dir / "childes_talkbank_master_file.xlsx",
        help="TalkBank master workbook; its first column supplies the corpus names.",
    )
    parser.add_argument(
        "--collection-map",
        type=Path,
        default=None,
        help="Optional JSON file merged over the collections derived from the master file.",
    )
    parser.add_argument(
        "--media-root",
        default=DEFAULT_MEDIA_ROOT,
        help="TalkBank media root; override for a different TalkBank collection.",
    )
    parser.add_argument(
        "--max-directory-pages",
        type=int,
        default=2000,
        help="Maximum TalkBank directory pages to inspect per transcript.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--ffmpeg-timeout",
        type=float,
        default=600,
        help="Maximum seconds allowed for one clip conversion (default: 600).",
    )
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def download_media(
    manifest: Path,
    selection_workbook: Path,
    transcripts_dir: Path,
    output_dir: Path,
    *,
    cookies_file: Path | None = None,
    login_url: str = "https://media.talkbank.org/childes/Eng-NA",
    chrome_user_data: Path | None = None,
    master_file: Path = DEFAULT_MASTER_FILE,
    collection_map: Path | None = None,
    media_root: str = DEFAULT_MEDIA_ROOT,
    max_directory_pages: int = 2000,
    ffmpeg: str = "ffmpeg",
    ffmpeg_timeout: float = 600,
    timeout: float = 45,
    tmp_dir: Path | None = None,
    dry_run: bool = False,
) -> int:
    """Download and convert the requested TalkBank media clips.

    This is the function interface used by the pipeline. It reads timestamped
    manifest rows, joins them with selection metadata, locates each source
    recording on TalkBank, and creates mono 16-kHz MP3 clips with ffmpeg.
    Existing clips are reused. Authentication and network access are skipped
    entirely when every requested clip already exists.

    Args:
        manifest: CSV containing corpus, transcript, and time-range rows.
        selection_workbook: Excel workbook containing group and child metadata.
        transcripts_dir: Local extracted CHAT transcript directory.
        output_dir: Directory where clips and ``media_download_report.csv``
            are written.
        cookies_file: Optional browser-exported TalkBank cookie file. If not
            provided and downloads are needed, interactive browser login is
            used.
        login_url: TalkBank page used for interactive login.
        chrome_user_data: Optional Chrome profile directory.
        master_file: TalkBank master workbook whose first column supplies the
            corpus and collection names.
        collection_map: Optional JSON file merged over those names.
        media_root: Root URL for TalkBank media collections.
        max_directory_pages: Maximum directory pages searched per transcript.
        ffmpeg: ffmpeg executable name or path.
        ffmpeg_timeout: Maximum seconds allowed for one clip conversion.
        timeout: Reserved request timeout setting for downloader configuration.
        tmp_dir: Optional directory for temporary source media.
        dry_run: If true, locate and plan downloads without creating clips.

    Returns:
        ``0`` when no media errors, missing files, or skipped records occur;
        ``2`` when one or more records could not be processed.

    Raises:
        FileNotFoundError: If an input manifest, workbook, or cookie file is
            missing.
        RuntimeError: If TalkBank authentication or media processing fails.
    """
    tqdm.write("Loading timestamp manifest...")
    manifest_rows = load_manifest(manifest)
    tqdm.write(f"Loaded {len(manifest_rows):,} manifest segments from {manifest}")
    selection = load_selection_metadata(selection_workbook)
    tqdm.write(f"Loaded selection metadata from {selection_workbook}")
    collections = load_collection_map(master_file, collection_map, media_root)
    tqdm.write(
        f"Derived {len(collections)} corpus collections from {master_file}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    report: list[dict[str, object]] = []
    for row in manifest_rows:
        corpus = str(row["corpus"])
        key = (corpus.casefold(), path_key(row["filepath"], corpus))
        metadata = selection.get(key) or selection.get((corpus.casefold(), filename_stem(row["filepath"])))
        if metadata is None:
            report.append({**row, "status": "skipped", "reason": "No group/child metadata match in selection workbook"})
            continue
        row = {**row, **metadata}
        grouped[(corpus, str(row["filepath"]))].append(row)

    tqdm.write(
        f"Prepared {len(grouped):,} transcript media groups from "
        f"{len(manifest_rows):,} manifest segments"
    )

    # Build the expected output paths before authenticating with TalkBank.
    # This lets a completed run finish without opening Chrome or making any
    # network requests.
    targets_by_group = {}
    for group_key, rows in grouped.items():
        corpus, filepath = group_key
        targets = []
        for ordinal, row in enumerate(rows, start=1):
            name = clip_name(
                corpus,
                filepath,
                float(row["start_seconds"]),
                float(row["end_seconds"]),
                ordinal,
            )
            target = (
                output_dir
                / clean_component(corpus)
                / clean_component(row["group"])
                / clean_component(row["child_id"])
                / name
            )
            targets.append((row, target))
        targets_by_group[group_key] = targets

    needs_download = any(
        not target.exists() or target.stat().st_size == 0
        for targets in targets_by_group.values()
        for _, target in targets
    )

    if needs_download:
        if cookies_file is not None:
            cookies = read_cookies(cookies_file)
        else:
            cookies = browser_login(login_url, chrome_user_data)

        session = requests.Session()
        session.cookies.update(cookies)
        session.headers.update({"User-Agent": "child-asr-media-downloader/1.0"})
    else:
        session = None
        tqdm.write("All requested clips already exist; skipping TalkBank login and download.")

    transcript_items = sorted(grouped.items())
    for group_number, ((corpus, filepath), rows) in enumerate(
        tqdm(
            transcript_items,
            total=len(transcript_items),
            desc="Processing transcript media",
            unit="transcript",
        ),
        start=1,
    ):
        tqdm.write(
            f"[{group_number}/{len(transcript_items)}] {corpus} :: {filepath} "
            f"({len(rows)} requested clips)"
        )
        targets = targets_by_group[(corpus, filepath)]
        if all(target.exists() and target.stat().st_size > 0 for _, target in targets):
            tqdm.write("  SKIP download: all requested clips already exist")
            for row, target in targets:
                report.append({**row, "status": "already_exists", "output": str(target)})
            continue

        collection = collections.get(corpus)
        if not collection:
            tqdm.write(
                f"  SKIP: {corpus} is not listed in the master file {master_file.name}"
            )
            for row in rows:
                report.append({**row, "status": "skipped", "reason": f"Corpus not listed in {master_file.name}"})
            continue

        relative = filepath.replace("transcripts/", "", 1).strip("/")
        local_transcript = transcripts_dir / Path(relative)
        media_name = transcript_media_name(local_transcript) or Path(relative).stem
        relative_path = Path(relative)
        if not relative_path.parts or relative_path.parts[0].casefold() != corpus.casefold():
            tqdm.write(
                f"  WARNING: local path does not begin with corpus name; "
                f"using corpus root for {corpus}"
            )
            corpus_directory = f"{media_root.rstrip('/')}/{collection}/{corpus}"
            preferred_directory = corpus_directory
        else:
            corpus_directory = f"{media_root.rstrip('/')}/{collection}/{relative_path.parts[0]}"
            subdirectory = Path(*relative_path.parts[1:-1])
            preferred_directory = urljoin(
                corpus_directory.rstrip("/") + "/",
                subdirectory.as_posix() + ("/" if subdirectory.parts else ""),
            ).rstrip("/")
        tqdm.write(
            f"  Traversing TalkBank corpus: {corpus_directory}"
            f" (preferred branch: {preferred_directory})"
        )
        media_url, pages_scanned, search_detail = resolve_media_url(
            session,
            corpus_directory,
            media_name,
            preferred_directory=preferred_directory,
            max_directory_pages=max_directory_pages,
        )
        if media_url is None:
            tqdm.write(
                f"  MISSING: no accessible media file found after scanning "
                f"{pages_scanned} directory page(s): {search_detail}"
            )
            for row in rows:
                report.append(
                    {
                        **row,
                        "status": "missing_media",
                        "reason": (
                            f"Media not found or inaccessible under {corpus_directory}; "
                            f"preferred branch: {preferred_directory}; "
                            f"scanned {pages_scanned} directory page(s); {search_detail}"
                        ),
                    }
                )
            continue

        if dry_run:
            tqdm.write(f"  DRY RUN: would download {media_url}")
            for row, target in targets:
                report.append({**row, "status": "planned", "output": str(target), "source_url": media_url})
            continue

        try:
            with tempfile.TemporaryDirectory(dir=tmp_dir) as temporary_directory:
                source = Path(temporary_directory) / (Path(urlparse(media_url).path).name or "source_media")
                tqdm.write(f"  Downloading source media: {media_url}")
                bytes_downloaded = download_source(session, media_url, source)
                tqdm.write(
                    f"  Downloaded {bytes_downloaded / (1024 * 1024):.2f} MiB "
                    f"to temporary storage"
                )
                clip_items = tqdm(
                    targets,
                    total=len(targets),
                    desc=f"  Clips {corpus}/{Path(filepath).name}",
                    unit="clip",
                    leave=False,
                )
                for row, target in clip_items:
                    if target.exists() and target.stat().st_size > 0:
                        tqdm.write(f"  EXISTS: {target}")
                        report.append({**row, "status": "already_exists", "output": str(target), "source_url": media_url})
                        continue
                    start = float(row["start_seconds"])
                    end = float(row["end_seconds"])
                    tqdm.write(
                        f"  Encoding clip {target.name} "
                        f"[{start:.3f}s - {end:.3f}s]"
                    )
                    run_ffmpeg(
                        source,
                        target,
                        start,
                        end,
                        ffmpeg,
                        ffmpeg_timeout,
                    )
                    tqdm.write(
                        f"  CREATED: {target} "
                        f"[{start:.3f}s - {end:.3f}s]"
                    )
                    report.append({**row, "status": "created", "output": str(target), "source_url": media_url})
                tqdm.write("  Deleted temporary source media")
        except Exception as error:
            tqdm.write(f"  ERROR: {type(error).__name__}: {error}")
            for row, target in targets:
                report.append({**row, "status": "error", "output": str(target), "source_url": media_url, "reason": f"{type(error).__name__}: {error}"})

    report_path = output_dir / "media_download_report.csv"
    if report:
        report = finalize_rows(report, output_dir)
        fields = REPORT_COLUMNS
        temporary_report = report_path.with_name(report_path.name + ".tmp")
        with temporary_report.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row.get(key, "") for key in REPORT_COLUMNS} for row in report)
        temporary_report.replace(report_path)
    counts = pd.Series([row["status"] for row in report]).value_counts().to_dict() if report else {}
    print(f"Manifest segments: {len(manifest_rows):,}")
    print(f"Transcript media groups: {len(grouped):,}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Report: {report_path.resolve()}")
    print(f"Status counts: {counts}")
    return 0 if not any(row["status"] in {"error", "missing_media", "skipped"} for row in report) else 2


def main() -> int:
    """Run the downloader from command-line options."""
    args = build_parser().parse_args()
    return download_media(
        args.manifest,
        args.selection_workbook,
        args.transcripts_dir,
        args.output_dir,
        cookies_file=args.cookies_file,
        login_url=args.login_url,
        chrome_user_data=args.chrome_user_data,
        master_file=args.master_file,
        collection_map=args.collection_map,
        media_root=args.media_root,
        max_directory_pages=args.max_directory_pages,
        ffmpeg=args.ffmpeg,
        ffmpeg_timeout=args.ffmpeg_timeout,
        timeout=args.timeout,
        tmp_dir=args.tmp_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
