"""Read target-child metadata and usable, timestamped CHAT utterances."""

import hashlib
import re

try:
    from .ground_truth import clean_ground_truth
except ImportError:
    from ground_truth import clean_ground_truth

TIMESTAMP = re.compile(r"\x15(\d+)_(\d+)\x15")
MIN_CLIP_SECONDS = 0.5
MAX_REFERENCE_WORDS_PER_SECOND = 4.0
LONG_SINGLE_WORD_SECONDS = 5.0


def alignment_reason(start, end, reference):
    """Reject intervals whose text cannot plausibly fit their timestamp."""
    duration = end - start
    words = len(reference.split())
    if duration <= 0:
        return "invalid_interval"
    if duration < MIN_CLIP_SECONDS:
        return "point_bullet_timestamp" if duration <= 0.1 + 1e-9 else "clip_too_short"
    if words / duration > MAX_REFERENCE_WORDS_PER_SECOND:
        return "impossible_speech_rate"
    if duration > LONG_SINGLE_WORD_SECONDS and words <= 1:
        return "long_clip_short_reference"
    return ""


def age_months(value):
    match = re.fullmatch(r"\s*(\d+);(\d+)(?:\.(\d*))?\s*", value)
    if not match:
        return None
    years, months, days = match.groups()
    return int(years) * 12 + int(months) + int(days or 0) / 30.4368


def read_header(text):
    header = re.split(r"^\*", text, maxsplit=1, flags=re.M)[0]
    participants = []
    for value in re.findall(r"^@ID:\s*(.*)$", header, re.M):
        fields = value.strip().split("|")
        if len(fields) > 7:
            participants.append(dict(code=fields[2], age_months=age_months(fields[3]),
                                     sex=fields[4].lower(), role=fields[7].lower()))
    targets = [p for p in participants if p["code"] == "CHI" or p["role"] == "target_child"]
    if len(targets) != 1:
        raise ValueError("No single, explicit target-child speaker")
    target = targets[0]
    births = re.findall(rf"^@Birth of {re.escape(target['code'])}:\s*(.*)$", header, re.M)
    # Some corpora store the birthdate in a comment rather than a CHAT birth field.
    births += re.findall(rf"^@Comment:\s*Birth of {re.escape(target['code'])}:\s*(.*)$", header, re.M)
    target["birthdate"] = births[0].strip().upper() if births else ""
    date = re.search(r"^@Date:\s*(.*)$", header, re.M)
    target["recording_date"] = date.group(1).strip() if date else ""
    return target


def utterance_id(corpus, filepath, start, end, ordinal):
    source_hash = hashlib.blake2s(f"{corpus}|{filepath}".encode(), digest_size=4).hexdigest()
    return f"{source_hash}_{round(start * 1000)}_{round(end * 1000)}_{ordinal}"


def read_utterances(text, corpus, filepath, speaker):
    """Use the same eligible utterance pool for capping, MLU and audio export."""
    rows, skipped = [], []
    ordinal = 0
    # Keep original line numbers when folding CHAT continuation lines.
    turns = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if line.startswith(("\t", " ")) and turns:
            turns[-1][1] += " " + line.strip()
        else:
            turns.append([line_number, line])
    speaker_turns = []
    for index, (_, line) in enumerate(turns):
        match = re.match(r"^\*([^:]+):\s*(.*)$", line)
        if match and match.group(1) != speaker:
            times = TIMESTAMP.findall(match.group(2))
            if len(times) == 1:
                adult_start, adult_end = (int(value) / 1000 for value in times[0])
                if adult_end > adult_start:
                    speaker_turns.append((index, adult_start, adult_end))
    for index, (line_number, line) in enumerate(turns):
        match = re.match(rf"^\*{re.escape(speaker)}:\s*(.*)$", line)
        if not match:
            continue
        times = TIMESTAMP.findall(match.group(1))
        reason = "no_timestamp" if not times else "multiple_timestamps" if len(times) != 1 else ""
        if times:
            ordinal += 1
        raw = TIMESTAMP.sub("", match.group(1)).strip()
        cleaned = clean_ground_truth(raw)
        start, end = (int(v) / 1000 for v in times[0]) if times else (0, 0)
        if not reason and (end <= start or start < 0):
            reason = "invalid_interval"
        if not reason and not cleaned:
            reason = "empty_cleaned_reference"
        if not reason:
            reason = alignment_reason(start, end, cleaned)
        if not reason:
            adjacent = [turn for turn in speaker_turns if abs(turn[0] - index) == 1]
            if any(max(start, adult_start) < min(end, adult_end)
                   for _, adult_start, adult_end in adjacent):
                reason = "speaker_timestamp_overlap"
        if reason:
            skipped.append(dict(corpus=corpus, filepath=filepath, line=line_number, reason=reason))
            continue
        rows.append(dict(corpus=corpus, filepath=filepath, speaker_code=speaker,
                         start_seconds=start, end_seconds=end, raw_utterance=raw,
                         utterance=cleaned, utterance_text=cleaned,
                         utterance_id=utterance_id(corpus, filepath, start, end, ordinal),
                         clip_ordinal=ordinal, num_tokens=len(cleaned.split()),
                         mlu_units=len(cleaned.split()), mlu_basis="words"))
    return rows, skipped
