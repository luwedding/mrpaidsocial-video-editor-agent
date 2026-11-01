"""Shared helpers for the long-form tools: binaries, JSON, transcripts, time maps.

Everything here is stdlib. numpy is only imported by analyze.py, which needs it for
signal work on hour-long files.
"""
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CUT_TIMELINE_DIR = ROOT / ".claude/skills/reel-recut/scripts"


def binary(name):
    """FFMPEG / FFPROBE env, then *_PATH, then PATH, then Homebrew."""
    env = name.upper()
    found = os.environ.get(env) or os.environ.get(env + "_PATH") or shutil.which(name.lower())
    if found:
        return found
    brew = Path("/opt/homebrew/bin") / name.lower()
    return str(brew) if brew.exists() else name.lower()


def run(argv, **kwargs):
    result = subprocess.run([str(v) for v in argv], text=True, capture_output=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f"{Path(str(argv[0])).name} failed ({result.returncode}): {result.stderr[-3000:]}")
    return result.stdout


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def probe(path):
    data = json.loads(run([binary("ffprobe"), "-v", "error", "-show_format", "-show_streams", "-of", "json", path]))
    video = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    audio = [s for s in data["streams"] if s.get("codec_type") == "audio"]
    fps = None
    if video:
        num, _, den = (video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1").partition("/")
        if float(den or 1) and float(num):
            fps = float(num) / float(den or 1)
    rotation = 0
    if video:
        for side in video.get("side_data_list", []) or []:
            if "rotation" in side:
                rotation = int(side["rotation"])
        rotation = int((video.get("tags") or {}).get("rotate", rotation))
    width, height = (video["width"], video["height"]) if video else (0, 0)
    if abs(rotation) % 180 == 90:
        width, height = height, width
    transfer = (video or {}).get("color_transfer", "")
    return {
        "path": str(Path(path).resolve()),
        "duration": float(data["format"].get("duration") or 0),
        "width": width, "height": height, "fps": fps,
        "fpsRational": (video or {}).get("avg_frame_rate"),
        "hdr": transfer in ("arib-std-b67", "smpte2084"),
        "videoCodec": (video or {}).get("codec_name"),
        "audioStreams": [{"index": s["index"], "channels": int(s.get("channels") or 0),
                          "sampleRate": int(s.get("sample_rate") or 0),
                          "layout": s.get("channel_layout")} for s in audio],
    }


# ── transcripts ─────────────────────────────────────────────────────────────────────

def _num(value, what):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite {what}")
    return value


def _whisper_cpp_tokens(segments):
    """whisper-cli -ojf: segments[].tokens[] carry ms offsets; a leading space opens a word."""
    words = []
    for seg in segments:
        for tok in seg.get("tokens", []):
            text = tok.get("text", "")
            if not text or text.startswith("[_") or text.startswith("<|"):
                continue
            off = tok.get("offsets") or {}
            start, end = off.get("from", 0) / 1000, off.get("to", 0) / 1000
            if words and not text.startswith(" "):
                words[-1]["text"] += text
                words[-1]["end"] = max(words[-1]["end"], end)
            else:
                words.append({"text": text.strip(), "start": start, "end": end})
    return words


def load_words(path):
    """Normalize the common word-timestamp shapes into [{text,start,end,speaker?}].

    Accepts: a bare list; {"words": [...]} (OpenAI verbose_json, Deepgram-like);
    {"segments": [{"words": [...], "speaker"?}]} (faster-whisper / whisperX, which also
    carries per-word "speaker" after diarization); whisper-cli -ojf {"transcription": [...]}.
    """
    data = read_json(path)
    rows = None
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        if isinstance(data.get("words"), list):
            rows = data["words"]
        elif isinstance(data.get("segments"), list) and any("words" in s for s in data["segments"]):
            rows = []
            for seg in data["segments"]:
                for w in seg.get("words", []):
                    rows.append(dict(w, speaker=w.get("speaker", seg.get("speaker"))))
        elif isinstance(data.get("transcription"), list):
            rows = _whisper_cpp_tokens(data["transcription"])
    if rows is None:
        raise ValueError(f"{path}: no word-level timestamps found (need words[], segments[].words[] or whisper-cli -ojf)")
    words = []
    for row in rows:
        text = str(row.get("text", row.get("word", row.get("punctuated_word", "")))).strip()
        if not text or text.startswith("[") and text.endswith("]"):
            continue
        if "start" in row:
            start, end = _num(row["start"], "start"), _num(row["end"], "end")
        elif "offsets" in row:
            start, end = row["offsets"]["from"] / 1000, row["offsets"]["to"] / 1000
        else:
            continue
        if end < start:
            start, end = end, start
        w = {"text": text, "start": round(start, 3), "end": round(end, 3)}
        if row.get("speaker") not in (None, ""):
            w["speaker"] = str(row["speaker"])
        words.append(w)
    words.sort(key=lambda w: w["start"])
    if not words:
        raise ValueError(f"{path}: transcript has no words")
    return words


def norm(text):
    return re.sub(r"[^\w']+", "", text.lower(), flags=re.UNICODE)


# Pure hesitations: safe to remove automatically when isolated. Multi-word / lexical
# fillers ("like", "o sea", "you know", "este") are ambiguous and are only SUGGESTED.
HESITATIONS = {
    "um", "umm", "uh", "uhh", "uhm", "erm", "er", "ah", "ahh", "eh", "ehh", "ehm", "em",
    "mm", "mmm", "hmm", "hm", "mhm", "eeh", "eee", "aaa", "mmmm",
}
SOFT_FILLERS = {
    "en": {"like", "basically", "literally", "actually", "so", "right", "okay", "ok"},
    "es": {"este", "pues", "bueno", "osea", "digamos", "tipo", "vale", "nada"},
}
SOFT_FILLER_PHRASES = [("you", "know"), ("i", "mean"), ("kind", "of"), ("sort", "of"),
                       ("o", "sea"), ("o", "sea", "que"), ("es", "que"), ("como", "que")]

# Words that make a clip start that depends on context the viewer never heard.
DANGLING_STARTERS = {
    "and", "but", "so", "because", "or", "then", "also", "which", "that", "it", "this",
    "y", "pero", "entonces", "porque", "o", "que", "eso", "esto", "también", "además", "pues",
}
HOOK_WORDS = {
    # en
    "secret", "mistake", "mistakes", "never", "always", "nobody", "everyone", "truth", "why", "how",
    "wrong", "best", "worst", "money", "million", "biggest", "stop", "nobody's", "crazy", "insane",
    "hack", "problem", "reason", "actually", "realized", "learned", "first", "only", "most",
    # es
    "secreto", "error", "errores", "nunca", "siempre", "nadie", "todos", "verdad", "por", "qué", "cómo",
    "mejor", "peor", "dinero", "millón", "millones", "deja", "loco", "problema", "razón", "aprendí",
    "primero", "único", "única", "mayoría", "jamás", "dato",
}
SECOND_PERSON = {"you", "your", "you're", "yourself", "tú", "tu", "tus", "te", "ti", "usted", "ustedes", "vos"}


def sentences(words, max_gap=0.7, max_len=20.0):
    """Group words into sentence-ish units: end punctuation, a pause, or a speaker change."""
    out, cur = [], []
    for i, w in enumerate(words):
        if cur:
            prev = words[cur[-1]]
            if (w["start"] - prev["end"] > max_gap or w.get("speaker") != prev.get("speaker")
                    or w["end"] - words[cur[0]]["start"] > max_len):
                out.append(cur)
                cur = []
        cur.append(i)
        if re.search(r"[.!?…]['\")»]*$", w["text"]):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    result = []
    for idx in out:
        spk = [words[i].get("speaker") for i in idx if words[i].get("speaker")]
        result.append({
            "first": idx[0], "last": idx[-1],
            "start": words[idx[0]]["start"], "end": words[idx[-1]]["end"],
            "text": " ".join(words[i]["text"] for i in idx),
            "speaker": max(set(spk), key=spk.count) if spk else None,
            "closed": bool(re.search(r"[.!?…]['\")»]*$", words[idx[-1]]["text"])),
        })
    return result


# ── time maps ───────────────────────────────────────────────────────────────────────

def merge_ranges(ranges, eps=1e-6):
    ranges = sorted([list(r) for r in ranges if r[1] - r[0] > eps])
    merged = []
    for r in ranges:
        if merged and r[0] <= merged[-1][1] + eps:
            merged[-1][1] = max(merged[-1][1], r[1])
        else:
            merged.append(r)
    return merged


def keeps_from_cuts(start, end, cuts, min_keep=0.2):
    keeps, t = [], start
    for a, b in merge_ranges([[max(start, a), min(end, b)] for a, b in cuts if b > start and a < end]):
        if a - t >= min_keep:
            keeps.append([t, a])
        t = max(t, b)
    if end - t >= min_keep:
        keeps.append([t, end])
    return keeps


class TimeMap:
    """Source time ↔ output time through an ordered list of kept [start, end] windows."""

    def __init__(self, keeps):
        self.windows, total = [], 0.0
        for a, b in keeps:
            self.windows.append({"raw_start": a, "raw_end": b, "master_start": total, "master_end": total + (b - a)})
            total += b - a
        self.duration = total

    def to_output(self, t):
        for w in self.windows:
            if w["raw_start"] - 1e-6 <= t <= w["raw_end"] + 1e-6:
                return w["master_start"] + (t - w["raw_start"])
        return None

    def to_output_clamped(self, t):
        best = None
        for w in self.windows:
            if t < w["raw_start"]:
                return w["master_start"] if best is None else best
            if t <= w["raw_end"]:
                return w["master_start"] + (t - w["raw_start"])
            best = w["master_end"]
        return best if best is not None else 0.0

    def window(self, t):
        for i, w in enumerate(self.windows):
            if w["raw_start"] - 1e-6 <= t <= w["raw_end"] + 1e-6:
                return i
        return None

    def words(self, words):
        """Remap words that survive whole; a word a cut runs through is dropped, not squeezed."""
        out = []
        for w in words:
            i = self.window(w["start"])
            if i is None or i != self.window(w["end"]):
                continue
            s, e = self.to_output(w["start"]), self.to_output(w["end"])
            out.append(dict(w, start=round(s, 3), end=round(e, 3)))
        return out


def cut_timeline():
    """Import the reel-recut frame/sample-grid assembler instead of forking it."""
    if str(CUT_TIMELINE_DIR) not in sys.path:
        sys.path.insert(0, str(CUT_TIMELINE_DIR))
    import cut_timeline as module
    return module


def fmt_ts(seconds):
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02}:{s:02}" if h else f"{m}:{s:02}"
