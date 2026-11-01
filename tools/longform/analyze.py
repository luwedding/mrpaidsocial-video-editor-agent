#!/usr/bin/env python3
"""analyze.py — one pass over a long recording, audio AND picture, into analysis.json.

    python3 analyze.py EPISODE.mp4 --words words.json --out work/analysis.json \
        [--speaker-channels "Ana=0,Luis=1"] [--speaker-boxes "Ana=0,0,0.5,1;Luis=0.5,0,1,1"]

What it measures (all from the file, nothing is guessed):

  audio   per-hop loudness (25 ms hops; 0.1 s series stored), an ADAPTIVE speech threshold
          (noise floor percentile + margin — a fixed -30 dB gate fails on noisy rooms),
          silences, and NONVERBAL events: loud spans with no transcribed words (laughs,
          reactions, applause, music stings). Those are protected from cuts and score clips.
  picture 4 fps 160x90 luma: motion per sample, scene cuts, black spans, and — when
          --speaker-boxes is given — motion inside each speaker's region of a wide shot.
  speaker who is talking when, from the best source available:
          1. per-word "speaker" labels in the transcript (whisperX / AssemblyAI / Deepgram)
          2. one mic per person: --speaker-channels (or auto when the 2 channels differ)
          3. picture: mouth/body motion inside --speaker-boxes while speech is present
  words   normalized word list (+ speaker), and sentence units for the planners.

Needs ffmpeg/ffprobe and numpy. Never calls a network service.
"""
import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lf_common import binary, load_words, probe, sentences, write_json  # noqa: E402

AUDIO_RATE = 8000
HOP = 0.025
SERIES_HOP = 0.1
VIDEO_FPS = 4
VW, VH = 160, 90


def parse_named(spec, kind):
    out = {}
    if not spec:
        return out
    for part in spec.replace(";", "|").split("|") if kind == "box" else spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, value = part.partition("=")
        if not value:
            raise ValueError(f"bad --speaker-{kind}s entry {part!r} (want Name=...)")
        if kind == "box":
            nums = [float(v) for v in value.split(",")]
            if len(nums) != 4 or not (0 <= nums[0] < nums[2] <= 1 and 0 <= nums[1] < nums[3] <= 1):
                raise ValueError(f"box {part!r} must be x0,y0,x1,y1 fractions of the frame")
            out[name.strip()] = nums
        else:
            out[name.strip()] = int(value)
    return out


def decode_audio(path, info):
    """Every audio stream merged into one multichannel f32 stream at 8 kHz."""
    streams = info["audioStreams"]
    if not streams:
        raise ValueError("source has no audio stream")
    cmd = [binary("ffmpeg"), "-nostdin", "-v", "error", "-i", path]
    if len(streams) > 1:
        labels = "".join(f"[0:a:{i}]" for i in range(len(streams)))
        cmd += ["-filter_complex", f"{labels}amerge=inputs={len(streams)}[a]", "-map", "[a]"]
    else:
        cmd += ["-map", "0:a:0"]
    channels = sum(max(1, s["channels"]) for s in streams)
    cmd += ["-ar", str(AUDIO_RATE), "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    hop = int(AUDIO_RATE * HOP)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = hop * channels * 4
    chunk = frame_bytes * 400
    rows, leftover = [], b""
    while True:
        data = proc.stdout.read(chunk)
        if not data:
            break
        data = leftover + data
        usable = len(data) // frame_bytes * frame_bytes
        leftover = data[usable:]
        if usable:
            block = np.frombuffer(data[:usable], dtype="<f4").reshape(-1, hop, channels)
            rows.append(np.sqrt(np.mean(block.astype(np.float64) ** 2, axis=1)))  # (frames, ch)
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait():
        raise RuntimeError(f"ffmpeg audio decode failed: {err[-2000:]}")
    rms = np.concatenate(rows) if rows else np.zeros((0, channels))
    return rms, channels


def to_db(x):
    return 20 * np.log10(np.maximum(x, 1e-6))


def runs(mask, hop, min_len):
    """[start, end] seconds of True runs at least min_len long."""
    out, i, n = [], 0, len(mask)
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    for a, b in zip(edges[::2], edges[1::2]):
        if (b - a) * hop >= min_len:
            out.append([round(a * hop, 3), round(b * hop, 3)])
    return out


def resample_series(values, src_hop, dst_hop, how="mean"):
    factor = int(round(dst_hop / src_hop))
    n = len(values) // factor * factor
    if n == 0:
        return np.zeros(0)
    block = values[:n].reshape(-1, factor)
    return block.max(axis=1) if how == "max" else block.mean(axis=1)


def audio_features(rms, channels, words, channel_names):
    mix = to_db(np.sqrt(np.mean(rms ** 2, axis=1)))
    floor = float(np.percentile(mix, 10))
    speech = float(np.percentile(mix, 90))
    threshold = floor + max(6.0, 0.35 * (speech - floor))
    voiced = mix > threshold
    silences = runs(~voiced, HOP, 0.3)

    # Loud spans with no transcribed word near them: laughs, reactions, applause, stings.
    nonverbal = []
    if words:
        covered = np.zeros(len(mix), dtype=bool)
        for w in words:
            a = max(0, int((w["start"] - 0.12) / HOP))
            b = min(len(mix), int(math.ceil((w["end"] + 0.12) / HOP)))
            covered[a:b] = True
        for a, b in runs(voiced & ~covered, HOP, 0.4):
            seg = mix[int(a / HOP):int(b / HOP)]
            nonverbal.append({"start": a, "end": b, "peakDb": round(float(seg.max()), 1),
                              "relDb": round(float(seg.mean() - speech), 1)})

    # One mic per speaker: dominant channel by >= 6 dB while that channel is voiced.
    channel_speaker = None
    ch_db = to_db(rms)
    if channels >= 2:
        if not channel_names and channels == 2:
            corr = np.corrcoef(ch_db[:, 0], ch_db[:, 1])[0, 1] if len(ch_db) > 10 else 1.0
            gap = float(np.median(np.abs(ch_db[:, 0] - ch_db[:, 1])))
            if corr < 0.9 or gap > 4.0:
                channel_names = {"S1": 0, "S2": 1}
        if channel_names:
            idx = list(channel_names.values())
            if max(idx) >= channels:
                raise ValueError(f"--speaker-channels names channel {max(idx)} but the audio has {channels}")
            sub = ch_db[:, idx]
            order = np.sort(sub, axis=1)
            top = np.argmax(sub, axis=1)
            dominant = (order[:, -1] - order[:, -2] >= 6.0) & voiced
            names = list(channel_names.keys())
            channel_speaker = [names[t] if d else None for t, d in zip(top, dominant)]

    return {
        "floorDb": round(floor, 1), "speechDb": round(speech, 1), "thresholdDb": round(threshold, 1),
        "hop": HOP, "voiced": voiced, "mix": mix,
        "silences": silences, "nonverbal": nonverbal, "channelSpeaker": channel_speaker,
        "channels": channels,
    }


def video_features(path, info, boxes):
    if not info["width"]:
        return None
    cmd = [binary("ffmpeg"), "-nostdin", "-v", "error", "-i", path, "-map", "0:v:0", "-an",
           "-vf", f"fps={VIDEO_FPS},scale={VW}:{VH}:flags=area,format=gray", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    size = VW * VH
    regions = {n: (int(b[1] * VH), int(math.ceil(b[3] * VH)), int(b[0] * VW), int(math.ceil(b[2] * VW)))
               for n, b in boxes.items()}
    luma, motion, box_motion, prev = [], [], {n: [] for n in boxes}, None
    while True:
        buf = proc.stdout.read(size * 64)
        if not buf:
            break
        frames = np.frombuffer(buf[:len(buf) // size * size], dtype=np.uint8).reshape(-1, VH, VW).astype(np.int16)
        for f in frames:
            luma.append(float(f.mean()))
            if prev is None:
                motion.append(0.0)
                for n in boxes:
                    box_motion[n].append(0.0)
            else:
                d = np.abs(f - prev)
                motion.append(float(d.mean()))
                for n, (y0, y1, x0, x1) in regions.items():
                    box_motion[n].append(float(d[y0:y1, x0:x1].mean()))
            prev = f
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait():
        raise RuntimeError(f"ffmpeg video decode failed: {err[-2000:]}")
    motion_arr = np.array(motion)
    hop = 1 / VIDEO_FPS
    scene = []
    if len(motion_arr) > 2:
        base = max(6.0, 4 * float(np.median(motion_arr[1:])))
        for i in range(1, len(motion_arr)):
            left = motion_arr[i - 1] if i > 1 else 0
            right = motion_arr[i + 1] if i + 1 < len(motion_arr) else 0
            if motion_arr[i] > base and motion_arr[i] >= left and motion_arr[i] >= right:
                if not scene or i * hop - scene[-1] > 1.0:
                    scene.append(round(i * hop, 2))
    black = runs(np.array(luma) < 16, hop, 0.5)
    return {"hop": hop, "luma": luma, "motion": motion, "boxMotion": box_motion,
            "sceneCuts": scene, "blackRanges": black}


def majority_smooth(labels, window):
    out = list(labels)
    half = window // 2
    for i in range(len(labels)):
        seg = [l for l in labels[max(0, i - half): i + half + 1] if l]
        out[i] = max(set(seg), key=seg.count) if seg else None
    return out


def speaker_timeline(words, audio, video, boxes, duration):
    """Labels on a 0.1 s grid, then merged segments; best available source wins."""
    n = int(math.ceil(duration / SERIES_HOP))
    grid, source = [None] * n, None
    if any("speaker" in w for w in words):
        source = "transcript"
        for w in words:
            if w.get("speaker"):
                for i in range(int(w["start"] / SERIES_HOP), min(n, int(math.ceil(w["end"] / SERIES_HOP)))):
                    grid[i] = w["speaker"]
    elif audio["channelSpeaker"]:
        source = "channels"
        step = int(round(SERIES_HOP / HOP))
        cs = audio["channelSpeaker"]
        for i in range(n):
            seg = [l for l in cs[i * step:(i + 1) * step] if l]
            grid[i] = max(set(seg), key=seg.count) if seg else None
        grid = majority_smooth(grid, 5)
    elif boxes and video:
        source = "picture"
        names = list(boxes)
        vhop = video["hop"]
        voiced = audio["voiced"]
        for i in range(n):
            t = i * SERIES_HOP
            a = int(t / HOP)
            if not voiced[a:a + int(SERIES_HOP / HOP)].any():
                continue
            k = min(len(video["motion"]) - 1, int(t / vhop))
            lo, hi = max(0, k - 2), k + 3
            scores = [float(np.mean(video["boxMotion"][nm][lo:hi])) for nm in names]
            order = sorted(range(len(names)), key=lambda j: scores[j], reverse=True)
            if len(names) == 1 or scores[order[0]] > 1.25 * scores[order[1]] + 0.05:
                grid[i] = names[order[0]]
        grid = majority_smooth(grid, 15)
    segments = []
    for i, label in enumerate(grid):
        if not label:
            continue
        t = round(i * SERIES_HOP, 2)
        if segments and segments[-1]["speaker"] == label and t - segments[-1]["end"] <= 0.6:
            segments[-1]["end"] = round(t + SERIES_HOP, 2)
        else:
            segments.append({"start": t, "end": round(t + SERIES_HOP, 2), "speaker": label})
    segments = [s for s in segments if s["end"] - s["start"] >= 0.4]
    return source, segments


def label_words(words, segments):
    if not segments or any("speaker" in w for w in words):
        return words
    j = 0
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        while j + 1 < len(segments) and segments[j]["end"] < mid:
            j += 1
        s = segments[j]
        if s["start"] - 0.3 <= mid <= s["end"] + 0.3:
            w["speaker"] = s["speaker"]
    return words


def analyze(path, words_path=None, channel_spec=None, box_spec=None):
    info = probe(path)
    words = load_words(words_path) if words_path else []
    channel_names = parse_named(channel_spec, "channel")
    boxes = parse_named(box_spec, "box")
    rms, channels = decode_audio(path, info)
    audio = audio_features(rms, channels, words, channel_names)
    video = video_features(path, info, boxes)
    src, timeline = speaker_timeline(words, audio, video, boxes, info["duration"])
    words = label_words(words, timeline)
    mix_series = resample_series(audio["mix"], HOP, SERIES_HOP)
    result = {
        "version": 1,
        "source": info,
        "audio": {k: audio[k] for k in ("floorDb", "speechDb", "thresholdDb", "silences", "nonverbal", "channels")}
        | {"seriesHop": SERIES_HOP, "loudnessDb": [round(float(v), 1) for v in mix_series]},
        "video": None if video is None else {
            "hop": video["hop"], "sceneCuts": video["sceneCuts"], "blackRanges": video["blackRanges"],
            "motion": [round(v, 2) for v in video["motion"]],
            "boxes": boxes,
        },
        "speakers": {"source": src, "names": sorted({s["speaker"] for s in timeline}), "timeline": timeline},
        "words": words,
        "sentences": sentences(words) if words else [],
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--words", help="word-timestamp JSON (whisper-cli -ojf, OpenAI verbose_json, whisperX, list)")
    p.add_argument("--out", required=True)
    p.add_argument("--speaker-channels", help='one mic per person, e.g. "Ana=0,Luis=1" (global channel index across all audio streams)')
    p.add_argument("--speaker-boxes", help='wide shot regions as frame fractions, e.g. "Ana=0,0,0.5,1;Luis=0.5,0,1,1"')
    a = p.parse_args()
    try:
        result = analyze(a.video, a.words, a.speaker_channels, a.speaker_boxes)
    except (ValueError, RuntimeError, OSError) as exc:
        p.exit(2, f"analyze: {exc}\n")
    write_json(a.out, result)
    s, au = result["source"], result["audio"]
    print(f"analysis → {a.out}")
    print(f"  {s['duration']:.1f}s {s['width']}x{s['height']} @ {s['fps'] and round(s['fps'], 3)} fps, "
          f"{au['channels']} audio ch{' (HDR source)' if s['hdr'] else ''}")
    print(f"  floor {au['floorDb']} dB · speech {au['speechDb']} dB · gate {au['thresholdDb']} dB · "
          f"{len(au['silences'])} silences · {len(au['nonverbal'])} nonverbal events")
    if result["video"]:
        print(f"  {len(result['video']['sceneCuts'])} scene cuts · {len(result['video']['blackRanges'])} black spans")
    sp = result["speakers"]
    print(f"  speakers: {sp['names'] or 'none'} (source: {sp['source'] or 'n/a'}, {len(sp['timeline'])} turns)")
    print(f"  {len(result['words'])} words · {len(result['sentences'])} sentences")


if __name__ == "__main__":
    main()
