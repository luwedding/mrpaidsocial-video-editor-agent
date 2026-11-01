#!/usr/bin/env python3
"""find_clips.py — rank candidate shorts inside a long recording from audio + picture.

    python3 find_clips.py work/analysis.json --out work/clips [--min 20 --max 60 --count 12]

A candidate is a run of whole sentences (never starts or ends mid-sentence). Each one is
scored on signals measured from the file — then YOU (the agent) read the shortlist and make
the editorial call. The score finds where to look; it does not know what is interesting.

  hook        first sentence: a question, a number, a strong claim word, "you/tú",
              short and punchy; penalised when it opens on a connector ("and", "pero",
              "eso") that needs context the viewer never heard
  payoff      ends on closed punctuation followed by a pause or a speaker change
  energy      loudness of the window vs the episode's speech level, and its variation
  pace        words/sec vs the episode median (a bit faster than usual reads well)
  reaction    laughs / reactions (nonverbal events) inside the window
  exchange    speaker turns: a back-and-forth beats a monologue for podcasts
  visual      motion vs the episode median (gestures, leaning in)
  penalties   hesitations/fillers density, long dead air, too many turns (crosstalk)

Writes clips.json (all features, shortlist ranked, overlaps suppressed) and clips.md
(a table + full text of every shortlisted clip, for reading).
"""
import argparse
import bisect
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lf_common import (DANGLING_STARTERS, HESITATIONS, HOOK_WORDS, SECOND_PERSON, SOFT_FILLERS,  # noqa: E402
                       fmt_ts, norm, read_json, write_json)

WEIGHTS = {"hook": 3.0, "payoff": 1.5, "energy": 1.2, "pace": 0.8, "reaction": 1.2,
           "exchange": 1.0, "visual": 0.5, "penalty": 2.0}
NUMBER_WORDS = {"one", "two", "three", "five", "ten", "hundred", "thousand", "million", "uno", "dos",
                "tres", "cinco", "diez", "cien", "mil", "millón", "percent", "por", "ciento"}


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def series_slice(values, hop, a, b):
    return values[int(a / hop):max(int(a / hop) + 1, int(b / hop))]


def hook_score(sent, words):
    toks = [norm(words[i]["text"]) for i in range(sent["first"], sent["last"] + 1)]
    toks = [t for t in toks if t]
    if not toks:
        return 0.0, ["empty"]
    notes, s = [], 0.3
    text = sent["text"]
    if "?" in text or toks[0] in {"why", "how", "what", "qué", "por", "cómo", "cuál", "cuánto", "quién"}:
        s += 0.25; notes.append("question")
    if re.search(r"\d", text) or any(t in NUMBER_WORDS for t in toks):
        s += 0.15; notes.append("number")
    strong = [t for t in toks if t in HOOK_WORDS]
    if strong:
        s += min(0.3, 0.12 * len(strong)); notes.append("claim:" + ",".join(strong[:3]))
    if any(t in SECOND_PERSON for t in toks):
        s += 0.1; notes.append("you")
    dur = sent["end"] - sent["start"]
    if dur <= 6:
        s += 0.1
    elif dur > 12:
        s -= 0.15; notes.append("long-opener")
    if toks[0] in DANGLING_STARTERS or toks[0] in HESITATIONS:
        s -= 0.45; notes.append(f"dangling-start:{toks[0]}")
    return clamp(s), notes


def score_window(an, i, j, ctx):
    sents, words = an["sentences"], an["words"]
    first, last = sents[i], sents[j]
    a, b = first["start"], last["end"]
    dur = b - a
    feats, notes = {}, []
    feats["hook"], hn = hook_score(first, words)
    notes += hn
    nxt = sents[j + 1] if j + 1 < len(sents) else None
    after = (nxt["start"] - b) if nxt else 9.0
    feats["payoff"] = clamp((0.5 if last["closed"] else 0.0) + clamp(after / 1.0) * 0.3
                            + (0.2 if nxt and nxt["speaker"] != last["speaker"] else 0.0))
    if not last["closed"]:
        notes.append("open-ending")
    au = an["audio"]
    loud = series_slice(au["loudnessDb"], au["seriesHop"], a, b)
    voiced = [v for v in loud if v > au["thresholdDb"]] or [au["thresholdDb"]]
    rel = statistics.fmean(voiced) - ctx["speech_mean"]
    spread = statistics.pstdev(voiced) if len(voiced) > 1 else 0.0
    feats["energy"] = clamp(0.5 + rel / 8 + (spread - ctx["speech_sd"]) / 12)
    lo, hi = bisect.bisect_left(ctx["starts"], a - 1e-6), bisect.bisect_right(ctx["starts"], b)
    wins = [w for w in words[lo:hi] if w["end"] <= b + 1e-6]
    wps = len(wins) / dur if dur else 0
    feats["pace"] = clamp(1 - abs((wps / ctx["wps"]) - 1.12) * 2.5) if ctx["wps"] else 0.5
    reactions = [n for n in au["nonverbal"] if a <= n["start"] and n["end"] <= b + 1.0]
    feats["reaction"] = clamp(len(reactions) / 2)
    if reactions:
        notes.append(f"reactions:{len(reactions)}")
    spk = [s["speaker"] for s in sents[i:j + 1] if s["speaker"]]
    turns = sum(1 for x, y in zip(spk, spk[1:]) if x != y)
    feats["exchange"] = clamp(turns / 3) if turns <= 6 else clamp(1 - (turns - 6) / 6)
    vid = an.get("video")
    if vid and vid["motion"]:
        mot = series_slice(vid["motion"], vid["hop"], a, b)
        feats["visual"] = clamp(0.5 + (statistics.fmean(mot) - ctx["motion_med"]) / (ctx["motion_med"] * 2 + 1e-6)) if mot else 0.5
    else:
        feats["visual"] = 0.5
    hes = sum(1 for w in wins if norm(w["text"]) in HESITATIONS)
    soft = sum(1 for w in wins if norm(w["text"]) in ctx["soft"])
    dead = sum(max(0, y["start"] - x["end"] - 1.2) for x, y in zip(wins, wins[1:]))
    feats["penalty"] = clamp((hes * 1.0 + soft * 0.3) / max(1, len(wins)) * 8 + dead / 4)
    if dead > 1:
        notes.append(f"dead-air:{dead:.1f}s")
    total = sum(WEIGHTS[k] * feats[k] for k in WEIGHTS if k != "penalty") - WEIGHTS["penalty"] * feats["penalty"]
    return {"start": round(a, 2), "end": round(b, 2), "duration": round(dur, 2),
            "firstSentence": i, "lastSentence": j, "score": round(total, 3),
            "features": {k: round(v, 3) for k, v in feats.items()}, "notes": notes,
            "speakers": sorted(set(spk)), "turns": turns,
            "text": " ".join(s["text"] for s in sents[i:j + 1])}


def context(an):
    au = an["audio"]
    voiced = [v for v in au["loudnessDb"] if v > au["thresholdDb"]] or [au["thresholdDb"]]
    words = an["words"]
    talk = sum(w["end"] - w["start"] for w in words) or 1
    span = (words[-1]["end"] - words[0]["start"]) if words else 1
    vid = an.get("video")
    return {"speech_mean": statistics.fmean(voiced),
            "speech_sd": statistics.pstdev(voiced) if len(voiced) > 1 else 0.0,
            "wps": len(words) / span if span else 0,
            "talk_ratio": talk / span if span else 0,
            "motion_med": statistics.median(vid["motion"][1:]) if vid and len(vid["motion"]) > 2 else 1.0,
            "soft": SOFT_FILLERS["en"] | SOFT_FILLERS["es"],
            "starts": [w["start"] for w in words]}


def find(an, min_len=20.0, max_len=60.0, count=12):
    sents = an["sentences"]
    if not sents:
        raise ValueError("analysis has no sentences — run analyze.py with --words")
    ctx = context(an)
    cands = []
    for i in range(len(sents)):
        for j in range(i, len(sents)):
            dur = sents[j]["end"] - sents[i]["start"]
            if dur > max_len:
                break
            if dur >= min_len:
                cands.append(score_window(an, i, j, ctx))
    cands.sort(key=lambda c: -c["score"])
    picked = []
    for c in cands:
        if all(min(c["end"], p["end"]) - max(c["start"], p["start"]) < 0.25 * min(c["duration"], p["duration"]) for p in picked):
            picked.append(c)
        if len(picked) >= count:
            break
    for n, c in enumerate(picked, 1):
        c["id"] = f"clip{n:02}"
    return {"params": {"min": min_len, "max": max_len, "count": count, "weights": WEIGHTS},
            "candidatesScored": len(cands), "clips": picked}


def write_md(result, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Short candidates — signal-ranked, NOT editorially judged\n\n"
                "Read every one. Keep clips that stand alone (a viewer with zero context gets it), "
                "open on a hook within 2 s and land a payoff. Trim to sentence edges, then write "
                "selected.json.\n\n| id | in | out | len | score | hook | notes |\n|---|---|---|---|---|---|---|\n")
        for c in result["clips"]:
            f.write(f"| {c['id']} | {fmt_ts(c['start'])} ({c['start']}) | {fmt_ts(c['end'])} ({c['end']}) | "
                    f"{c['duration']:.0f}s | {c['score']:.2f} | {c['features']['hook']:.2f} | {', '.join(c['notes'])} |\n")
        for c in result["clips"]:
            f.write(f"\n## {c['id']} · {fmt_ts(c['start'])}–{fmt_ts(c['end'])} · {c['duration']:.0f}s · "
                    f"{', '.join(c['speakers']) or 'single speaker'}\n\n{c['text']}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("analysis")
    p.add_argument("--out", required=True, help="directory for clips.json + clips.md")
    p.add_argument("--min", type=float, default=20.0)
    p.add_argument("--max", type=float, default=60.0)
    p.add_argument("--count", type=int, default=12)
    a = p.parse_args()
    if not (0 < a.min < a.max) or a.count < 1:
        p.exit(2, "find_clips: need 0 < --min < --max and --count >= 1\n")
    try:
        result = find(read_json(a.analysis), a.min, a.max, a.count)
    except (ValueError, OSError, KeyError) as exc:
        p.exit(2, f"find_clips: {exc}\n")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "clips.json", result)
    write_md(result, out / "clips.md")
    print(f"{result['candidatesScored']} windows scored → {len(result['clips'])} shortlisted → {out}/clips.md")
    for c in result["clips"]:
        print(f"  {c['id']} {fmt_ts(c['start'])}-{fmt_ts(c['end'])} {c['duration']:.0f}s score {c['score']:.2f}  {c['text'][:70]}…")


if __name__ == "__main__":
    main()
