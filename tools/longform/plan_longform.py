#!/usr/bin/env python3
"""plan_longform.py — turn analysis.json into a cut plan (EDL) for a long-form edit.

    python3 plan_longform.py work/analysis.json --out work/plan-v1 \
        [--profile podcast|tight|short] [--apply stutter,retake] [--max-pause 0.8]
        [--range 120.5,1840] [--force-cut 300.2,305.0] [--force-keep 610,640]

Cut classes. AUTO ones are applied; SUGGESTED ones are listed in report.md and only
applied with --apply (after you have read them — a podcast's "no, no, no" is a joke,
not a stutter):

  AUTO       pause      a word gap longer than --max-pause whose energy is genuinely
                        silent; edges are walked out from the transcript times along the
                        actual waveform, then exactly --target-gap of air is kept
             hesitation "um / uh / eh / ehm / mmm" with air around it
             head/tail  dead air before the first and after the last word
  SUGGESTED  stutter    the same word twice in a row ("the the", "que que")
             retake     a 3+ word phrase restarted within ~12 s by the same speaker:
                        cut from the abandoned attempt to the restart
             filler     lexical fillers ("like", "you know", "o sea", "este") — usually
                        left in for conversation; listed so you can decide

Never cut: nonverbal events (laughs, reactions) and anything inside --force-keep.

Writes: keeps.json, edl.json, cuts.json, words-output.json, qa-manifest.json,
transcript.md (output-time, one line per sentence), chapters.draft.md, report.md
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lf_common import (HESITATIONS, SOFT_FILLERS, SOFT_FILLER_PHRASES, TimeMap, cut_timeline, fmt_ts,  # noqa: E402
                       keeps_from_cuts, merge_ranges, norm, read_json, sentences, write_json)

PROFILES = {
    # Conversation needs air; a podcast cut too tight sounds like an argument.
    "podcast": {"max_pause": 0.9, "target_gap": 0.35, "head_pad": 0.5, "tail_pad": 0.8},
    "tight":   {"max_pause": 0.55, "target_gap": 0.22, "head_pad": 0.3, "tail_pad": 0.6},
    "short":   {"max_pause": 0.3, "target_gap": 0.12, "head_pad": 0.05, "tail_pad": 0.35},
}
LEGIT_DOUBLES = {"no", "si", "sí", "very", "so", "muy", "ya", "bye", "ha", "ja", "go", "yeah", "okay", "ok", "really", "chao"}


class Energy:
    def __init__(self, analysis):
        au = analysis["audio"]
        self.hop = au["seriesHop"]
        self.db = au["loudnessDb"]
        self.thr = au["thresholdDb"]

    def voiced(self, t):
        i = int(t / self.hop)
        return 0 <= i < len(self.db) and self.db[i] > self.thr

    def silent_fraction(self, a, b, exclude=()):
        n = s = 0
        t = a
        while t < b:
            if not any(x0 <= t <= x1 for x0, x1 in exclude):
                n += 1
                s += not self.voiced(t)
            t += self.hop
        return 1.0 if n == 0 else s / n

    def walk(self, t, direction, limit, exclude=()):
        """Walk from a transcript edge along voiced audio (a whisper end is often early)."""
        step = self.hop * direction
        while (t + step - limit) * direction <= 0 and self.voiced(t + step / 2) and not any(
                x0 <= t + step <= x1 for x0, x1 in exclude):
            t += step
        return t


def plan(analysis, start=None, end=None, profile="podcast", overrides=None, apply=(),
         force_cut=(), force_keep=(), fps="source"):
    prm = dict(PROFILES[profile], **{k: v for k, v in (overrides or {}).items() if v is not None})
    dur = analysis["source"]["duration"]
    start = 0.0 if start is None else max(0.0, start)
    end = dur if end is None else min(dur, end)
    words = [w for w in analysis["words"] if w["start"] >= start - 0.01 and w["end"] <= end + 0.01]
    if not words:
        raise ValueError("no transcribed words inside the requested range")
    energy = Energy(analysis)
    protect = [[n["start"] - 0.15, n["end"] + 0.15] for n in analysis["audio"]["nonverbal"]]
    protect += [list(r) for r in force_keep]
    cuts = []

    def add(a, b, reason, applied, detail=""):
        if b - a >= 0.06:
            cuts.append({"start": round(a, 3), "end": round(b, 3), "reason": reason,
                         "applied": applied, "detail": detail})

    hes = [w for w in words if norm(w["text"]) in HESITATIONS]
    hes_ranges = [(w["start"] - 0.02, w["end"] + 0.02) for w in hes]
    real = [w for w in words if norm(w["text"]) not in HESITATIONS]
    if not real:
        raise ValueError("range holds only hesitations")
    half = prm["target_gap"] / 2

    # head / tail
    first_head = energy.walk(real[0]["start"], -1, start)
    add(start, max(start, first_head - prm["head_pad"]), "head", True)
    last_tail = energy.walk(real[-1]["end"], +1, end)
    add(min(end, last_tail + prm["tail_pad"]), end, "tail", True)

    # pauses + hesitations, between consecutive real words AND nonverbal events (a laugh
    # splits a gap in two: the air on each side is judged on its own)
    marks = [{"start": n["start"], "end": n["end"], "text": "<nonverbal>"} for n in analysis["audio"]["nonverbal"]
             if real[0]["start"] < n["start"] and n["end"] < real[-1]["end"]]
    seq = sorted(real + marks, key=lambda w: w["start"])
    for a, b in zip(seq, seq[1:]):
        gap = b["start"] - a["end"]
        inside = [r for r in hes_ranges if r[0] >= a["end"] - 0.05 and r[1] <= b["start"] + 0.05]
        if gap <= 0 or (gap <= prm["max_pause"] and not inside):
            continue
        tail = energy.walk(a["end"], +1, b["start"], inside)
        head = energy.walk(b["start"], -1, a["end"], inside)
        if head - tail <= prm["target_gap"] + 0.05:
            continue
        silent = energy.silent_fraction(tail, head, inside)
        if inside:
            if silent >= 0.6:
                # the cut must swallow the WHOLE hesitation, even if that leaves less air
                cs = max(a["end"] + 0.02, min(tail + half, min(r[0] for r in inside) - 0.01))
                ce = min(b["start"] - 0.02, max(head - half, max(r[1] for r in inside) + 0.01))
                add(cs, ce, "hesitation",
                    True, " ".join(w["text"] for w in hes if a["end"] <= w["start"] <= b["start"]))
        elif silent >= 0.85 and head - tail > prm["max_pause"]:
            add(tail + half, head - half, "pause", True, f"{head - tail:.2f}s of air")

    # stutters
    for a, b in zip(real, real[1:]):
        na, nb = norm(a["text"]), norm(b["text"])
        if na and na == nb and na not in LEGIT_DOUBLES and b["start"] - a["end"] < 0.6:
            add(a["start"] - 0.02, b["start"] - 0.02, "stutter", "stutter" in apply, f"{a['text']} {b['text']}")

    # retakes: a 3-gram restarted by the same speaker, 3–30 words later, within 12 s
    keys = [norm(w["text"]) for w in real]
    seen, resume = set(), 0
    for i in range(len(real) - 2):
        gram = tuple(keys[i:i + 3])
        if i < resume or not all(gram) or gram in seen:
            continue
        for j in range(i + 3, min(len(real) - 2, i + 31)):
            if real[j]["start"] - real[i]["start"] > 12:
                break
            if tuple(keys[j:j + 3]) == gram and real[i].get("speaker") == real[j].get("speaker"):
                abandoned = " ".join(w["text"] for w in real[i:j])
                add(real[i]["start"] - 0.03, real[j]["start"] - 0.03, "retake", "retake" in apply,
                    f"abandoned: “{abandoned}”")
                seen.add(gram)
                resume = j  # the abandoned attempt is one retake, not one per shifted 3-gram
                break

    # lexical fillers (suggest only)
    lex = SOFT_FILLERS["en"] | SOFT_FILLERS["es"]
    for i, w in enumerate(real):
        k = keys[i]
        phrase = next((p for p in SOFT_FILLER_PHRASES if tuple(keys[i:i + len(p)]) == p), None)
        if phrase:
            last = real[i + len(phrase) - 1]
            add(w["start"] - 0.02, last["end"] + 0.02, "filler", "filler" in apply, " ".join(x["text"] for x in real[i:i + len(phrase)]))
        elif k in lex and (i == 0 or w["start"] - real[i - 1]["end"] > 0.15):
            add(w["start"] - 0.02, w["end"] + 0.02, "filler", "filler" in apply, w["text"])

    for a, b in force_cut:
        add(a, b, "manual", True)

    # protection: trim cuts back to the edge of a laugh/reaction or a forced keep
    guarded = []
    for c in cuts:
        hits = [(p0, p1) for p0, p1 in protect if c["start"] < p1 and c["end"] > p0]
        if c["reason"] == "manual" or not hits:
            guarded.append(c)
            continue
        pieces = [[c["start"], c["end"]]]
        for p0, p1 in hits:
            pieces = [q for x, y in pieces for q in ([x, min(y, p0)], [max(x, p1), y]) if q[1] - q[0] >= 0.15]
        if c["reason"] in ("stutter", "retake", "filler", "hesitation") or not pieces:
            c["applied"] = False  # a word-level cut that touches a laugh is no longer the same cut
            c["detail"] = (c["detail"] + " [protected: nonverbal/force-keep]").strip()
            guarded.append(c)
            continue
        for x, y in pieces:
            guarded.append(dict(c, start=round(x, 3), end=round(y, 3),
                                detail=(c["detail"] + " [trimmed at protected range]").strip()))
    cuts = guarded

    cuts.sort(key=lambda c: c["start"])
    for c in cuts:
        c["id"] = f"cut:src{c['start']:.2f}"
    applied = merge_ranges([[c["start"], c["end"]] for c in cuts if c["applied"]])
    keeps = keeps_from_cuts(start, end, applied)
    # Snap to the frame grid the assembler renders on, so every map built from these keeps
    # (output words, manifest, chapters) matches the rendered file exactly.
    rate = fps if fps not in (None, "source") else (analysis["source"].get("fpsRational") or analysis["source"].get("fps") or 30)
    keeps = cut_timeline().plan(keeps, rate)["keeps"]
    return {"profile": profile, "params": prm, "range": [start, end], "cuts": cuts, "keeps": keeps}


def chapters(sents, tmap, min_len=150.0):
    """Candidate chapter starts: after the longest pauses / speaker resets, spaced out."""
    if not sents:
        return []
    scored = []
    for prev, s in zip(sents, sents[1:]):
        out = tmap.to_output(s["start"])
        if out is None:
            continue
        scored.append((s["start"] - prev["end"], out, s))
    scored.sort(key=lambda x: -x[0])
    picks = [(0.0, sents[0])]
    for gap, out, s in scored:
        if all(abs(out - p[0]) >= min_len for p in picks) and out < tmap.duration - 60:
            picks.append((out, s))
    return sorted(picks, key=lambda p: p[0])


def write_outputs(analysis, result, outdir, label="long-form-edit", loudness=-14.0):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    src = analysis["source"]
    tmap = TimeMap(result["keeps"])
    a, b = result["range"]
    words = [w for w in analysis["words"] if a - 0.01 <= w["start"] and w["end"] <= b + 0.01]
    out_words = tmap.words(words)
    write_json(outdir / "keeps.json", result["keeps"])
    write_json(outdir / "cuts.json", result["cuts"])
    write_json(outdir / "words-output.json", out_words)
    write_json(outdir / "edl.json", {"fps": src["fps"], "source": src["path"], "windows": tmap.windows,
                                      "profile": result["profile"], "params": result["params"]})
    events = []
    for c in result["cuts"]:
        if not c["applied"]:
            continue
        events.append({"id": c["id"], "kind": "cut", "dialogueCut": c["reason"] not in ("head", "tail"),
                       "out": {"start": round(tmap.to_output_clamped(c["start"]) or 0, 3)},
                       "src": {"start": c["start"], "end": c["end"]},
                       "meta": {"origin": "silence" if c["reason"] in ("pause", "head", "tail") else c["reason"]}})
    for n in analysis["audio"]["nonverbal"]:
        s0, s1 = tmap.to_output(n["start"]), tmap.to_output(n["end"])
        if s0 is not None and s1 is not None and s1 > s0:
            # laughs / reactions are content: a wordless span here is intentional, not dead air
            events.append({"id": f"nonverbal:src{n['start']:.2f}", "kind": "other",
                           "out": {"start": round(s0, 3), "end": round(s1, 3)},
                           "src": {"start": n["start"], "end": n["end"]}, "meta": {"origin": "nonverbal"}})
    write_json(outdir / "qa-manifest.json", {
        "version": 1, "lane": "generic", "source": src["path"],
        "expectedDuration": round(tmap.duration, 3),
        "expected": {"width": src["width"], "height": src["height"], "fps": round(src["fps"] or 30, 3)},
        "events": events,
        "intentional": {"loudnessTarget": {"lufs": loudness, "tolerance": 1.5}},
        "words": out_words, "sourceWords": words,
    })
    sents = sentences(out_words) if out_words else []
    with open(outdir / "transcript.md", "w", encoding="utf-8") as f:
        f.write("# Output-time transcript\n\n")
        for s in sents:
            f.write(f"[{fmt_ts(s['start'])}] {s['speaker'] + ': ' if s['speaker'] else ''}{s['text']}\n")
    src_sents = sentences(words)
    with open(outdir / "chapters.draft.md", "w", encoding="utf-8") as f:
        f.write("# Chapter candidates (output time) — write a title for each you keep\n\n"
                "YouTube needs the first chapter at 0:00, at least 3 chapters, each ≥ 10 s.\n\n")
        for out, s in chapters(src_sents, tmap):
            f.write(f"{fmt_ts(out)} — TITLE?   (opens with: “{s['text'][:110]}”)\n")
    removed = {}
    for c in result["cuts"]:
        if c["applied"]:
            removed[c["reason"]] = removed.get(c["reason"], 0) + (c["end"] - c["start"])
    with open(outdir / "report.md", "w", encoding="utf-8") as f:
        f.write(f"# Cut plan ({result['profile']})\n\n")
        f.write(f"- source range {fmt_ts(a)}–{fmt_ts(b)} ({b - a:.1f}s) → output {tmap.duration:.1f}s "
                f"({fmt_ts(tmap.duration)}), {len(result['keeps'])} keeps\n")
        for k, v in sorted(removed.items(), key=lambda x: -x[1]):
            f.write(f"- removed {v:.1f}s of {k}\n")
        f.write("\n## Suggested (not applied) — read each, then re-plan with --apply / --force-cut\n\n")
        f.write("| id | src | len | reason | detail |\n|---|---|---|---|---|\n")
        for c in result["cuts"]:
            if not c["applied"]:
                f.write(f"| {c['id']} | {fmt_ts(c['start'])} ({c['start']:.2f}) | {c['end'] - c['start']:.2f}s | "
                        f"{c['reason']} | {c['detail'].replace('|', '/')} |\n")
    return tmap


def parse_pairs(values):
    out = []
    for v in values or []:
        a, b = (float(x) for x in v.split(","))
        if b <= a:
            raise ValueError(f"range {v} ends before it starts")
        out.append((a, b))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("analysis")
    p.add_argument("--out", required=True, help="plan directory (use a new one per version)")
    p.add_argument("--profile", choices=sorted(PROFILES), default="podcast")
    p.add_argument("--max-pause", type=float)
    p.add_argument("--target-gap", type=float)
    p.add_argument("--apply", default="", help="comma list of suggested classes to apply: stutter,retake,filler")
    p.add_argument("--range", help="start,end in source seconds")
    p.add_argument("--force-cut", action="append", help="start,end source seconds to remove (repeatable)")
    p.add_argument("--force-keep", action="append", help="start,end source seconds never to cut (repeatable)")
    p.add_argument("--loudness", type=float, default=-14.0, help="QA loudness target (YouTube -14, podcast audio -16)")
    a = p.parse_args()
    try:
        analysis = read_json(a.analysis)
        rng = parse_pairs([a.range])[0] if a.range else (None, None)
        result = plan(analysis, rng[0], rng[1], a.profile,
                      {"max_pause": a.max_pause, "target_gap": a.target_gap},
                      {x.strip() for x in a.apply.split(",") if x.strip()},
                      parse_pairs(a.force_cut), parse_pairs(a.force_keep))
        tmap = write_outputs(analysis, result, a.out, loudness=a.loudness)
    except (ValueError, OSError, KeyError) as exc:
        p.exit(2, f"plan_longform: {exc}\n")
    applied = sum(c["applied"] for c in result["cuts"])
    print(f"plan → {a.out}: {applied} cuts applied, {len(result['cuts']) - applied} suggested; "
          f"{result['range'][1] - result['range'][0]:.1f}s → {tmap.duration:.1f}s")


if __name__ == "__main__":
    main()
