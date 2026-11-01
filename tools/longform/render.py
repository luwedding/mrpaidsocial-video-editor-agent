#!/usr/bin/env python3
"""render.py — render a long-form cut or a vertical short from a plan.

  Long form (16:9 master, cut plan applied, loudness-normalized):
    python3 render.py longform --source EP.mp4 --plan work/plan-v1 --out renders/ep-v1.mp4 \
        [--loudness -14] [--fps source]

  One short (cut + pause-tightened + reframed to 1080x1920 + loudness):
    python3 render.py short --source EP.mp4 --analysis work/analysis.json \
        --start 812.4 --end 861.9 --out shorts/clip03-v1.mp4 \
        [--layout follow|split|center|fit] [--boxes "Ana=0,0,0.5,1;Luis=0.5,0,1,1"]
        [--zoom 1.3] [--hold 1.2] [--no-tighten] [--force-cut a,b] [--apply stutter,retake]

Layouts for a podcast wide shot:
  follow  9:16 window that hard-cuts to whoever is talking (speaker timeline from the
          analysis; needs --boxes or analysis boxes). Short turns (< --hold) don't switch.
  split   two speakers stacked top/bottom, each a 1080x960 crop of their box.
  center  plain centre crop.
  fit     whole 16:9 frame over a blurred full-bleed copy of itself.

Every cut goes through reel-recut's frame/sample-grid assembler (cut_timeline.py): one grid
for video and audio, lossless batches, 3 ms edge fades, verified frame + sample counts.
Then ONE encode applies the reframe and a two-pass EBU R128 loudnorm.

A short also gets: <out>.words.json (output-time words), <out>.qa-manifest.json (for
tools/video-qa) and <out>.reel-spec.json — a reel-recut spec with DRAFT captions to clean
before adding banner/captions with build_reel.py.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lf_common import (HESITATIONS, TimeMap, binary, cut_timeline, norm, probe, read_json,  # noqa: E402
                       run, write_json)
import plan_longform  # noqa: E402

W, H = 1080, 1920


def loudnorm_measure(path, target, tp=-1.0, lra=11):
    res = subprocess.run([binary("ffmpeg"), "-nostdin", "-hide_banner", "-i", str(path), "-vn", "-af",
                          f"loudnorm=I={target}:TP={tp}:LRA={lra}:print_format=json", "-f", "null", "-"],
                         text=True, capture_output=True)
    if res.returncode:
        raise RuntimeError(f"loudnorm measure failed: {res.stderr[-2000:]}")
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", res.stderr, re.S)
    if not m:
        raise RuntimeError("loudnorm printed no measurement")
    return json.loads(m.group(0))


def loudnorm_filter(meas, target, tp=-1.0, lra=11):
    return (f"loudnorm=I={target}:TP={tp}:LRA={lra}:measured_I={meas['input_i']}:measured_TP={meas['input_tp']}"
            f":measured_LRA={meas['input_lra']}:measured_thresh={meas['input_thresh']}:offset={meas['target_offset']}"
            ":linear=true,aresample=48000")


def measure(path):
    """Integrated loudness + true peak of a finished file (receipt, not intention)."""
    m = loudnorm_measure(path, -14)
    return {"lufs": float(m["input_i"]), "truePeak": float(m["input_tp"])}


def assemble(source, keeps, work, fps):
    ct = cut_timeline()
    base = Path(work) / "base.mp4"
    receipt = ct.render_base(source, base, Path(work) / "cut", keeps, fps)
    return base, receipt


def resolve_fps(info, fps):
    if fps == "source":
        return info["fpsRational"] or str(round(info["fps"] or 30, 3))
    return fps


def finish(base, out, vf, target, crf=18, preset="medium"):
    meas = loudnorm_measure(base, target)
    cmd = [binary("ffmpeg"), "-nostdin", "-v", "error", "-n", "-i", base]
    if vf:
        cmd += ["-vf", vf, "-c:v", "libx264", "-crf", crf, "-preset", preset, "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-af", loudnorm_filter(meas, target), "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", out]
    run(cmd)
    return meas


def cmd_longform(a):
    plan_dir = Path(a.plan)
    keeps = read_json(plan_dir / "keeps.json")
    out = Path(a.out)
    if out.exists():
        raise ValueError(f"{out} exists — new versions are new files")
    info = probe(a.source)
    if info["hdr"]:
        print("WARNING: HDR (HLG/PQ) source — this path writes 8-bit BT.709; check skin tones on the render.")
    work = out.with_name(out.stem + "_work")
    base, receipt = assemble(a.source, keeps, work, resolve_fps(info, a.fps))
    meas = finish(base, out, None, a.loudness)
    final = measure(out)
    receipt.update({"loudnessIn": float(meas["input_i"]), "loudnessOut": final, "plan": str(plan_dir)})
    write_json(out.with_suffix(".receipt.json"), receipt)
    manifest = read_json(plan_dir / "qa-manifest.json")
    manifest["video"] = str(out.resolve())
    manifest["expectedDuration"] = round(receipt["duration"], 3)  # the grid-snapped length actually rendered
    write_json(out.with_suffix(".qa-manifest.json"), manifest)
    print(f"long form → {out}  {receipt['duration']:.1f}s  {final['lufs']:.1f} LUFS  TP {final['truePeak']:.1f} dBTP")


def parse_boxes(spec):
    from analyze import parse_named
    return parse_named(spec, "box")


def even(x):
    return int(x) // 2 * 2


def speaker_switches(segments, tmap, hold, names):
    """Output-time [(t, speaker)] with turns shorter than `hold` absorbed."""
    out = []
    for s in segments:
        if s["speaker"] not in names:
            continue
        a, b = tmap.to_output_clamped(s["start"]), tmap.to_output_clamped(s["end"])
        if b is None or a is None or b - a < hold:
            continue
        if not out or out[-1][1] != s["speaker"]:
            out.append((round(max(0.0, a), 3), s["speaker"]))
    if out:
        out[0] = (0.0, out[0][1])
    return out


def piecewise(switches, values):
    """ffmpeg expression: value of the active speaker at time t."""
    expr = str(values[switches[-1][1]])
    for (t, _), (_, spk) in zip(reversed(switches[1:]), reversed(switches[:-1])):
        expr = f"if(lt(t\\,{t})\\,{values[spk]}\\,{expr})"
    return expr


def vertical_filter(layout, info, boxes, switches, zoom):
    iw, ih = info["width"], info["height"]
    if layout == "center" or (layout == "follow" and not switches):
        ch = even(ih / zoom)
        cw = even(min(iw, ch * 9 / 16))
        return f"crop={cw}:{ch}:{(iw - cw) // 2}:{(ih - ch) // 2},scale={W}:{H}:flags=lanczos,setsar=1"
    if layout == "fit":
        return (f"split[bg][fg];[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=30:2[b];"
                f"[fg]scale={W}:-2[f];[b][f]overlay=(W-w)/2:(H-h)/2,setsar=1")
    if layout == "follow":
        ch = even(ih / zoom)
        cw = even(min(iw, ch * 9 / 16))
        xs, ys = {}, {}
        for n, (x0, y0, x1, y1) in boxes.items():
            xs[n] = int(min(max((x0 + x1) / 2 * iw - cw / 2, 0), iw - cw))
            ys[n] = int(min(max((y0 + y1) / 2 * ih - ch / 2, 0), ih - ch))
        return (f"crop={cw}:{ch}:'{piecewise(switches, xs)}':'{piecewise(switches, ys)}',"
                f"scale={W}:{H}:flags=lanczos,setsar=1")
    if layout == "split":
        names = list(boxes)[:2]
        if len(names) < 2:
            raise ValueError("split layout needs two --boxes")
        parts = []
        for k, n in enumerate(names):
            x0, y0, x1, y1 = boxes[n]
            bh = (y1 - y0) * ih
            ch = even(min(ih, bh))
            cw = even(min(iw, ch * W / (H / 2)))
            ch = even(cw * (H / 2) / W)
            cx = int(min(max((x0 + x1) / 2 * iw - cw / 2, 0), iw - cw))
            cy = int(min(max((y0 + y1) / 2 * ih - ch / 2, 0), ih - ch))
            parts.append(f"[s{k}]crop={cw}:{ch}:{cx}:{cy},scale={W}:{H // 2}:flags=lanczos,setsar=1[p{k}]")
        return "split[s0][s1];" + ";".join(parts) + ";[p0][p1]vstack"
    raise ValueError(f"unknown layout {layout}")


def draft_captions(words, max_words=5, max_len=2.2):
    caps, cur = [], []
    for w in words:
        if norm(w["text"]) in HESITATIONS:
            continue
        if cur and (len(cur) >= max_words or w["end"] - cur[0]["start"] > max_len or w["start"] - cur[-1]["end"] > 0.5):
            caps.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.!?…]$", w["text"]) or (re.search(r"[,;:]$", w["text"]) and len(cur) >= 2):
            caps.append(cur)
            cur = []
    if cur:
        caps.append(cur)
    out = []
    for c in caps:
        text = re.sub(r"[.,;:]+$", "", " ".join(x["text"] for x in c)).strip()
        out.append({"start": round(c[0]["start"], 2), "end": round(c[-1]["end"] + 0.05, 2), "text": text})
    for x, y in zip(out, out[1:]):
        x["end"] = min(x["end"], y["start"])
    return out


def cmd_short(a):
    out = Path(a.out)
    if out.exists():
        raise ValueError(f"{out} exists — new versions are new files")
    analysis = read_json(a.analysis)
    info = probe(a.source)
    if abs(info["duration"] - analysis["source"]["duration"]) > 0.5:
        raise ValueError("analysis.json was made from a different file (duration mismatch)")
    if info["hdr"]:
        print("WARNING: HDR (HLG/PQ) source — output is 8-bit BT.709 SDR; check skin tones on the render.")
    boxes = parse_boxes(a.boxes) if a.boxes else (analysis.get("video") or {}).get("boxes") or {}
    work = out.with_name(out.stem + "_work")
    profile = "short" if not a.no_tighten else "podcast"
    result = plan_longform.plan(analysis, a.start, a.end, profile,
                                {"max_pause": a.max_pause},
                                {x for x in a.apply.split(",") if x},
                                plan_longform.parse_pairs(a.force_cut), plan_longform.parse_pairs(a.force_keep),
                                fps=resolve_fps(info, a.fps))
    if a.no_tighten:
        for c in result["cuts"]:
            if c["reason"] not in ("head", "tail", "manual"):
                c["applied"] = False
        from lf_common import keeps_from_cuts, merge_ranges
        result["keeps"] = cut_timeline().plan(keeps_from_cuts(
            result["range"][0], result["range"][1],
            merge_ranges([[c["start"], c["end"]] for c in result["cuts"] if c["applied"]])), resolve_fps(info, a.fps))["keeps"]
    tmap = plan_longform.write_outputs(analysis, result, work / "plan", label="longform-to-shorts", loudness=a.loudness)
    fps = resolve_fps(info, a.fps)
    base, receipt = assemble(a.source, result["keeps"], work, fps)
    tmap = TimeMap(result["keeps"])  # already on the render grid
    switches = []
    if a.layout == "follow":
        if not boxes:
            raise ValueError("follow layout needs speaker boxes (--boxes, or analyze.py --speaker-boxes)")
        known = set(analysis["speakers"]["names"])
        if known and not known & set(boxes):
            raise ValueError(f"speaker names {sorted(known)} (from {analysis['speakers']['source']}) match none of the "
                             f"box names {sorted(boxes)} — name the boxes after the speakers, e.g. "
                             f"--boxes \"{sorted(known)[0]}=0,0,0.5,1;...\"")
        switches = speaker_switches(analysis["speakers"]["timeline"], tmap, a.hold, set(boxes))
        if not switches:
            print("NOTE: no usable speaker turns in this clip — falling back to a centre crop.")
    vf = vertical_filter(a.layout, info, boxes, switches, a.zoom)
    meas = finish(base, out, vf, a.loudness)
    final = measure(out)
    words = tmap.words([w for w in analysis["words"] if a.start - 0.01 <= w["start"] and w["end"] <= a.end + 0.01])
    write_json(out.with_suffix(".words.json"), words)
    manifest = read_json(work / "plan" / "qa-manifest.json")
    manifest.update({"video": str(out.resolve()), "expected": {"width": W, "height": H, "fps": round(info["fps"] or 30, 3)},
                     "expectedDuration": round(receipt["duration"], 3), "words": words})
    write_json(out.with_suffix(".qa-manifest.json"), manifest)
    spec = {"_comment": "DRAFT captions from the transcript: clean them (fix names, drop fillers, house copy rules), "
                        "write a 1-2 line banner hook, then: python3 <pack>/.claude/skills/reel-recut/scripts/build_reel.py this.json",
            "source": out.name, "output": out.stem + "-captioned.mp4", "work_dir": out.stem + "_reel_work",
            "fps": "source", "silence_cut": {"enabled": False}, "words": out.with_suffix(".words.json").name,
            "banner": ["HOOK LINE ONE", "HOOK LINE TWO"], "captions": draft_captions(words), "callouts": []}
    write_json(out.with_suffix(".reel-spec.json"), spec)
    receipt.update({"clip": [a.start, a.end], "layout": a.layout, "switches": switches,
                    "loudnessIn": float(meas["input_i"]), "loudnessOut": final})
    write_json(out.with_suffix(".receipt.json"), receipt)
    print(f"short → {out}  {receipt['duration']:.1f}s (from {a.end - a.start:.1f}s)  {a.layout}"
          f"{f' {len(switches)} shots' if switches else ''}  {final['lufs']:.1f} LUFS  TP {final['truePeak']:.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    lf = sub.add_parser("longform")
    lf.add_argument("--source", required=True)
    lf.add_argument("--plan", required=True)
    lf.add_argument("--out", required=True)
    lf.add_argument("--loudness", type=float, default=-14.0)
    lf.add_argument("--fps", default="source")
    sh = sub.add_parser("short")
    sh.add_argument("--source", required=True)
    sh.add_argument("--analysis", required=True)
    sh.add_argument("--start", type=float, required=True)
    sh.add_argument("--end", type=float, required=True)
    sh.add_argument("--out", required=True)
    sh.add_argument("--layout", choices=["follow", "split", "center", "fit"], default="follow")
    sh.add_argument("--boxes")
    sh.add_argument("--zoom", type=float, default=1.0, help="punch-in for follow/center (1.0 = full height)")
    sh.add_argument("--hold", type=float, default=1.2, help="min turn length (s) that earns a camera switch")
    sh.add_argument("--no-tighten", action="store_true", help="keep the clip's natural pauses")
    sh.add_argument("--max-pause", type=float)
    sh.add_argument("--apply", default="")
    sh.add_argument("--force-cut", action="append")
    sh.add_argument("--force-keep", action="append")
    sh.add_argument("--loudness", type=float, default=-14.0)
    sh.add_argument("--fps", default="30")
    a = p.parse_args()
    if a.cmd == "short" and (a.end <= a.start or a.zoom < 1):
        p.exit(2, "render: need --end > --start and --zoom >= 1\n")
    try:
        (cmd_longform if a.cmd == "longform" else cmd_short)(a)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        p.exit(2, f"render: {exc}\n")


if __name__ == "__main__":
    main()
