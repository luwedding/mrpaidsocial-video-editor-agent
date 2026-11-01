---
name: longform-to-shorts
description: >-
  Pull the best SHORTS (9:16 Reels / TikTok / YouTube Shorts, 15-90 s) out of a LONG
  recording — podcast, interview, livestream, webinar, talking-head video — by analyzing
  audio and picture: rank candidate moments by hook, payoff, energy, pace, laughs/reactions,
  back-and-forth and motion, read the shortlist and pick the ones that stand alone, then
  render each one tightened and reframed vertical (camera follows whoever is talking, or a
  two-person split screen), loudness-normalized, with draft captions ready for reel-recut.
  Use for "make shorts from this podcast", "find the best clips", "cut 10 reels out of this
  episode", "clip this interview for TikTok", "viral moments from this stream". Not for
  editing the full episode (long-form-edit), a single raw vertical take (reel-recut), or one
  long file of separate hooks (hook-splitter).
---

# longform-to-shorts — find the moments, cut them vertical

The machine measures; you judge. `find_clips.py` scores every run of whole sentences on
signals it can measure from the file. It does not know what is interesting, funny or true.
The shortlist tells you **where to look**; reading the text and deciding is your job, and it
is the part that makes the clips good.

Tools: `tools/longform/` (README there). Media stays in the projects directory.

## Workflow

### 0. Analysis (shared with long-form-edit)

If the episode already has `work/analysis.json`, reuse it. Otherwise run long-form-edit
steps 0–2 (probe, transcribe with word timestamps, `analyze.py`). For a two-person show
the speaker timeline matters here: it drives the reframe. Name speakers consistently in
`--speaker-channels` and `--speaker-boxes` (the box names must match the speaker names).
To get boxes, grab one frame (`ffmpeg -ss 60 -i EP.mp4 -frames:v 1 frame.png`), LOOK at it,
and write each person's region as frame fractions `x0,y0,x1,y1` (head and shoulders with
some headroom).

### 1. Shortlist

```bash
python3 tools/longform/find_clips.py work/analysis.json --out work/clips --min 20 --max 60 --count 15
```

Each candidate starts and ends on a sentence boundary. Signals: **hook** (question, number,
strong claim word, second person, short opener; heavy penalty for opening on a connector
like "and / but / so / pero / entonces / eso" — context the viewer never heard), **payoff**
(closed sentence followed by a pause or a turn change), **energy** (loudness vs the episode,
and its variation), **pace**, **reaction** (laughs inside), **exchange** (speaker turns),
**visual** motion; minus hesitation/filler density and dead air. Overlapping windows are
suppressed. Ask for ~2× the clips you will ship.

### 2. Judge (read `clips.md`, every candidate)

Keep a clip only if all hold:

1. **Stands alone.** A viewer with zero context understands it. Unresolved "he", "that
   thing", "like I said" → extend the start to include the referent, or drop it.
2. **Hook in the first 2 s.** If the best line is 10 s in, start there — or plan a cold
   open (the payoff line first, then the setup) and say so.
3. **Payoff.** It lands: a punchline, a number, a reversal, a clear opinion. Ends on it,
   not on "...so yeah".
4. **Safe to publish.** Nothing off-record, private, defamatory or a sponsor's claim out of
   context. Flag anything doubtful to the user instead of deciding.

Trim to sentence edges (use `start`/`end` of the sentences in `analysis.json`), then write
`work/clips/selected.json`: `[{id, start, end, title, hook, why, layout}]`. Show the user the
list (time range, hook line, why) before rendering a batch of more than ~5.

### 3. Render each clip

```bash
python3 tools/longform/render.py short --source EP.mp4 --analysis work/analysis.json \
    --start 812.4 --end 861.9 --out shorts/clip03-v1.mp4 --layout follow \
    [--boxes "Ana=0,0,0.5,1;Luis=0.5,0,1,1"] [--zoom 1.3] [--apply retake,stutter]
```

- **Pacing:** the clip is re-planned with the `short` profile (pauses > 0.3 s cut to
  0.12 s, hesitations removed, laughs protected). `--no-tighten` keeps natural pauses
  (emotional moments). `--apply retake,stutter` after reading the suggestions in
  `shorts/clip03-v1_work/plan/report.md`.
- **Layouts:** `follow` — a 9:16 window that hard-cuts to whoever is talking (turns shorter
  than `--hold` 1.2 s don't switch; `--zoom` punches in on the box); `split` — two speakers
  stacked, each a 1080×960 crop of their box (good for fast back-and-forth); `center`;
  `fit` — the whole 16:9 frame over a blurred copy (screen shares, wide group shots).
  Render ONE frame of two layouts and show both before choosing for a whole batch.
- Output: `clip03-v1.mp4` (1080×1920, -14 LUFS), `.words.json` (output-time), `.qa-manifest.json`,
  `.receipt.json` (shots, measured loudness), and `.reel-spec.json`.

### 4. Captions + hook title (reel-recut)

`clip03-v1.reel-spec.json` is a reel-recut spec pointing at the rendered short, with
**draft** captions chunked from the transcript and a placeholder banner. Clean the captions
(fix names whisper mangled, drop fillers, house copy rules from MASTER_CONTEXT.md), write a
1–2 line banner hook, set the brand `style`, then:

```bash
python3 .claude/skills/reel-recut/scripts/build_reel.py shorts/clip03-v1.reel-spec.json
```

Keep captions out of the speaker's face in `split` (move `style.caption_cy` to the seam at
~960) and out of the platform UI zones (top 14 %, bottom 26 %). For the native Instagram text
look on many clips, use edits-hook-overlay instead of a banner.

### 5. QA, name, deliver

- `npm --prefix tools/video-qa run qa:video -- --manifest shorts/clip03-v1.qa-manifest.json`
  on the base render; QA the captioned file with reel-recut's manifest. Fix every HIGH.
- Frame-tile every short at each speaker switch: the crop must land on a face, not a
  shoulder or the gap between two people. A bad box is fixed in `--boxes`, not by hand.
- More than two files → **naming-convention** (`<show>-<ep>-<clip#>-<topic>-<layout>-v1.mp4`).
- Deliver on **video-review-canvas** (gallery of all clips), reply leads with the URL.

## Limits (honest)

- No face detection: framing comes from the boxes you set by looking at a frame. A host who
  moves out of their box gets cropped; use `fit` or re-box.
- The score is a filter, not taste. Never ship the top N unread.
- Separate camera files are not switched; reframing works on a single wide shot.
