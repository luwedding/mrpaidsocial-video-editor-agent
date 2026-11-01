---
name: long-form-edit
description: >-
  Edit a LONG recording (podcast, interview, video-podcast, talking-head tutorial, 5-120+
  min, usually 16:9) into a finished YouTube/podcast master: analyze audio and picture in
  one pass, cut dead air, hesitations ("um", "eh", "ehm"), stutters and restarted takes with
  waveform-refined edges, never cut laughs or reactions, write chapters, normalize loudness
  (-14 LUFS YouTube / -16 podcast audio), render and verify the file. Use for "edit this
  podcast", "edit the full episode", "clean up this interview", "cut the dead air out of
  this hour-long video", "make the YouTube version", "add chapters", or a revision on a
  delivered long cut. To pull SHORTS out of a long recording use longform-to-shorts (it
  reuses this skill's analysis). Not for a short talking-head reel (reel-recut /
  branded-ad-edit) or for splitting a file of many hooks (hook-splitter).
---

# long-form-edit — the episode master

Long-form is a different job from a reel. A reel is cut to the bone; an episode is cut so
it **sounds like the conversation, minus the parts nobody wants to hear**. Conversation
needs air: a podcast cut at reel pacing sounds like an argument. Laughs, reactions and the
beat before a punchline are content. The defaults below are tuned for that.

Tools live in `tools/longform/` (Python + ffmpeg + numpy; details in its README). Media
stays in the projects directory (`<projects dir>/<slug>/`), never in this repo.

## Workflow

### 0. Intake and probe

- `ffprobe` every file: duration, fps, resolution, **audio streams and channels**, HDR.
  A podcast often arrives as one wide shot plus **one mic per person** (separate streams,
  or L/R of a stereo file). That is the best speaker signal there is — ask for the
  isolated mic tracks if they exist and only the mixed track was sent.
- Multi-camera with separate files is not built yet (see Limits). Ask for the switched
  program feed or the wide shot.
- Start the project receipt: `python3 tools/editor/editor.py project init <slug> --source
  <file> --lane long-form-edit`.

### 1. Transcribe (word timestamps)

Word-level timestamps are required; diarized ones are better. Any of these load directly:

- **local:** `whisper-cli -m <ggml-large-v3> -f audio.wav -l <lang> -ojf -of words`
  (whisper.cpp; extract `audio.wav` with `ffmpeg -i EP.mp4 -vn -ac 1 -ar 16000 audio.wav`).
- **whisperX** (`--diarize`): per-word `speaker` labels; the analysis uses them first.
- **OpenAI** `whisper-1` `verbose_json` + `timestamp_granularities=word` (25 MB limit:
  mono 48 kbps mp3 holds ~70 min — split longer files and offset the times).
- Set the language explicitly. Spanish auto-detect on a bilingual episode drifts.

Whisper drifts on hour-long files and pads sentence ends by 0.1–1.0 s. The planner does
not trust word edges blindly: every pause edge is walked out along the actual waveform.

### 2. Analyze (one pass over audio + picture)

```bash
python3 tools/longform/analyze.py EP.mp4 --words words.json --out work/analysis.json \
    [--speaker-channels "Ana=0,Luis=1"] [--speaker-boxes "Ana=0,0,0.5,1;Luis=0.5,0,1,1"]
```

It reports the noise floor, speech level and an **adaptive** speech gate (a fixed -30 dB
gate fails on noisy rooms — see reel-recut), silences, **nonverbal events** (loud spans with
no words: laughs, reactions, applause, stings), scene cuts, black spans, motion, and a
speaker timeline from the best source available: transcript labels → one mic per person
(auto-detected when two channels clearly differ, named `S1`/`S2` unless you name them) →
motion inside each person's box in a wide shot. Read the summary it prints; if
`speakers: none` on a two-person show, fix that now (channels or boxes) — the shorts lane
depends on it.

### 3. Plan the cut

```bash
python3 tools/longform/plan_longform.py work/analysis.json --out work/plan-v1 --profile podcast
```

| profile | cuts pauses longer than | leaves | use for |
|---|---|---|---|
| `podcast` (default) | 0.9 s | 0.35 s | conversation, interviews |
| `tight` | 0.55 s | 0.22 s | solo talking head, tutorials, YouTube essays |
| `short` | 0.3 s | 0.12 s | used by longform-to-shorts |

Applied automatically: dead air at head/tail, real silences (≥ 85 % below the gate), and
hesitations with air around them — the cut always swallows the whole "um", even if less
air is left. **Suggested only** (listed in `report.md`): stutters ("the the", "que que"),
retakes (a 3+ word phrase restarted by the same speaker within ~12 s) and lexical fillers
("like", "you know", "o sea", "este"). Read every suggestion; "no, no, no" is a joke, not a
stutter. Then re-plan into a NEW directory:

```bash
python3 tools/longform/plan_longform.py work/analysis.json --out work/plan-v2 \
    --apply retake,stutter --force-keep 1520.2,1531.0 --force-cut 2210.4,2290.0
```

`--force-cut` removes a tangent, a sponsor read the client dropped, an off-the-record
moment; `--force-keep` protects a range from every automatic cut. Laughs and reactions are
always protected; a pause cut that runs into one is trimmed back to its edge.

The plan directory holds `keeps.json`, `edl.json` (raw↔master windows), `cuts.json` with
reasons, `words-output.json`, `qa-manifest.json`, `transcript.md` (output-time, one line
per sentence) and `chapters.draft.md`. Keeps are already snapped to the render's frame grid,
so every timestamp in these files matches the rendered file exactly.

**Content cuts are the editor's call, not the script's.** Read `transcript.md` end to end
before rendering a first cut. Propose tangents, repetition and slow intros to the user
with timestamps; do not remove content on your own judgment on a client's show.

### 4. Render

```bash
python3 tools/longform/render.py longform --source EP.mp4 --plan work/plan-v2 \
    --out renders/ep-v1.mp4 [--loudness -14] [--fps source]
```

The cut runs through reel-recut's frame/sample-grid assembler (`cut_timeline.py`: one grid
for video and audio, lossless batches, 3 ms seam fades, verified frame + sample counts),
then a two-pass EBU R128 loudnorm with the video stream copied. `-14` LUFS is the YouTube
target; use `-16` for a podcast audio master. It writes a receipt with the measured
loudness and true peak of the finished file, and `ep-v1.qa-manifest.json`.
Outputs are never overwritten — a new version is a new file.

### 5. Chapters

Open `chapters.draft.md`: candidate starts at the biggest pauses, spaced ≥ 150 s. Read the
transcript around each, move them to real topic changes, and write a title per chapter (3–7
words, what the viewer gets, not "Intro"/"Part 2"). YouTube rules: first at 0:00, at least
three, each ≥ 10 s. Deliver them as the description block.

### 6. QA the rendered file

```bash
npm --prefix tools/video-qa run qa:video -- --manifest renders/ep-v1.qa-manifest.json
```

Every applied cut is a `dialogueCut` event, so L2 checks clipped words, duplicated words and
seam dead air at each one. Expect a few MEDIUM `dead_air`/`seam_dead_air` findings in a
podcast: 0.6–0.9 s of air between turns and the beat after a laugh are deliberate at the
`podcast` profile (laughs are listed as `nonverbal` events in the manifest). Fix every
HIGH. Then spot-check by listening: 5 random seams plus every retake cut.

### 7. Deliver and revise

Deliver on **video-review-canvas** (a 720p proxy for the player; the master stays on disk)
and lead with the URL, plus the chapter block. Revisions: read notes back, screenshot the
noted frame, re-plan with `--force-cut`/`--force-keep` into a new plan dir, render a new
version file, re-run QA, per-note evidence table (pipeline Stage 6).

## Limits (honest)

- **Separate camera files** (true multicam) are not synced or switched here yet; the shorts
  lane can reframe a single wide shot by speaker instead.
- HDR (HLG/PQ) sources are written as 8-bit BT.709 — check skin tones on the render.
- No music/ambience: use sound-design for intro/outro music. No B-roll pass in this lane.
- The speaker timeline from picture motion is a heuristic; per-mic channels or diarized
  words are reliable. Check the reported turn count against what you see.
