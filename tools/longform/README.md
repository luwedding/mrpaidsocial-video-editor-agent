# tools/longform — analysis, cut planning and rendering for long recordings

Backs two skills: **long-form-edit** (the episode master) and **longform-to-shorts** (the
best moments, cut vertical). Python 3.10+, ffmpeg/ffprobe and numpy. No network calls,
no model downloads. Renders reuse reel-recut's `cut_timeline.py` assembler.

```
EP.mp4 + words.json
   │  analyze.py ─────────→ analysis.json  (audio, picture, speakers, words, sentences)
   ├─ plan_longform.py ───→ plan-vN/  keeps · edl · cuts · words-output · qa-manifest ·
   │                                  transcript.md · chapters.draft.md · report.md
   │    render.py longform → ep-vN.mp4 (+ receipt, qa-manifest)          → video-qa
   └─ find_clips.py ──────→ clips/ clips.json · clips.md  → (agent judges) selected.json
        render.py short ──→ clipNN-vN.mp4 1080x1920 (+ words, qa-manifest, receipt,
                                                    reel-spec for reel-recut captions)
```

## analyze.py

`analyze.py EP.mp4 --words words.json --out analysis.json [--speaker-channels "A=0,B=1"]
[--speaker-boxes "A=x0,y0,x1,y1;B=..."]`

- **Audio** — every stream merged, 8 kHz, 25 ms RMS hops. Noise floor = 10th percentile,
  speech = 90th, gate = floor + max(6 dB, 35 % of the range). Stored: `loudnessDb` every
  0.1 s, `silences` (≥ 0.3 s), `nonverbal` (≥ 0.4 s above the gate with no word within
  120 ms: laughs, reactions, applause).
- **Picture** — 4 fps, 160×90 luma: `motion` per sample, `sceneCuts` (local peaks over
  4× median motion, ≥ 1 s apart), `blackRanges`, and per-box motion when boxes are given.
- **Speakers** — transcript labels (whisperX/AssemblyAI/Deepgram `speaker`) win; else one
  mic per person: named channels, or auto when two channels differ (correlation < 0.9 or
  median gap > 4 dB) → `S1`/`S2`; dominant by ≥ 6 dB while voiced, 0.5 s majority smoothing.
  Else the box with ≥ 25 % more motion while speech is present, 1.5 s smoothing.
- **Words** — whisper-cli `-ojf`, OpenAI `verbose_json`, whisperX `segments[].words`, or a
  flat list. Sentences split on end punctuation, a 0.7 s pause, a speaker change or 20 s.

## plan_longform.py

Profiles `podcast` / `tight` / `short` (see the skill). AUTO: head/tail, pauses (edges walked
along voiced audio from the transcript times, ≥ 85 % silent, exactly the target gap kept),
hesitations (whole word always removed). SUGGESTED (`--apply`): stutter, retake, filler.
Nonverbal events split gaps (air on each side is judged alone) and are never cut into;
`--force-keep` / `--force-cut` override. Keeps are snapped to the render's frame grid.

## find_clips.py

Every run of whole sentences between `--min` and `--max` seconds is scored (weights in
`WEIGHTS`): hook, payoff, energy, pace, reaction, exchange, visual, minus penalties. Overlaps
> 25 % of the shorter clip are suppressed. A ranking aid, not an editorial decision.

## render.py

- `longform` — assembler → two-pass loudnorm (`-c:v copy`). Receipt holds measured LUFS/TP.
- `short` — re-plans the clip at the `short` profile, assembles at `--fps` (30 default),
  then one encode with the reframe (`follow` piecewise crop by speaker, `split`, `center`,
  `fit`) and loudnorm. Refuses to overwrite outputs.

## Tests

`python3 -m unittest discover -s tools/longform/tests -v` (also run by `scripts/test.sh`).
Media cases build a 40 s two-mic synthetic podcast (tones per channel, a wordless laugh,
pauses, "um", a stutter, a restart) and skip without numpy/ffmpeg.
