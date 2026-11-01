"""Offline regressions for the long-form tools. Pure-logic cases run anywhere; media cases
generate a ~40 s synthetic two-mic "podcast" with ffmpeg (tones per channel, a wordless
laugh, long pauses, an "um", a stutter and a restarted phrase)."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.dont_write_bytecode = True
import lf_common  # noqa: E402
import find_clips  # noqa: E402
import plan_longform  # noqa: E402

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))

SCRIPT = [
    ("A", "Why do most podcasts fail?", 2.5),
    ("B", "um", 0.15),
    ("B", "Because nobody edits the the first minute.", 0.6),
    ("A", "So the hook is the problem so the hook is everything.", 2.0),
    ("LAUGH", None, 0.8),
    ("B", "Exactly. You lose half the audience in ten seconds.", 1.8),
    ("A", "How do you fix it?", 0.5),
    ("B", "Cut the dead air, open on the best line, and never start with hello.", 3.0),
    ("A", "That is the secret nobody tells you.", 1.0),
]


def build_fixture(root):
    t, words, segs = 1.5, [], []
    for spk, text, gap in SCRIPT:
        if spk == "LAUGH":
            segs.append(("L", t, t + 1.2))
            t += 1.2 + gap
            continue
        for tok in text.split():
            words.append({"text": tok, "start": round(t, 3), "end": round(t + 0.32, 3)})
            segs.append((spk, t, t + 0.32))
            t += 0.4
        t += gap
    dur = round(t + 2.0, 1)

    def expr(ch):
        terms = []
        for spk, a, b in segs:
            if spk == ch:
                terms.append(f"between(t,{a:.3f},{b:.3f})*0.5*sin(2*PI*{220 if ch == 'A' else 330}*t)")
            if spk == "L":
                terms.append(f"between(t,{a:.3f},{b:.3f})*0.35*sin(2*PI*600*t)*sin(2*PI*7*t)")
        return "+".join(terms) + "+0.002*(random(0)-0.5)"

    video = root / "podcast.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=30:duration={dur}",
                    "-f", "lavfi", "-i", f"aevalsrc=exprs='{expr('A')}|{expr('B')}':s=48000:d={dur}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", video],
                   check=True)
    (root / "words.json").write_text(json.dumps({"words": words}))
    return video, root / "words.json"


class TranscriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="longform-")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, data):
        p = self.root / "w.json"
        p.write_text(json.dumps(data))
        return p

    def test_word_shapes_normalize(self):
        flat = [{"text": "hola", "start": 0, "end": 0.3}]
        openai = {"words": [{"word": "hola", "start": 0, "end": 0.3}]}
        whisperx = {"segments": [{"speaker": "SPEAKER_00", "words": [{"word": "hola", "start": 0, "end": 0.3}]}]}
        cpp = {"transcription": [{"tokens": [{"text": " ho", "offsets": {"from": 0, "to": 150}},
                                             {"text": "la", "offsets": {"from": 150, "to": 300}},
                                             {"text": "[_TT_30]", "offsets": {"from": 300, "to": 300}}]}]}
        for data in (flat, openai, whisperx, cpp):
            words = lf_common.load_words(self.write(data))
            self.assertEqual([(w["text"], w["start"], w["end"]) for w in words], [("hola", 0, 0.3)])
        self.assertEqual(lf_common.load_words(self.write(whisperx))[0]["speaker"], "SPEAKER_00")

    def test_sentences_split_on_punctuation_pause_and_speaker(self):
        words = [{"text": "Hi.", "start": 0, "end": 0.2, "speaker": "A"},
                 {"text": "so", "start": 0.3, "end": 0.4, "speaker": "A"},
                 {"text": "yes", "start": 0.5, "end": 0.6, "speaker": "B"},
                 {"text": "ok", "start": 2.0, "end": 2.2, "speaker": "B"}]
        self.assertEqual([s["text"] for s in lf_common.sentences(words)], ["Hi.", "so", "yes", "ok"])

    def test_time_map_drops_words_cut_through(self):
        tm = lf_common.TimeMap([[0, 1], [2, 3]])
        self.assertEqual(tm.to_output(2.5), 1.5)
        self.assertIsNone(tm.to_output(1.5))
        out = tm.words([{"text": "a", "start": 0.1, "end": 0.3}, {"text": "b", "start": 0.9, "end": 2.1},
                        {"text": "c", "start": 2.2, "end": 2.4}])
        self.assertEqual([(w["text"], round(w["start"], 3)) for w in out], [("a", 0.1), ("c", 1.2)])

    def test_hook_penalises_dangling_start(self):
        words = [{"text": t, "start": i * 0.3, "end": i * 0.3 + 0.2} for i, t in
                 enumerate("Why nobody edits the first minute? But that was it.".split())]
        sents = lf_common.sentences(words)
        good, _ = find_clips.hook_score(sents[0], words)
        bad, notes = find_clips.hook_score(sents[1], words)
        self.assertGreater(good, bad)
        self.assertIn("dangling-start:but", notes)


@unittest.skipUnless(HAVE_NUMPY and HAVE_FFMPEG, "needs numpy + ffmpeg")
class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="longform-media-")
        cls.root = Path(cls.tmp.name)
        cls.video, cls.words = build_fixture(cls.root)
        import analyze
        cls.analysis = analyze.analyze(str(cls.video), str(cls.words))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_analysis_finds_speakers_laugh_and_silences(self):
        a = self.analysis
        self.assertEqual(a["speakers"]["source"], "channels")
        self.assertEqual(a["speakers"]["names"], ["S1", "S2"])
        self.assertEqual(len(a["audio"]["nonverbal"]), 1, "the wordless laugh is one nonverbal event")
        self.assertTrue(all(w.get("speaker") for w in a["words"]))
        self.assertGreater(len(a["audio"]["silences"]), 3)

    def test_plan_auto_cuts_and_suggestions(self):
        r = plan_longform.plan(self.analysis)
        applied = {c["reason"] for c in r["cuts"] if c["applied"]}
        suggested = {c["reason"] for c in r["cuts"] if not c["applied"]}
        self.assertTrue({"head", "tail", "pause", "hesitation"} <= applied)
        self.assertTrue({"stutter", "retake"} <= suggested)
        self.assertEqual(sum(c["reason"] == "retake" for c in r["cuts"]), 1, "one restart, one retake")
        um = next(w for w in self.analysis["words"] if w["text"] == "um")
        hes = next(c for c in r["cuts"] if c["reason"] == "hesitation")
        self.assertLessEqual(hes["start"], um["start"])
        self.assertGreaterEqual(hes["end"], um["end"], "a hesitation cut swallows the whole word")

    def test_plan_never_cuts_into_the_laugh(self):
        laugh = self.analysis["audio"]["nonverbal"][0]
        r = plan_longform.plan(self.analysis, apply={"stutter", "retake", "filler"})
        for a, b in lf_common.merge_ranges([[c["start"], c["end"]] for c in r["cuts"] if c["applied"]]):
            self.assertFalse(a < laugh["end"] and b > laugh["start"], (a, b, laugh))
        self.assertTrue(any(c["reason"] == "pause" and c["end"] <= laugh["start"] for c in r["cuts"] if c["applied"]),
                        "the air before the laugh is still tightened")

    def test_find_clips_ranks_whole_sentences(self):
        res = find_clips.find(self.analysis, 8, 25, 4)
        self.assertTrue(res["clips"])
        starts = {s["start"] for s in self.analysis["sentences"]}
        ends = {s["end"] for s in self.analysis["sentences"]}
        for c in res["clips"]:
            self.assertIn(c["start"], {round(s, 2) for s in starts})
            self.assertIn(c["end"], {round(e, 2) for e in ends})

    def test_render_short_follow_and_long(self):
        import render
        an = self.root / "analysis.json"
        lf_common.write_json(an, self.analysis)
        plan_dir = self.root / "plan"
        res = plan_longform.plan(self.analysis)
        plan_longform.write_outputs(self.analysis, res, plan_dir)
        out_long = self.root / "ep-v1.mp4"
        ns = type("A", (), {})()
        ns.source, ns.plan, ns.out, ns.loudness, ns.fps = str(self.video), str(plan_dir), str(out_long), -14.0, "source"
        render.cmd_longform(ns)
        receipt = json.loads(out_long.with_suffix(".receipt.json").read_text())
        manifest = json.loads(out_long.with_suffix(".qa-manifest.json").read_text())
        self.assertAlmostEqual(manifest["expectedDuration"], receipt["duration"], places=2)
        self.assertLess(abs(receipt["loudnessOut"]["lufs"] + 14), 1.0)

        out = self.root / "clip-v1.mp4"
        ns = type("A", (), {})()
        ns.__dict__.update(source=str(self.video), analysis=str(an), start=1.5, end=22.1, out=str(out),
                           layout="follow", boxes="S1=0,0,0.5,1;S2=0.5,0,1,1", zoom=1.0, hold=1.2,
                           no_tighten=False, max_pause=None, apply="retake,stutter", force_cut=None,
                           force_keep=None, loudness=-14.0, fps="30")
        render.cmd_short(ns)
        info = lf_common.probe(out)
        self.assertEqual((info["width"], info["height"]), (1080, 1920))
        rc = json.loads(out.with_suffix(".receipt.json").read_text())
        self.assertGreaterEqual(len(rc["switches"]), 2, "the crop follows both speakers")
        self.assertTrue(json.loads(out.with_suffix(".reel-spec.json").read_text())["captions"])
        with self.assertRaises(ValueError):
            render.cmd_short(ns)  # delivered files are never overwritten


if __name__ == "__main__":
    unittest.main()
