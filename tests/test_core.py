"""Fast tests without speech models: word fixes, caption groups, comparison, glossary, decisions.

    python -m pytest            (or: python -m unittest discover -s tests)
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from karaoke import words as wd  # noqa: E402
from karaoke_agent import compare, config  # noqa: E402
from karaoke_agent import suggestions as sg


def W(text, start=0.0, step=0.3, gap=0.0):
    out, t = [], start
    for x in text.split():
        out.append({"w": x, "s": round(t, 3), "e": round(t + step, 3)})
        t += step + gap
    return out


class Words(unittest.TestCase):
    def test_replace_keeps_outer_times(self):
        ws = W("it ran on a real humanoid Aptronics Apollo")
        out, info = wd.replace(ws, "Aptronics", "Apptronik's")
        self.assertEqual(out[6]["w"], "Apptronik's")
        self.assertEqual((out[6]["s"], out[6]["e"]), (ws[6]["s"], ws[6]["e"]))

    def test_replace_changes_word_count(self):
        ws = W("we met at the corn hole stand today")
        out, _ = wd.replace(ws, "the corn hole stand", "the cornhole booth over there")
        self.assertEqual([w["w"] for w in out], ["we", "met", "at", "the", "cornhole", "booth", "over", "there", "today"])
        self.assertEqual(out[3]["s"], ws[3]["s"])
        self.assertEqual(out[7]["e"], ws[6]["e"])

    def test_replace_missing_raises(self):
        with self.assertRaises(KeyError):
            wd.replace(W("one two three"), "four", "4")

    def test_apply_text_from_review(self):
        ws = W("watched a person throw map the scene in 3D")
        out, ch = wd.apply_text(ws, "watched a person throw, mapped the scene in 3D.")
        self.assertEqual([w["w"] for w in out][3:5], ["throw,", "mapped"])
        self.assertEqual(out[4]["s"], ws[4]["s"])
        self.assertEqual(len(out), len(ws))
        self.assertTrue(any(c["new"] == "mapped" for c in ch))

    def test_apply_text_insert_and_delete(self):
        ws = W("one two three four", gap=0.2)
        out, _ = wd.apply_text(ws, "one two and a half three")
        self.assertEqual([w["w"] for w in out], ["one", "two", "and", "a", "half", "three"])
        self.assertTrue(all(a["e"] <= b["s"] + 1e-9 for a, b in zip(out, out[1:])))

    def test_chunk_never_ends_on_leaning_word(self):
        groups = wd.chunk(W("Seven in a row! Gemini watched the person throw"), max_words=3)
        for g in groups:
            if len(g) > 1:
                self.assertNotIn(wd.norm(g[-1]["w"]), wd.LEANING, [x["w"] for x in g])

    def test_chunk_breaks_on_pause(self):
        ws = W("first second", gap=0.0) + W("third", start=3.0)
        self.assertEqual([[w["w"] for w in g] for g in wd.chunk(ws, 3)], [["first", "second"], ["third"]])

    def test_srt(self):
        with tempfile.TemporaryDirectory() as d:
            n = wd.write_srt(W("one two three four five six seven"), os.path.join(d, "a.srt"))
            with open(os.path.join(d, "a.srt"), encoding="utf-8") as fh:
                text = fh.read()
        self.assertGreaterEqual(n, 1)
        self.assertIn("00:00:00,000 -->", text)
        self.assertEqual(wd.srt_ts(3661.9996), "01:01:02,000")

    def test_save_load_roundtrip_unicode(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.words.json")
            wd.save(p, {"words": W("Grüße Привет Γειά")})
            self.assertEqual(wd.load(p)["words"][1]["w"], "Привет")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("[]")
            with self.assertRaises(ValueError):
                wd.load(p)


class Agent(unittest.TestCase):
    def test_disagreements(self):
        ws = W("Gemini 4 Argon watched a person throw, map the scene in 3D")
        d = compare.disagreements(ws, "Gemini four Argon watched a person throw, mapped the scene in 3-D")
        kinds = {x["parakeet"]: x["kind"] for x in d}
        self.assertEqual(kinds.get("map"), "words")
        self.assertEqual(kinds.get("4"), "number")
        self.assertNotIn("3D", kinds)                          # 3D vs 3-D: equal after normalising

    def test_glossary_hit_and_no_false_positive(self):
        ws = W("Google's AI taught a robot cornhole. Then it ran on a real humanoid, Aptronics Apollo. Gemini 4 Argon")
        terms = [{"term": "Apptronik", "note": ""}, {"term": "cornhole", "note": ""}, {"term": "Gemini 4 Argon", "note": ""}]
        hits = compare.glossary_hits(ws, terms)
        self.assertEqual([h["old"] for h in hits], ["Aptronics"])

    def test_glossary_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "g.txt")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("# names\nApptronik (robot maker)\n\nGemini 4 Argon  # model\n")
            self.assertEqual(compare.load_glossary(p), [{"term": "Apptronik", "note": "robot maker"},
                                                        {"term": "Gemini 4 Argon", "note": ""}])

    def test_decide_back_to_front(self):
        ws = W("a map of Aptronics Apollo")
        doc = {"suggestions": []}
        sg.add(doc, [{"i0": 1, "i1": 2, "old": "map", "new": "mapped"},
                     {"i0": 3, "i1": 4, "old": "Aptronics", "new": "Apptronik"}], "test")
        out, applied = sg.decide(ws, doc, accept=[1, 2], edits={2: "Apptronik's"})
        self.assertEqual([w["w"] for w in out], ["a", "mapped", "of", "Apptronik's", "Apollo"])
        self.assertEqual({s["status"] for s in doc["suggestions"]}, {"accepted"})
        self.assertEqual(len(applied), 2)

    def test_add_skips_duplicates_and_noops(self):
        doc = {"suggestions": []}
        items = [{"i0": 0, "i1": 1, "old": "map", "new": "mapped"}, {"i0": 0, "i1": 1, "old": "map", "new": "Mapped"},
                 {"i0": 2, "i1": 3, "old": "the", "new": "The"}]
        self.assertEqual(len(sg.add(doc, items, "test")), 1)

    def test_config_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "karaoke.toml")
            cfg = {"whisper": {"backend": "whisper.cpp", "model": "small", "reason": 'M1 "fast"\\path', "n": 2,
                               "x": None}}
            config.save(cfg, p)
            back = config.load(p)["whisper"]
            self.assertEqual(back["reason"], 'M1 "fast"\\path')
            self.assertNotIn("x", back)


class AgentSetup(unittest.TestCase):
    def test_guide_ships_and_init_never_overwrites(self):
        from karaoke_agent import cli as acli
        self.assertTrue(os.path.isfile(acli.GUIDE))
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(acli.main(["init", d]), 0)
            agents = os.path.join(d, "AGENTS.md")
            skill = os.path.join(d, ".claude", "skills", "karaoke-captions", "SKILL.md")
            with open(skill, encoding="utf-8") as fh:
                self.assertIn("karaoke-agent guide", fh.read())
            with open(os.path.join(d, "CLAUDE.md"), encoding="utf-8") as fh:
                self.assertEqual(fh.read().strip(), "@AGENTS.md")
            with open(agents, "w", encoding="utf-8") as fh:
                fh.write("my own rules")
            self.assertEqual(acli.main(["init", d]), 0)
            with open(agents, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "my own rules")
            self.assertEqual(acli.main(["init", d, "--force"]), 0)
            with open(agents, encoding="utf-8") as fh:
                self.assertIn("karaoke-agent", fh.read())

    def test_sysinfo_finds_models_on_disk(self):
        from karaoke_agent import sysinfo, whisper
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "ggml-small.bin"), "wb") as fh:
                fh.write(b"0" * 1024)
            old = list(whisper.GGML_DIRS)
            whisper.GGML_DIRS[:] = [d]
            try:
                found = [m for m in sysinfo.whisper_models() if m["backend"] == "whisper.cpp"]
                self.assertEqual(whisper.ggml_path("small"), os.path.join(d, "ggml-small.bin"))
            finally:
                whisper.GGML_DIRS[:] = old
        self.assertEqual([m["model"] for m in found], ["small"])
        info = sysinfo.collect()
        for k in ("os", "ram_gb", "gpus", "whisper_installed", "whisper_models", "whisper_options"):
            self.assertIn(k, info)


if __name__ == "__main__":
    unittest.main()
