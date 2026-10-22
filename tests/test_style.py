"""Presets, colors, fonts and layout — no ffmpeg, no models."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from karaoke import render as rd  # noqa: E402

FONTS = rd.FONTS


def W(text, start=0.0, step=0.3):
    return [{"w": x, "s": round(start + i * step, 3), "e": round(start + (i + 1) * step, 3)}
            for i, x in enumerate(text.split())]


class Colors(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(rd.color("#FFC400"), (255, 196, 0, 255))
        self.assertEqual(rd.color("#ffc40080"), (255, 196, 0, 128))
        self.assertEqual(rd.color("#fff"), (255, 255, 255, 255))
        self.assertEqual(rd.color("white"), (255, 255, 255, 255))
        self.assertEqual(rd.color("rgb(255,0,128)"), (255, 0, 128, 255))
        self.assertEqual(rd.color("255, 80, 0"), (255, 80, 0, 255))
        self.assertEqual(rd.color([0, 0, 0, 190]), (0, 0, 0, 190))

    def test_bad(self):
        for bad in ("notacolor", [1, 2], [300, 0, 0], "#12345", {"r": 1}, None):
            with self.assertRaises(rd.StyleError, msg=repr(bad)):
                rd.color(bad)


class Presets(unittest.TestCase):
    def test_builtin_presets_load(self):
        self.assertGreaterEqual(set(rd.presets()), {"amber", "bold", "clean"})
        for name in rd.presets():
            p = rd.load_preset(name)
            self.assertEqual(p["_warnings"], [], name)
            self.assertEqual(len(p["accent"]), 4)

    def test_minimal_custom_preset_gets_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mine.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"accent": "#00FF88", "typo_key": 1, "y": 0.8}, fh)
            p = rd.load_preset(path)
        self.assertEqual(p["name"], "mine")
        self.assertEqual(p["accent"], (0, 255, 136, 255))
        self.assertEqual(p["y"], {"vertical": 0.8, "landscape": 0.8})
        self.assertEqual(p["size"], rd.DEFAULTS["size"])
        self.assertTrue(any("typo_key" in w for w in p["_warnings"]))

    def test_overrides_win_and_plate_none(self):
        p = rd.load_preset("amber", {"accent": "red", "plate": False, "size": 50, "font": None})
        self.assertEqual(p["accent"], (255, 0, 0, 255))
        self.assertIsNone(p["plate"])
        self.assertEqual(p["size"], 50)
        self.assertEqual(p["font"], "Archivo-Variable.ttf")

    def test_invalid_values(self):
        for bad in ({"max_width": 3}, {"size": -1}, {"size": "big"}, {"y": "low"}, {"axes": "bold"}):
            with self.assertRaises(rd.StyleError, msg=repr(bad)):
                rd.load_preset("amber", bad)
        with self.assertRaises(FileNotFoundError):
            rd.load_preset("does-not-exist")

    def test_example_preset(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "my-style.json")
        p = rd.load_preset(path)
        self.assertEqual(p["_warnings"], [])
        self.assertEqual(rd.font_for(p, 40).getname()[0], "Anton")

    def test_broken_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{oops")
            with self.assertRaises(rd.StyleError):
                rd.load_preset(path)


class Fonts(unittest.TestCase):
    def test_bundled_by_name_and_path(self):
        self.assertTrue(rd.resolve_font("Anton").endswith("Anton-Regular.ttf"))
        self.assertEqual(os.path.basename(rd.resolve_font("anton-regular.ttf")).lower(), "anton-regular.ttf")
        full = os.path.join(FONTS, "Archivo-Variable.ttf")
        self.assertEqual(rd.resolve_font(full), full)

    def test_relative_to_preset_folder(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "fonts"))
            with open(os.path.join(FONTS, "Anton-Regular.ttf"), "rb") as src, \
                    open(os.path.join(d, "fonts", "My Font.ttf"), "wb") as dst:
                dst.write(src.read())
            path = os.path.join(d, "p.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"font": "fonts/My Font.ttf"}, fh)
            p = rd.load_preset(path)
            self.assertEqual(rd.font_for(p, 50).getname()[0], "Anton")

    def test_unknown_font(self):
        with self.assertRaises(rd.StyleError):
            rd.resolve_font("Definitely Not A Font 12345")

    def test_variable_axes_list_dict_and_clamp(self):
        a = rd.font_for(rd.load_preset("amber", {"axes": [400, 100]}), 40)
        b = rd.font_for(rd.load_preset("amber", {"axes": {"wght": 400, "wdth": 100}}), 40)
        self.assertEqual(a.getlength("Hello"), b.getlength("Hello"))
        heavy = rd.font_for(rd.load_preset("amber", {"axes": {"wght": 5000}}), 40)      # clamped to the font's maximum
        self.assertGreater(heavy.getlength("Hello"), a.getlength("Hello") * 0.9)
        with self.assertRaises(rd.StyleError):
            rd.font_for(rd.load_preset("amber", {"axes": {"zzzz": 1}}), 40)

    def test_static_font_ignores_axes(self):
        p = rd.load_preset("bold", {"axes": {"wght": 400}})
        self.assertEqual(rd.font_for(p, 40).getname()[0], "Anton")
        self.assertFalse(rd.is_variable(p))

    def test_missing_glyphs(self):
        f = rd.font_for(rd.load_preset("amber"), 40)
        self.assertEqual(rd.missing_glyphs(f, "Grüße, Øre, ŁÓDŹ!"), [])
        self.assertIn("П", rd.missing_glyphs(f, "Привет"))

    def test_fallback_font_for_other_scripts(self):
        p, miss, fb = rd.cover(rd.load_preset("amber"), "Привет мир")
        self.assertTrue(miss)
        if fb is None:
            self.skipTest("no wide-coverage system font on this machine")
        self.assertEqual(rd.missing_glyphs(rd.font_for(p, 40), "Привет мир"), [])


class Layout(unittest.TestCase):
    def test_long_word_shrinks_to_fit(self):
        for w, h in ((1080, 1920), (320, 568), (1920, 1080)):
            pt = rd.Painter(rd.load_preset("bold"), w, h)
            lines = pt.lines(W("Donaudampfschifffahrtsgesellschaftskapitän"))
            blk = pt.block(lines, 0)
            box = blk.getbbox()
            assert box is not None
            self.assertGreaterEqual(box[0], 0)
            self.assertLessEqual(box[2], w)
            if pt._width(lines[0][0]["w"].upper(), pt.font) > pt.safe_w:
                self.assertLess(pt.fit(lines).size, pt.size)

    def test_scales_with_short_side(self):
        small = rd.Painter(rd.load_preset("amber"), 540, 960)
        big = rd.Painter(rd.load_preset("amber"), 2160, 3840)
        self.assertEqual(big.size, small.size * 4)

    def test_position(self):
        p = rd.load_preset("amber")
        self.assertEqual(rd.position_y(p, 1080, 1920), int(0.72 * 1920))
        self.assertEqual(rd.position_y(p, 1920, 1080), int(0.9 * 1080))
        self.assertEqual(rd.position_y(p, 1080, 1920, 1500), 1500)
        self.assertEqual(rd.Painter(p, 1080, 1920, 5000).bottom, 1920)        # clamped into the frame

    def test_timeline_states(self):
        tl = rd.Timeline(W("one two three four five six", start=1.0), rd.load_preset("bold"), 360, 640)
        self.assertIsNone(tl.state(0.0)[0])
        key, _ = tl.state(1.35)
        assert key is not None
        self.assertEqual(key[:2], (0, 1))
        self.assertIsNone(tl.state(10.0)[0])
        self.assertEqual(tl.image(key).size[0], 360)


if __name__ == "__main__":
    unittest.main()
