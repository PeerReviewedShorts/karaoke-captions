"""Render + burn on real (tiny, generated) videos: every aspect ratio, phone rotation, non-square pixels, odd sizes,
containers and audio codecs that cannot go into .mp4 as they are, video without sound, audio without video.
Needs ffmpeg/ffprobe on PATH; no speech models."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from karaoke import cli, media  # noqa: E402
from karaoke import render as rd
from karaoke import words as wd

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def make(d, name, size="320x568", rate="30", dur=1.6, vcodec=("-c:v", "libx264", "-pix_fmt", "yuv420p"),
         acodec=("-c:a", "aac"), vf=None, audio=True):
    path = os.path.join(d, name)
    args = ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}:duration={dur}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}"]
    if vf:
        args += ["-vf", vf]
    args += list(vcodec) + (list(acodec) if audio else []) + ["-shortest", path]
    ff(*args)
    return path


def rotated(d, src, name="rot.mp4"):
    """A copy of src with a 90 degree display rotation flag, like a phone video (needs ffmpeg >= 6.1)."""
    out = os.path.join(d, name)
    try:
        ff("-display_rotation", "90", "-i", src, "-c", "copy", out)
    except subprocess.CalledProcessError:
        raise unittest.SkipTest("this ffmpeg cannot write rotation flags (-display_rotation)") from None
    return out


def words_for(clip, text="one two three four five six"):
    p = wd.paths(clip)
    ws = [{"w": x, "s": round(0.1 + i * 0.22, 3), "e": round(0.3 + i * 0.22, 3)} for i, x in enumerate(text.split())]
    wd.save(p["words"], {"words": ws})
    return p


def stream(path, kind="v"):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", f"{kind}:0", "-show_entries",
                          "stream=codec_name,width,height,sample_aspect_ratio", "-of", "json", path],
                         capture_output=True, text=True).stdout
    return (json.loads(out).get("streams") or [None])[0]


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg")
class Probe(unittest.TestCase):
    def test_rotation_anamorphic_odd(self):
        with tempfile.TemporaryDirectory() as d:
            rot = rotated(d, make(d, "land.mp4", size="640x360"))
            i = media.probe(rot)
            self.assertEqual((i["width"], i["height"], i["rotation"] in (90, 270)), (360, 640, True))
            ana = make(d, "ana.mp4", size="480x360", vf="setsar=4/3")
            self.assertEqual((media.probe(ana)["width"], media.probe(ana)["sar"]), (640, round(4 / 3, 4)))
            odd = make(d, "odd.mkv", size="320x568", vf="scale=321:567,setsar=1", vcodec=("-c:v", "ffv1"),
                       acodec=("-c:a", "pcm_s16le"))
            self.assertEqual((media.probe(odd)["width"], media.probe(odd)["height"]), (321, 567))

    def test_audio_only(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.m4a")
            ff("-f", "lavfi", "-i", "sine=duration=1", a)
            i = media.probe(a)
            self.assertFalse(i["has_video"])
            self.assertTrue(i["has_audio"])
            self.assertAlmostEqual(i["duration"], 1.0, delta=0.1)


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg")
class RenderBurn(unittest.TestCase):
    def check(self, clip, want_w, want_h, preset="amber", extra=()):
        p = words_for(clip)
        self.assertEqual(cli.main(["render", clip, "--preset", preset, *extra]), 0)
        self.assertEqual(cli.main(["burn", clip]), 0)
        v = stream(p["out"])
        assert v is not None
        self.assertEqual((v["width"], v["height"]), (want_w, want_h), clip)
        self.assertIn(v.get("sample_aspect_ratio", "1:1"), ("1:1", "N/A", None))
        return p

    def test_aspect_ratios_and_presets(self):
        with tempfile.TemporaryDirectory() as d:
            for size in ("320x568", "568x320", "400x400", "640x272"):
                w, h = map(int, size.split("x"))
                clip = make(d, f"c{size}.mp4", size=size)
                for preset in rd.presets():
                    self.check(clip, w, h, preset)

    def test_phone_rotation(self):
        with tempfile.TemporaryDirectory() as d:
            self.check(rotated(d, make(d, "land.mp4", size="640x360")), 360, 640)

    def test_anamorphic_becomes_square(self):
        with tempfile.TemporaryDirectory() as d:
            self.check(make(d, "ana.mp4", size="480x360", vf="setsar=4/3"), 640, 360)

    def test_odd_size_padded_even(self):
        with tempfile.TemporaryDirectory() as d:
            odd = make(d, "odd.mkv", size="320x568", vf="scale=321:567,setsar=1", vcodec=("-c:v", "ffv1"),
                       acodec=("-c:a", "pcm_s16le"))
            self.check(odd, 322, 568)

    def test_audio_codecs_end_up_playable_in_mp4(self):
        with tempfile.TemporaryDirectory() as d:
            cases = [("pcm.mov", ("-c:a", "pcm_s16le"), "aac"), ("aac.mp4", ("-c:a", "aac"), "aac")]
            if media.has_encoder("libopus") and media.has_encoder("libvpx-vp9"):
                cases.append(("vp9.webm", ("-c:a", "libopus"), "aac"))
            for name, acodec, want in cases:
                vc = (("-c:v", "libvpx-vp9", "-b:v", "300k") if name.endswith(".webm")
                      else ("-c:v", "libx264", "-pix_fmt", "yuv420p"))
                p = self.check(make(d, name, vcodec=vc, acodec=acodec), 320, 568)
                a = stream(p["out"], "a")
                assert a is not None
                self.assertEqual(a["codec_name"], want, name)

    @unittest.skipUnless(HAVE_FFMPEG and media.has_encoder("libx265"), "needs libx265")
    def test_hdr_stays_hdr(self):
        with tempfile.TemporaryDirectory() as d:
            hdr = make(d, "hdr.mov", vf="setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc",
                       vcodec=("-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "log-level=error"))
            self.assertTrue(media.probe(hdr)["hdr"])
            p = self.check(hdr, 320, 568)
            out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                  "stream=codec_name,pix_fmt,color_transfer", "-of", "json", p["out"]],
                                 capture_output=True, text=True).stdout
            v = json.loads(out)["streams"][0]
            self.assertEqual((v["codec_name"], v["pix_fmt"], v.get("color_transfer")),
                             ("hevc", "yuv420p10le", "arib-std-b67"))

    def test_no_audio_and_high_frame_rate(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.check(make(d, "silent60.mp4", rate="60", audio=False), 320, 568)
            self.assertIsNone(stream(p["out"], "a"))

    def test_band_overlay_is_small_and_full_frame_option(self):
        with tempfile.TemporaryDirectory() as d:
            clip = make(d, "c.mp4", size="360x640")
            p = words_for(clip)
            self.assertEqual(cli.main(["render", clip]), 0)
            w, h, y = cli.overlay_info(p["mov"])
            self.assertEqual(w, 360)
            self.assertLess(h or 0, 320)
            self.assertGreater(y, 0)
            self.assertEqual(cli.main(["render", clip, "--full-frame"]), 0)
            self.assertEqual(cli.overlay_info(p["mov"]), (360, 640, 0))

    def test_stale_overlay_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = make(d, "a.mp4", size="360x640"), make(d, "b.mp4", size="640x360")
            pa, pb = words_for(a), words_for(b)
            self.assertEqual(cli.main(["render", b]), 0)
            shutil.copy(pb["mov"], pa["mov"])
            self.assertEqual(cli.main(["burn", a]), 1)

    def test_style_options_and_preview(self):
        with tempfile.TemporaryDirectory() as d:
            clip = make(d, "c.mp4", size="360x640")
            words_for(clip, "Grüße aus Köln, Donaudampfschifffahrtsgesellschaftskapitän!")
            self.assertEqual(cli.main(["preview", clip, "--font", "Anton", "--accent", "#00FF88", "--plate", "none",
                                       "--uppercase", "--size", "90", "--at", "0.35"]), 0)
            png = os.path.join(d, "c.preview.png")
            self.assertTrue(os.path.getsize(png) > 0)
            self.assertEqual(cli.main(["render", clip, "--color", "yellow", "--outline-width", "0", "--y", "0.5"]), 0)
            self.assertEqual(cli.main(["render", clip, "--accent", "notacolor"]), 1)
            self.assertEqual(cli.main(["render", clip, "--font", "No Such Font 123"]), 1)

    def test_audio_only_render_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.m4a")
            ff("-f", "lavfi", "-i", "sine=duration=1", a)
            words_for(a)
            self.assertEqual(cli.main(["render", a]), 1)

    def test_burn_never_overwrites_the_clip(self):
        with tempfile.TemporaryDirectory() as d:
            clip = make(d, "c.mp4")
            words_for(clip)
            self.assertEqual(cli.main(["render", clip]), 0)
            self.assertEqual(cli.main(["burn", clip, "-o", clip]), 1)

    def test_check_json(self):
        with tempfile.TemporaryDirectory() as d:
            clip = make(d, "c.mp4")
            p = words_for(clip)
            code = cli.main(["check", clip])
            self.assertIn(code, (0, 2))
            with open(p["check"], encoding="utf-8") as fh:
                self.assertIn("metrics", json.load(fh))


class DeviceFallback(unittest.TestCase):
    def test_gpu_failure_falls_back_to_cpu(self):
        from karaoke import parakeet
        calls, logs = [], []
        orig = (parakeet.pick_device, parakeet._run, media.read_audio)

        def fake_run(wav, model, device):
            calls.append(device)
            if device != "cpu":
                raise RuntimeError("MPS backend out of memory")
            return [{"w": "hi", "s": 0.0, "e": 0.2}]
        try:
            parakeet.pick_device = lambda device="auto": "mps"
            parakeet._run = fake_run
            media.read_audio = lambda path, sr=16000: __import__("numpy").zeros(16000, dtype="float32")
            words, prov = parakeet.transcribe("x.wav", device="auto", log=logs.append)
        finally:
            parakeet.pick_device, parakeet._run, media.read_audio = orig
        self.assertEqual(calls, ["mps", "cpu"])
        self.assertEqual(prov["device"], "cpu")
        self.assertTrue(any("falling back" in x for x in logs))


if __name__ == "__main__":
    unittest.main()
