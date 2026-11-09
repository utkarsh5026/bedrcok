"""Export the animated tour on the docs site as an MP4 (``docs/assets/kbforge-tour.mp4``).

Why a separate file: the tour on the GitHub Pages guide is drawn live in the browser by
``docs/assets/demo.js`` from the screenshots in ``docs/assets/screens/``. That's ideal on the
site, but people also want something they can drop into Slack, an email or a slide deck.

How it works:

1. Opens ``docs/demo.html?record`` in headless Chromium at 1280×720 (rendered at 1.5× for a
   1920×1080 video). ``?record`` hides the player controls.
2. The tour is a pure function of time, so instead of recording in real time (which drops
   frames), it calls ``kbDemo.renderAt(t)`` for every frame and takes a screenshot.
3. Pipes the frames into ``ffmpeg``, which must be on your PATH, to make an H.264 MP4 that
   plays everywhere.

Run ``scripts/docs_screenshots.py`` first if the screenshots changed, then:

    uv run --group docs python scripts/docs_video.py
    uv run --group docs python scripts/docs_video.py --chromium /path/to/chrome --fps 30
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "demo.html"
OUT = ROOT / "docs" / "assets" / "kbforge-tour.mp4"


def ffmpeg_command(fps: int, out: Path) -> list:
    """The ffmpeg call that turns a stream of JPEG frames on stdin into a web-friendly MP4."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg isn't installed. Install it (e.g. `brew install ffmpeg` or `apt install ffmpeg`) and retry.")
    return [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "image2pipe",
        "-framerate",
        str(fps),
        "-c:v",
        "mjpeg",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",  # the pixel format every browser and phone can play
        "-movflags",
        "+faststart",  # put the index first so the video starts playing before it's fully downloaded
        str(out),
    ]


def record(fps: int, scale: float, chromium: Optional[str], out: Path) -> None:
    """Step through the tour frame by frame and encode it."""
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium) if chromium else p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=scale)
        page.goto(PAGE.as_uri() + "?record")
        page.wait_for_function("window.kbDemo !== undefined")
        page.evaluate("async () => { kbDemo.pause(); await kbDemo.ready; await document.fonts.ready; }")
        duration = page.evaluate("kbDemo.duration")
        frames = int(duration * fps)
        screen = page.locator(".screen")
        print(f"Rendering {frames} frames ({duration:.1f}s at {fps} fps)…")

        encoder = subprocess.Popen(ffmpeg_command(fps, out), stdin=subprocess.PIPE)
        assert encoder.stdin is not None
        try:
            for frame in range(frames):
                page.evaluate("t => kbDemo.renderAt(t)", frame / fps)
                encoder.stdin.write(screen.screenshot(type="jpeg", quality=92, animations="disabled"))
                if frame % (fps * 5) == 0:
                    print(f"  {frame / fps:5.1f}s")
        finally:
            encoder.stdin.close()
            encoder.wait()
            browser.close()
    if encoder.returncode != 0:
        sys.exit("ffmpeg failed; see its message above.")
    print(f"Saved {out.relative_to(ROOT)} ({out.stat().st_size / 1_000_000:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the docs tour as an MP4.")
    parser.add_argument("--chromium", help="path to a Chromium binary (default: Playwright's own)")
    parser.add_argument("--fps", type=int, default=30, help="frames per second (default 30)")
    parser.add_argument("--scale", type=float, default=1.5, help="1.5 gives 1920×1080 (default), 1 gives 1280×720")
    parser.add_argument("--out", type=Path, default=OUT, help=f"output file (default {OUT.relative_to(ROOT)})")
    args = parser.parse_args()
    record(args.fps, args.scale, args.chromium, args.out.resolve())


if __name__ == "__main__":
    main()
