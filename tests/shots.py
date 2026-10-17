"""Screenshots of the page, for looking at a change to it.

    tests/browser/.venv/bin/python tests/shots.py [name] [--make style:stars]

Starts the app on a spare port with a copy of the real library, opens it in a headless
browser and saves tests/browser/shots/<name>-*.png: the page, the page with every
option open, the preview, and the page at phone width. The browser lives in
tests/browser (set up with: uv venv tests/browser/.venv; uv pip install playwright;
PLAYWRIGHT_BROWSERS_PATH=tests/browser/browsers playwright install chromium-headless-shell)."""

import os
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = os.path.join(ROOT, "tests", "browser", "browsers")
from playwright.sync_api import sync_playwright  # noqa: E402

PORT = 8792


def main(argv):
    make = argv[argv.index("--make") + 1] if "--make" in argv else None
    argv = [a for a in argv if a != "--make" and a != make]
    name = argv[0] if argv else "page"
    shots = os.path.join(ROOT, "tests", "browser", "shots")
    os.makedirs(shots, exist_ok=True)
    library = os.path.join(ROOT, "tests", "out", "library")
    shutil.rmtree(library, ignore_errors=True)
    shutil.copytree(os.path.join(ROOT, "library"), library)
    stray = os.path.join(library, "songs", "song.json")     # a song whose folder was emptied out
    if os.path.isfile(stray):
        folder = os.path.join(library, "songs", "af84addce66d")
        os.makedirs(folder, exist_ok=True)
        for entry in os.listdir(os.path.join(library, "songs")):
            if entry.startswith("d4vd") or entry == "song.json":
                if os.path.exists(os.path.join(folder, entry)):     # the song was added again since
                    os.remove(os.path.join(library, "songs", entry))
                else:
                    shutil.move(os.path.join(library, "songs", entry), folder)
    server = subprocess.Popen([os.path.join(ROOT, ".venv", "bin", "python"), "-m", "mapmaker",
                               "--port", str(PORT), "--no-browser"], cwd=ROOT,
                              env=dict(os.environ, MAPMAKER_LIBRARY=library),
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(80):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/state", timeout=2)
                break
            except OSError:
                time.sleep(0.25)
        if make:                        # a fresh map of the newest song first, to see its card
            import json
            state = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/state"))
            song = sorted(state["songs"], key=lambda s: -s["added"])[0]["id"]
            style, stars = make.split(":")
            body = json.dumps({"song": song, "options": {"style": style, "stars": float(stars)}}).encode()
            job = json.load(urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{PORT}/api/jobs", body, method="POST")))
            for _ in range(240):
                state = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/state"))
                if next(j for j in state["jobs"] if j["id"] == job["id"])["status"] in ("done", "error"):
                    break
                time.sleep(0.5)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
            errors = []
            page.on("pageerror", lambda err: errors.append(str(err)))
            page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
            page.goto(f"http://127.0.0.1:{PORT}/")
            page.wait_for_selector(".map, .empty")
            page.wait_for_timeout(600)
            page.screenshot(path=os.path.join(shots, f"{name}-1-page.png"), full_page=True)
            if page.locator(".song").count():
                page.locator(".song").first.click()
                page.wait_for_timeout(300)
            page.locator("#more summary").click()
            page.locator("#starInfo summary").click()
            page.wait_for_timeout(1500)
            page.screenshot(path=os.path.join(shots, f"{name}-2-options.png"), full_page=True)
            print("what the star rating changes:", page.locator("#starText").inner_text()[:300].replace("\n", " | "))
            print("under the styles:", page.locator("#tempoNote").inner_text())
            preview = page.get_by_text("Preview", exact=True)
            if preview.count():
                preview.first.click()
                page.wait_for_timeout(1200)
                page.screenshot(path=os.path.join(shots, f"{name}-3-preview.png"))
            phone = browser.new_page(viewport={"width": 400, "height": 860}, device_scale_factor=2)
            phone.goto(f"http://127.0.0.1:{PORT}/")
            phone.wait_for_selector(".map, .empty")
            phone.wait_for_timeout(500)
            phone.screenshot(path=os.path.join(shots, f"{name}-4-phone.png"), full_page=True)
            width = phone.evaluate("document.documentElement.scrollWidth")
            browser.close()
        print(f"page errors: {errors or 'none'}; phone page {width} px wide (400 shown)")
    finally:
        server.terminate()
        shutil.rmtree(os.path.join(ROOT, "tests", "out"), ignore_errors=True)
    print("saved to", shots)


if __name__ == "__main__":
    main(sys.argv[1:])
