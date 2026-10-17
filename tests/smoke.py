"""The web app end to end: start it on a spare port with a throwaway library, upload
the 2016 song, make a map through the page's own API, and print what comes back.

    .venv/bin/python tests/smoke.py
"""
import glob, json, os, shutil, subprocess, sys, time, urllib.parse, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
LIBRARY = os.path.join(ROOT, "tests", "out", "library")
env = dict(os.environ, MAPMAKER_LIBRARY=LIBRARY)
srv = subprocess.Popen([sys.executable, "-m", "mapmaker", "--port", "8791", "--no-browser"], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
base = "http://127.0.0.1:8791"
def call(path, data=None, method=None):
    req = urllib.request.Request(base + path, data=data, method=method)
    with urllib.request.urlopen(req, timeout=60) as res:
        body = res.read()
    return json.loads(body) if body[:1] in b"{[" else body
try:
    for _ in range(60):
        try:
            state = call("/api/state"); break
        except Exception:
            time.sleep(0.5)
    print("styles offered:", [s["id"] for s in state["styles"]])
    page = call("/")
    print("page bytes:", len(page), "| mentions finer snaps:", b"s.finer" in page)
    name = "d4vd - 2016 (Official Lyric Video) [IqjeqpkQEQ8].opus"
    path = next((p for where in ("songs", os.path.join("library", "songs"))
                 for p in sorted(glob.glob(os.path.join(where, "**", glob.escape(name)), recursive=True))), name)
    song = call("/api/songs?name=" + urllib.parse.quote(name), open(path, "rb").read(), "POST")
    print("song:", song.get("id"), song.get("title"))
    job = call("/api/jobs", json.dumps({"song": song["id"], "stars": 4.5, "keys": 4, "style": "coordination"}).encode(), "POST")
    for _ in range(240):
        st = call("/api/state")
        j = next(x for x in st["jobs"] if x["id"] == job["id"])
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    print("job:", j["status"], j.get("error") or {k: j["stats"][k] for k in ("stars", "objects", "holds", "grid", "finer", "heard", "bpm", "offset")})
    chart = call(f"/api/jobs/{job['id']}/chart")
    print("chart notes:", len(chart["notes"]), "first", chart["notes"][0])
    osz = call(f"/api/jobs/{job['id']}/osz")
    print("osz bytes:", len(osz))
    ranked = call(f"/api/songs/{song['id']}/styles?stars=4.5&keys=4")
    print("styles for it:", [r["style"] for r in ranked["styles"]], "|", ranked["why"])
    job = call("/api/jobs", json.dumps({"song": song["id"], "options": {"stars": 4.5, "style": "auto"}}).encode(), "POST")
    for _ in range(240):
        j = next(x for x in call("/api/state")["jobs"] if x["id"] == job["id"])
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    print("recommended job:", j["status"], j.get("error") or (j["options"]["style"], j["stats"]["style"], j["stats"]["picked"]["why"]))
finally:
    srv.terminate()
    print("server log tail:", srv.communicate(timeout=10)[0].decode()[-300:])
    shutil.rmtree(os.path.join(ROOT, "tests", "out"), ignore_errors=True)
