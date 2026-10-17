"""The drag-and-drop page: a small local web app around make_map().

Everything stays on this machine. The server listens on 127.0.0.1 only and keeps
uploads and finished maps in the `library` folder next to this package."""

import json
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import HANDS, STYLES, MapError, Options, guess_tags, make_map
from .chart import suit_range
from .audio import CACHE, ROOT
from .pipeline import recommend_for, song_key
from .recommend import AUTO, explain

LIBRARY = os.environ.get("MAPMAKER_LIBRARY") or os.path.join(ROOT, "library")
SONGS, MAPS = os.path.join(LIBRARY, "songs"), os.path.join(LIBRARY, "maps")
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "index.html")
MAX_UPLOAD = 400 * 1024 * 1024
# how far along the whole job each stage starts, and how much of it the stage covers
STAGES = {"Reading the audio": (0.0, 0.03), "Separating the instruments": (0.03, 0.52),
          "Listening to drums, voice and bass": (0.55, 0.20), "Measuring the tempo": (0.75, 0.05),
          "Writing the chart": (0.80, 0.17), "Packing the .osz": (0.97, 0.03)}

lock = threading.RLock()
songs, jobs = {}, {}
pending = queue.Queue()


# ------------------------------------------------------------------ storage

def _clean(text, limit=80):
    """A single tidy line: metadata ends up inside the .osu file."""
    return re.sub(r"[\x00-\x1f\x7f]+", " ", str(text or "")).strip()[:limit]


def _save(folder, name, record):
    tmp = os.path.join(folder, name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh)
    os.replace(tmp, os.path.join(folder, name))


def load_library():
    for root, store, name in ((SONGS, songs, "song.json"), (MAPS, jobs, "job.json")):
        os.makedirs(root, exist_ok=True)
        for entry in sorted(os.listdir(root)):
            try:
                with open(os.path.join(root, entry, name), encoding="utf-8") as fh:
                    store[entry] = json.load(fh)
            except (OSError, ValueError):
                continue
    for job in jobs.values():           # anything cut off by a restart can't be resumed
        if job["status"] in ("queued", "running"):
            job.update(status="error", error="Interrupted - the app was closed while this ran.")
    # the style first called "tech" is now "coordination", and "tech" means something
    # else: maps made under the old name are relabelled, once
    stamp = os.path.join(LIBRARY, "format")
    try:
        with open(stamp) as fh:
            done = int(fh.read().strip() or 1)
    except (OSError, ValueError):
        done = 1
    if done < 2:
        for jid, job in jobs.items():
            changed = False
            for part in (job.get("options"), job.get("stats")):
                if part and part.get("style") == "tech":
                    part["style"], changed = "coordination", True
            if changed:
                _save(os.path.join(MAPS, jid), "job.json", job)
        with open(stamp, "w") as fh:
            fh.write("2\n")
    if done < 3:                        # the Holds style has become part of LN
        for jid, job in jobs.items():
            changed = False
            for part in (job.get("options"), job.get("stats")):
                if part and part.get("style") == "holds":
                    part["style"], changed = "ln", True
            if changed:
                _save(os.path.join(MAPS, jid), "job.json", job)
        with open(stamp, "w") as fh:
            fh.write("3\n")


def add_song(name, body):
    name = _clean(os.path.basename(name.replace("\\", "/")), 150) or "song"
    tmp = os.path.join(SONGS, f".upload-{secrets.token_hex(4)}")
    with open(tmp, "wb") as fh:
        fh.write(body)
    sid = song_key(tmp)
    folder = os.path.join(SONGS, sid)
    with lock:
        if sid in songs:                # the same file again: keep the one we have
            os.remove(tmp)
            return songs[sid]
        os.makedirs(folder, exist_ok=True)
        os.replace(tmp, os.path.join(folder, name))
        artist, title = guess_tags(name)
        songs[sid] = {"id": sid, "file": name, "artist": artist, "title": title,
                      "bg": None, "added": time.time(), "size": len(body)}
        _save(folder, "song.json", songs[sid])
        return songs[sid]


def add_job(body):
    sid = body.get("song")
    with lock:
        if sid not in songs:
            raise MapError("That song isn't in the library any more - drop it in again.")
        song = songs[sid]
    o = body.get("options") or {}

    def number(key, default, lo, hi, kind=float):
        value = o.get(key)
        if value in (None, ""):
            return default
        try:
            value = kind(value)
        except (TypeError, ValueError):
            raise MapError(f"'{key}' should be a number.")
        if not lo <= value <= hi:
            raise MapError(f"'{key}' should be between {lo} and {hi}.")
        return value

    keys, style = number("keys", 4, 1, 18, int), o.get("style") or "coordination"
    if keys not in HANDS or (style not in STYLES and style != AUTO):
        raise MapError("Unknown key count or style.")
    options = {
        "stars": number("stars", 4.5, 0.5, 10.0), "keys": keys, "style": style,
        "ln": number("ln", None, 0.0, 1.0), "seed": number("seed", 2016, 0, 2**31, int),
        "creator": _clean(o.get("creator")) or Options.creator,
        "title": _clean(o.get("title")) or song["title"],
        "artist": _clean(o.get("artist")) or song["artist"],
        "version": _clean(o.get("version"), 60),
        "bpm": number("bpm", None, 30.0, 400.0), "offset": number("offset", None, 0.0, 6e5),
        "tempo": o.get("tempo") if o.get("tempo") in ("half", "double") else "auto",
        "nudge": number("nudge", 0.0, -500.0, 500.0),
        "od": number("od", 8.0, 0.0, 10.0), "hp": number("hp", 8.0, 0.0, 10.0),
    }
    jid = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
    job = {"id": jid, "song": sid, "options": options, "status": "queued", "stage": "Waiting",
           "progress": 0.0, "created": time.time(), "file": song["file"]}
    with lock:
        songs[sid].update(artist=options["artist"], title=options["title"])
        _save(os.path.join(SONGS, sid), "song.json", songs[sid])
        jobs[jid] = job
        os.makedirs(os.path.join(MAPS, jid), exist_ok=True)
        _save(os.path.join(MAPS, jid), "job.json", job)
    pending.put(jid)
    return job


def worker():
    while True:
        jid = pending.get()
        with lock:
            job = jobs.get(jid)
            song = songs.get(job["song"]) if job else None
            if not job or not song or job["status"] != "queued":
                continue
            job.update(status="running", stage="Reading the audio", started=time.time())
        folder = os.path.join(MAPS, jid)

        def progress(stage, fraction):
            start, span = STAGES.get(stage, (0.0, 0.0))
            with lock:
                job.update(stage=stage, progress=round(start + span * min(1.0, fraction), 3))

        try:
            bg = os.path.join(SONGS, song["id"], song["bg"]) if song.get("bg") else None
            done = make_map(os.path.join(SONGS, song["id"], song["file"]),
                            Options(bg=bg, **job["options"]), folder, progress)
            with open(os.path.join(folder, "chart.json"), "w", encoding="utf-8") as fh:
                json.dump(done["chart"], fh)
            with lock:
                job.update(status="done", stage="Done", progress=1.0, stats=done["stats"],
                           warnings=done["warnings"], meta=done["meta"],
                           osz=os.path.basename(done["osz"]), took=round(time.time() - job["started"], 1))
        except MapError as err:
            with lock:
                job.update(status="error", error=str(err))
        except Exception as err:        # a bug, not the person's fault: say so, and log it
            traceback.print_exc()
            with lock:
                job.update(status="error",
                           error=f"Something broke while charting this ({type(err).__name__}: {err}).")
        with lock:
            if jid in jobs:
                _save(folder, "job.json", job)


def remove_job(jid):
    with lock:
        job = jobs.get(jid)
        if not job or job["status"] == "running":
            return False
        jobs.pop(jid)
    shutil.rmtree(os.path.join(MAPS, jid), ignore_errors=True)
    return True


def remove_song(sid):
    with lock:
        if sid not in songs or any(j["song"] == sid and j["status"] in ("queued", "running")
                                   for j in jobs.values()):
            return False
        mine = [j for j, job in jobs.items() if job["song"] == sid]
    for jid in mine:
        remove_job(jid)
    with lock:
        songs.pop(sid, None)
    shutil.rmtree(os.path.join(SONGS, sid), ignore_errors=True)
    for name in os.listdir(CACHE) if os.path.isdir(CACHE) else []:
        if f"-{sid}." in name:          # stems, analysis, packed audio, preview
            os.remove(os.path.join(CACHE, name))
    return True


def styles_for(sid, query):
    """The styles ranked for a song that has been mapped once (so it has been listened
    to), at the rating, key count and tempo the page has set."""
    def number(key, default, lo, hi):
        try:
            return min(hi, max(lo, float((query.get(key) or [default])[0])))
        except ValueError:
            return default
    keys = int(number("keys", 4, 4, 7))
    tempo = (query.get("tempo") or ["auto"])[0]
    ranking = recommend_for(sid, number("stars", 4.5, 0.5, 10.0), keys if keys in HANDS else 4,
                            tempo if tempo in ("half", "double") else "auto")
    if ranking is None:
        return {"ready": False}
    return {"ready": True, "why": explain(ranking),
            "styles": [{k: r[k] for k in ("style", "score", "why")} for r in ranking]}


RECOMMENDED = {"id": AUTO, "label": "Recommended", "auto": True, "ln": 0.0, "short": False, "weave": False,
               "suits": None, "rows": None, "quick": None,
               "blurb": "Picks the style that suits the song: by its tempo, what its instruments play and how loud it is."}


def state():
    with lock:
        return {"styles": [RECOMMENDED] + [{"id": k, "label": s.label, "blurb": s.blurb, "ln": s.ln,
                            "short": bool(s.short), "weave": bool(s.weave),
                            # how far apart its rows should come, by the rating asked for
                            "suits": {f"{n / 10:.1f}": suit_range(s, n / 10) for n in range(10, 101)}
                            if s.suits else None, "rows": s.rows, "quick": s.quick}
                           for k, s in STYLES.items()],
                "keys": sorted(HANDS), "creator": Options.creator,
                "songs": sorted(songs.values(), key=lambda s: -s["added"]),
                "jobs": sorted(jobs.values(), key=lambda j: -j["created"])}


# --------------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "mapmaker"

    def log_message(self, fmt, *args):
        pass

    # --- plumbing
    def _local(self):
        """Only this machine's own page may talk to the app."""
        host = (self.headers.get("Host") or "").split(":")[0]
        origin = self.headers.get("Origin")
        ok = host in ("127.0.0.1", "localhost")
        if origin:
            ok = ok and urllib.parse.urlparse(origin).hostname in ("127.0.0.1", "localhost")
        return ok

    def _send(self, code, body=b"", kind="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _fail(self, code, message):
        self.close_connection = True    # whatever body came with the request stays unread
        self._send(code, {"error": message})

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD:
            raise MapError("That file is too big (limit 400 MB).")
        return self.rfile.read(length) if length else b""

    def _file(self, path, kind, download=None):
        """Send a file, honouring Range so the browser can seek in audio."""
        size = os.path.getsize(path)
        start, end = 0, size - 1
        match = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range") or "")
        match = match if match and (match.group(1) or match.group(2)) else None
        if match:
            if match.group(1):
                start = int(match.group(1))
                end = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
            else:
                start = max(0, size - int(match.group(2)))
            if start > end or start >= size:
                return self._send(416, b"", kind, {"Content-Range": f"bytes */{size}"})
        self.send_response(206 if match else 200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if match:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            quoted = urllib.parse.quote(download)
            self.send_header("Content-Disposition",
                             f"attachment; filename=\"map.osz\"; filename*=UTF-8''{quoted}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = fh.read(min(65536, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    # --- routes
    def do_GET(self):
        if not self._local():
            return self._fail(403, "local use only")
        url = urllib.parse.urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        if not parts:
            with open(PAGE, "rb") as fh:
                return self._send(200, fh.read(), "text/html; charset=utf-8")
        if parts == ["api", "state"]:
            return self._send(200, state())
        if len(parts) == 4 and parts[:2] == ["api", "jobs"]:
            with lock:
                job = dict(jobs.get(parts[2]) or {})
            folder = os.path.join(MAPS, parts[2])
            if job.get("status") == "done" and parts[3] == "osz":
                return self._file(os.path.join(folder, job["osz"]), "application/octet-stream",
                                  job["osz"])
            if job.get("status") == "done" and parts[3] == "chart":
                return self._file(os.path.join(folder, "chart.json"), "application/json")
        if len(parts) == 4 and parts[:2] == ["api", "songs"] and parts[3] == "styles":
            if re.fullmatch(r"[0-9a-f]{12}", parts[2]):
                return self._send(200, styles_for(parts[2], urllib.parse.parse_qs(url.query)))
        if len(parts) == 4 and parts[:2] == ["api", "songs"] and parts[3] == "audio":
            path = os.path.join(CACHE, f"preview-{parts[2]}.wav")
            if re.fullmatch(r"[0-9a-f]{12}", parts[2]) and os.path.isfile(path):
                return self._file(path, "audio/wav")
        return self._fail(404, "not found")

    do_HEAD = do_GET

    def do_POST(self):
        if not self._local():
            return self._fail(403, "local use only")
        url = urllib.parse.urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        query = urllib.parse.parse_qs(url.query)
        try:
            body = self._body()
            if parts == ["api", "songs"]:
                if not body:
                    raise MapError("That file is empty.")
                return self._send(200, add_song((query.get("name") or ["song"])[0], body))
            if len(parts) == 4 and parts[:2] == ["api", "songs"] and parts[3] == "bg":
                return self._background(parts[2], (query.get("name") or ["bg.jpg"])[0], body)
            if parts == ["api", "jobs"]:
                return self._send(200, add_job(json.loads(body or b"{}")))
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "open":
                return self._open(parts[2])
        except MapError as err:
            return self._fail(400, str(err))
        except ValueError:
            return self._fail(400, "That request didn't make sense.")
        return self._fail(404, "not found")

    def do_DELETE(self):
        if not self._local():
            return self._fail(403, "local use only")
        parts = [p for p in urllib.parse.urlparse(self.path).path.split("/") if p]
        if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
            return self._send(200, {"removed": remove_job(parts[2])})
        if len(parts) == 3 and parts[:2] == ["api", "songs"]:
            return self._send(200, {"removed": remove_song(parts[2])})
        return self._fail(404, "not found")

    def _background(self, sid, name, body):
        ext = os.path.splitext(name)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            raise MapError("Backgrounds should be .jpg or .png.")
        with lock:
            if sid not in songs:
                raise MapError("That song isn't in the library any more.")
            with open(os.path.join(SONGS, sid, "bg" + ext), "wb") as fh:
                fh.write(body)
            songs[sid]["bg"] = "bg" + ext
            _save(os.path.join(SONGS, sid), "song.json", songs[sid])
            return self._send(200, songs[sid])

    def _open(self, jid):
        """Hand the finished .osz to osu! (whatever opens .osz files on this machine)."""
        with lock:
            job = dict(jobs.get(jid) or {})
        if job.get("status") != "done":
            raise MapError("That map isn't ready.")
        path = os.path.join(MAPS, jid, job["osz"])
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif os.name == "nt":
            os.startfile(path)          # noqa: only exists on Windows
        else:
            subprocess.Popen(["xdg-open", path])
        return self._send(200, {"opened": True})


def serve(port=8765, open_browser=True):
    load_library()
    threading.Thread(target=worker, daemon=True).start()
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError:
        sys.exit(f"Port {port} is busy - is mapmaker already running? "
                 f"Open http://127.0.0.1:{port} or pass --port.")
    httpd.daemon_threads = True
    url = f"http://127.0.0.1:{port}"
    print(f"mapmaker is running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.4, webbrowser.open, [url]).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
