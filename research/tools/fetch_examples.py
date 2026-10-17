"""Download the example charts the styles are modelled on (chart text only, no audio)."""
import json, os, re, ssl, sys, time, urllib.request

import certifi

TOOLS = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(TOOLS)
OUT = os.path.join(RESEARCH, "examples")
# category -> [(beatmapset id, difficulty id or None for "hardest 4K")]
WANTED = {
    "stream": [(1957138, 4054401), (468353, 1001780), (409429, 888774)],
    "chordjack": [(2043161, 4855703), (2182334, 4610936)],
    "ln": [(2367080, 5105790), (2561939, 5695377)],
    "speedjack": [(1881122, 3872878), (1085873, None)],
    "tech": [(2178391, None), (2195081, None)],
    "coordination": [(2042525, 4262160), (2405754, 5219715)],
}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    trusted = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=40, context=trusted) as res:
        return res.read()


def main():
    os.makedirs(OUT, exist_ok=True)
    index = []
    for category, items in WANTED.items():
        for set_id, diff_id in items:
            page = get(f"https://osu.ppy.sh/beatmapsets/{set_id}").decode("utf-8", "replace")
            m = re.search(r'<script id="json-beatmapset"[^>]*>(.*?)</script>', page, re.S)
            info = json.loads(m.group(1))
            mania = [b for b in info["beatmaps"] if b["mode"] == "mania"]
            if diff_id is None:
                four = [b for b in mania if round(b["cs"]) == 4] or mania
                pick = max(four, key=lambda b: b["difficulty_rating"])
            else:
                pick = next(b for b in mania if b["id"] == diff_id)
            path = os.path.join(OUT, f"{category}-{pick['id']}.osu")
            if not os.path.isfile(path):
                with open(path, "wb") as fh:
                    fh.write(get(f"https://osu.ppy.sh/osu/{pick['id']}"))
                time.sleep(0.8)
            index.append({"category": category, "set": set_id, "id": pick["id"],
                          "artist": info["artist"], "title": info["title"],
                          "mapper": info["creator"], "version": pick["version"],
                          "keys": round(pick["cs"]), "site_stars": pick["difficulty_rating"],
                          "bpm": pick["bpm"], "length": pick["total_length"],
                          "status": info["status"], "chosen_by": "you" if diff_id else "hardest 4K",
                          "others": [(b["id"], b["version"], round(b["cs"]), b["difficulty_rating"])
                                     for b in mania if b["id"] != pick["id"]]})
            print(f"{category:10s} {pick['id']:>8} {round(pick['cs'])}K {pick['difficulty_rating']:5.2f}* "
                  f"{pick['bpm']:>6} BPM  {info['artist']} - {info['title']} [{pick['version']}] "
                  f"by {info['creator']} ({info['status']})")
            time.sleep(0.8)
    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
