"""Download the songs the regression checks run on.

They are by Kevin MacLeod (incompetech.com), licensed Creative Commons BY 4.0, and
were picked to cover what the engine has to cope with besides the reference song."""
import os
import ssl
import sys
import urllib.parse
import urllib.request

import certifi

HERE = os.path.dirname(os.path.abspath(__file__))
SONGS = {
    "Griphop": "straight 4/4, 90 BPM charted at 180, no vocals",
    "Getting it Done": "135 BPM: needs 1/8 lines to reach higher ratings",
    "Captain Scurvy": "6/8 jig: compound meter",
    "Rollin at 5": "swing",
    "Ether Vox": "hardly any drums, vocals",
    "Electrodoodle": "120 BPM electro with plenty of 1/8 detail",
    "Mega Hyper Ultrastorm": "220 BPM, dense",
}


def main():
    os.makedirs(os.path.join(HERE, "songs"), exist_ok=True)
    trusted = ssl.create_default_context(cafile=certifi.where())
    for name in SONGS:
        path = os.path.join(HERE, "songs", name + ".mp3")
        if not os.path.isfile(path):
            url = ("https://incompetech.com/music/royalty-free/mp3-royaltyfree/"
                   + urllib.parse.quote(name + ".mp3"))
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120, context=trusted) as res:
                data = res.read()
            with open(path, "wb") as fh:
                fh.write(data)
        print(f"{os.path.getsize(path) / 1e6:5.1f} MB  {name}")


if __name__ == "__main__":
    sys.exit(main())
