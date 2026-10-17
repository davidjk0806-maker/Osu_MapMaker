"""mapmaker - turns a song into an osu!mania chart at a chosen star rating."""

import warnings

warnings.filterwarnings("ignore")      # librosa and friends are chatty

from .audio import MapError                               # noqa: E402
from .chart import HANDS, STYLES                          # noqa: E402
from .pipeline import Options, guess_tags, make_map       # noqa: E402

__all__ = ["MapError", "Options", "STYLES", "HANDS", "guess_tags", "make_map"]
