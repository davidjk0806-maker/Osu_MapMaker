#!/bin/zsh
# Double-click to start mapmaker; the page opens in your browser. Close this window to stop it.
cd "$(dirname "$0")" || exit 1
exec .venv/bin/python -m mapmaker
