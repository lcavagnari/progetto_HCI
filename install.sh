#!/usr/bin/env bash
set -e

install_pkg() {
    if command -v apt-get >/dev/null 2>&1; then sudo apt-get update && sudo apt-get install -y "$@"
    elif command -v dnf >/dev/null 2>&1; then sudo dnf install -y "$@"
    elif command -v pacman >/dev/null 2>&1; then sudo pacman -Sy --noconfirm "$@"
    elif command -v brew >/dev/null 2>&1; then brew install "$@"
    else echo "No supported package manager found. Install manually: $*" >&2; exit 1
    fi
}

command -v python3 >/dev/null 2>&1 || install_pkg python3 python3-venv python3-pip
command -v unzip >/dev/null 2>&1 || install_pkg unzip

if [ ! -d progetto_HCI ]; then
    git clone https://github.com/lcavagnari/progetto_HCI.git
fi
cd progetto_HCI

[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

if [ -f ../deap-dataset.zip ]; then
    unzip -q -o ../deap-dataset.zip -d .
else
    echo "Warning: ../deap-dataset.zip not found — skipping dataset extraction." >&2
fi

echo ""
echo "Done. In new terminals, activate the venv with:"
echo "  cd $(pwd) && source venv/bin/activate"
