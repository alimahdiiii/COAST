#!/usr/bin/env bash
set -e

TMUX_VERSION=3.3a
PREFIX="$(cd "$(dirname "$0")" && pwd)/tmux"

echo "Installing tmux $TMUX_VERSION locally into:"
echo "  $PREFIX"

mkdir -p "$PREFIX"
cd "$PREFIX"

if [ -x bin/tmux ]; then
  echo "tmux already installed at $PREFIX/bin/tmux"
  exit 0
fi

echo "Downloading tmux source..."
wget https://github.com/tmux/tmux/releases/download/${TMUX_VERSION}/tmux-${TMUX_VERSION}.tar.gz

tar xf tmux-${TMUX_VERSION}.tar.gz
cd tmux-${TMUX_VERSION}

echo "Configuring..."
./configure --prefix="$PREFIX"

echo "Building..."
make -j$(nproc)

echo "Installing..."
make install

echo
echo "✅ tmux installed at:"
echo "  $PREFIX/bin/tmux"

