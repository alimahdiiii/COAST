#!/usr/bin/env bash
set -euo pipefail

# Install COAST tools without sudo.
# Install tmux locally and add bin/ to PATH.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$REPO_DIR/bin"
TMUX_BIN="$REPO_DIR/tools/tmux/bin/tmux"

echo "== OpenFoam_Workflow installer =="
echo "Repo: $REPO_DIR"

# Install tmux locally.
if [ -x "$TMUX_BIN" ]; then
  echo "✅ tmux already present: $TMUX_BIN"
else
  echo "Installing local tmux..."
  if [ ! -x "$REPO_DIR/tools/install_tmux.sh" ]; then
    echo "❌ Missing tools/install_tmux.sh" >&2
    exit 1
  fi
  (cd "$REPO_DIR" && bash "$REPO_DIR/tools/install_tmux.sh")
  if [ ! -x "$TMUX_BIN" ]; then
    echo "❌ tmux install did not produce $TMUX_BIN" >&2
    exit 1
  fi
  echo "✅ tmux installed: $TMUX_BIN"
fi

# Install Python dependencies.
if command -v pip3 &>/dev/null || command -v pip &>/dev/null; then
  PIP_CMD="$(command -v pip3 2>/dev/null || command -v pip)"
  echo "Installing Python dependencies..."
  "$PIP_CMD" install --quiet -r "$REPO_DIR/requirements.txt"
  echo "✅ Python dependencies installed"
else
  echo "⚠️  pip not found — install Python packages manually:"
  echo "   pip install -r $REPO_DIR/requirements.txt"
fi

# Choose the bash or zsh startup file.
RC_FILE="$HOME/.bashrc"
if [ -n "${ZSH_VERSION:-}" ]; then
  RC_FILE="$HOME/.zshrc"
elif [ -n "${BASH_VERSION:-}" ]; then
  RC_FILE="$HOME/.bashrc"
else
  # fallback
  [ -f "$HOME/.bashrc" ] && RC_FILE="$HOME/.bashrc"
fi

# Use .zshrc if it exists and .bashrc does not.
if [ -f "$HOME/.zshrc" ] && [ ! -f "$HOME/.bashrc" ]; then
  RC_FILE="$HOME/.zshrc"
fi

touch "$RC_FILE"

BEGIN_MARKER="# >>> openfoam_workflow tools >>>"
END_MARKER="# <<< openfoam_workflow tools <<<"

# Add the PATH block if it is missing.
if grep -qF "$BEGIN_MARKER" "$RC_FILE"; then
  echo "ℹ️ PATH block already present in $RC_FILE"
else
  {
    echo ""
    echo "$BEGIN_MARKER"
    echo "# Add OpenFoam_Workflow commands (joblist, killcase)"
    echo "export PATH=\"$BIN_DIR:\$PATH\""
    echo "$END_MARKER"
  } >> "$RC_FILE"
  echo "✅ Added $BIN_DIR to PATH in $RC_FILE"
fi

# Print setup instructions.
echo
echo "Done."
echo "To apply PATH changes in the current terminal, run:"
echo "  source \"$RC_FILE\""
echo
echo "Quick checks after sourcing:"
echo "  which joblist"
echo "  which killcase"
echo "  \"$TMUX_BIN\" -V"
echo "  python -c 'import streamlit, numpy, scipy, pandas, stl, seaborn, matplotlib'"

