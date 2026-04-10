#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

REPO_URL="${REPO_URL:-https://github.com/plarotta/fca.git}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/fca}"
BRANCH="${BRANCH:-}"

APT_PACKAGES=(
  build-essential
  ca-certificates
  git
  python3
  python3-pip
  python3-venv
  tmux
)

if command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
fi

echo "==> Updating apt package index"
$SUDO apt-get update

echo "==> Upgrading installed packages"
$SUDO apt-get -y upgrade

echo "==> Installing system packages"
$SUDO apt-get install -y "${APT_PACKAGES[@]}"

if [[ ! -d "${INSTALL_DIR}/.git" ]]; then
  echo "==> Cloning repo into ${INSTALL_DIR}"
  git clone "${REPO_URL}" "${INSTALL_DIR}"
else
  echo "==> Repo already exists at ${INSTALL_DIR}; reusing it"
fi

cd "${INSTALL_DIR}"

if [[ -n "${BRANCH}" ]]; then
  echo "==> Checking out branch ${BRANCH}"
  git fetch origin "${BRANCH}"
  git checkout "${BRANCH}"
fi

echo "==> Fetching submodules"
git submodule update --init --recursive

echo "==> Creating virtual environment"
python3 -m venv .venv

echo "==> Activating virtual environment"
source .venv/bin/activate

echo "==> Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing CUDA 12.4 PyTorch wheels"
python -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "==> Installing Python dependencies"
python -m pip install -r requirements.txt

echo
echo "Setup complete."
echo "Repo: ${INSTALL_DIR}"
echo "Next step:"
echo "  cd ${INSTALL_DIR} && source .venv/bin/activate"
