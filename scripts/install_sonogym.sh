#!/usr/bin/env bash
# Install the pinned SonoGym orthopedic-ultrasound provider beside Dr.Anmar.
set -euo pipefail

DR_ANMAR_ROOT="${DR_ANMAR_ROOT:-$HOME/.local/share/dr-anmar}"
INSTALL_ROOT="${DR_ANMAR_SONOGYM_INSTALL_ROOT:-$DR_ANMAR_ROOT/sonogym}"
SONOGYM_ROOT="$INSTALL_ROOT/vendor/SonoGym"
SONOGYM_COMMIT="e67be58334d1a5274f0913af36f56e4b0b7ffe5a"
ASSETS_COMMIT="b37b080a8673f856266a2306724e48d5e034521a"
ASSETS_SHA256="0d5840da3af2eb97e1ca9ebc7c2a969d32b24eef3f3107bc0c9f7db41e4a8f77"
MODELS_SHA256="545113c94b641eda199c6edee1e3586f8cd3a844f0b600470a828fec6210b14f"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

if [[ -z "$UV_BIN" ]]; then
  echo "uv is required to create the isolated Python 3.10 Isaac Lab runtime." >&2
  exit 1
fi

mkdir -p "$INSTALL_ROOT/vendor" "$INSTALL_ROOT/downloads" "$INSTALL_ROOT/cache" "$INSTALL_ROOT/python"
if [[ ! -d "$SONOGYM_ROOT/.git" ]]; then
  git clone https://github.com/SonoGym/SonoGym.git "$SONOGYM_ROOT"
fi

if [[ -n "$(git -C "$SONOGYM_ROOT" status --porcelain)" ]]; then
  echo "SonoGym source has local changes; preserve or remove them before pinning the provider." >&2
  exit 1
fi
git -C "$SONOGYM_ROOT" fetch origin main
git -C "$SONOGYM_ROOT" checkout --detach "$SONOGYM_COMMIT"

download() {
  local name="$1"
  local target="$INSTALL_ROOT/downloads/$name"
  if [[ ! -f "$target" ]]; then
    curl -L --fail --retry 5 -o "$target.part" \
      "https://huggingface.co/datasets/yunkao/SonoGym_assets_models/resolve/$ASSETS_COMMIT/$name"
    mv "$target.part" "$target"
  fi
}

download assets.tar.gz
download models.tar.gz

verify_archive() {
  local name="$1"
  local expected="$2"
  local actual
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$INSTALL_ROOT/downloads/$name" | awk '{print $1}')"
  else
    actual="$(shasum -a 256 "$INSTALL_ROOT/downloads/$name" | awk '{print $1}')"
  fi
  if [[ "$actual" != "$expected" ]]; then
    echo "$name failed its pinned SHA-256 check; refusing to install corrupted or changed assets." >&2
    exit 1
  fi
}

verify_archive assets.tar.gz "$ASSETS_SHA256"
verify_archive models.tar.gz "$MODELS_SHA256"

PACKAGE_ROOT="$SONOGYM_ROOT/source/spinal_surgery/spinal_surgery"
if [[ ! -d "$PACKAGE_ROOT/assets/data/HumanModels" ]]; then
  tar -xzf "$INSTALL_ROOT/downloads/assets.tar.gz" -C "$PACKAGE_ROOT"
fi
if [[ ! -d "$SONOGYM_ROOT/models" ]]; then
  tar -xzf "$INSTALL_ROOT/downloads/models.tar.gz" -C "$SONOGYM_ROOT"
fi

export UV_CACHE_DIR="$INSTALL_ROOT/cache"
export UV_PYTHON_INSTALL_DIR="$INSTALL_ROOT/python"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
if [[ ! -x "$INSTALL_ROOT/env_isaaclab/bin/python" ]]; then
  "$UV_BIN" venv --python 3.10 "$INSTALL_ROOT/env_isaaclab"
fi

PYTHON="$INSTALL_ROOT/env_isaaclab/bin/python"
"$UV_BIN" pip install --python "$PYTHON" setuptools==80.9.0 wheel
"$UV_BIN" pip install --python "$PYTHON" --no-build-isolation flatdict==4.0.1
"$UV_BIN" pip install --python "$PYTHON" \
  "isaaclab[isaacsim,all]==2.1.0" --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match
"$UV_BIN" pip install --python "$PYTHON" \
  "torch==2.5.1" "numpy==1.26.4" "gymnasium==1.2.3" \
  "isaaclab==2.1.0" "isaacsim==4.5.0.0" \
  pyvista ruamel.yaml pydicom nibabel monai
"$UV_BIN" pip install --python "$PYTHON" --no-deps -e "$SONOGYM_ROOT/source/spinal_surgery"

cat > "$INSTALL_ROOT/install_manifest.json" <<EOF
{
  "schema": "dr.anmar.sonogym-install.v1",
  "source_commit": "$SONOGYM_COMMIT",
  "assets_commit": "$ASSETS_COMMIT",
  "assets_sha256": "$ASSETS_SHA256",
  "models_sha256": "$MODELS_SHA256",
  "isaaclab_release": "2.1.0",
  "source_root": "$SONOGYM_ROOT",
  "python": "$PYTHON"
}
EOF

echo "SonoGym orthopedic-ultrasound provider installed at $INSTALL_ROOT"
