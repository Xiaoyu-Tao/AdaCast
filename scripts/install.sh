#!/bin/sh
set -eu
adacast_lite_release=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
adacast_lite_prefix=${1:?Usage: install.sh INSTALL_DIRECTORY}
mkdir -p "$adacast_lite_prefix"
adacast_lite_prefix=$(CDPATH= cd -- "$adacast_lite_prefix" && pwd)
if [ -e "$adacast_lite_prefix/libexec/adacast-lite" ]; then
  printf '%s\n' 'Installation already exists; use a new directory.' >&2
  exit 1
fi
mkdir -p "$adacast_lite_prefix/bin" "$adacast_lite_prefix/libexec" "$adacast_lite_prefix/python"
set -- "$adacast_lite_release"/python/*.whl
if [ "$#" -ne 1 ] || [ ! -f "$1" ]; then
  printf '%s\n' 'Release must contain exactly one Python wheel.' >&2
  exit 1
fi
adacast_lite_wheel=$1
if [ -n "${ADACAST_LITE_INSTALL_EXTRAS:-}" ]; then
  adacast_lite_wheel="$adacast_lite_wheel[$ADACAST_LITE_INSTALL_EXTRAS]"
fi
cp "$adacast_lite_release/python/model-overrides.txt" "$adacast_lite_prefix/python/model-overrides.txt"
if command -v uv >/dev/null 2>&1; then
  uv venv --python "${ADACAST_LITE_INSTALL_PYTHON:-3.12}" "$adacast_lite_prefix/python/.venv"
  uv pip install --overrides "$adacast_lite_release/python/model-overrides.txt" --python "$adacast_lite_prefix/python/.venv/bin/python" "$adacast_lite_wheel"
else
  "${ADACAST_LITE_INSTALL_PYTHON:-python3}" -m venv "$adacast_lite_prefix/python/.venv"
  "$adacast_lite_prefix/python/.venv/bin/python" -m pip install "$adacast_lite_wheel"
fi
cp "$adacast_lite_release/adacast-lite" "$adacast_lite_prefix/libexec/adacast-lite"
mkdir -p "$adacast_lite_prefix/share/doc/adacast-lite"
for adacast_lite_doc in README.md LICENSE THIRD_PARTY_NOTICES.md ADACAST_ENGINE_LICENSE.txt; do
  cp "$adacast_lite_release/$adacast_lite_doc" "$adacast_lite_prefix/share/doc/adacast-lite/$adacast_lite_doc"
done
if [ -d "$adacast_lite_release/docs" ]; then
  cp -R "$adacast_lite_release/docs" "$adacast_lite_prefix/share/doc/adacast-lite/"
fi
cat > "$adacast_lite_prefix/bin/adacast-lite" <<'WRAPPER'
#!/bin/sh
set -eu
adacast_lite_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export ADACAST_LITE_HOME="$adacast_lite_root"
export ADACAST_LITE_PYTHON="${ADACAST_LITE_PYTHON:-$adacast_lite_root/python/.venv/bin/python}"
exec "$adacast_lite_root/libexec/adacast-lite" "$@"
WRAPPER
chmod +x "$adacast_lite_prefix/bin/adacast-lite" "$adacast_lite_prefix/libexec/adacast-lite"
printf 'Installed: %s/bin/adacast-lite\n' "$adacast_lite_prefix"
