#!/bin/sh
set -eu

RELEASE_REPO="pullbase/kontinuo-releases"
RELEASE_BASE_URL="${KONTINUO_INSTALL_RELEASE_BASE_URL:-https://github.com/$RELEASE_REPO/releases}"
COSIGN_VERSION="${KONTINUO_INSTALL_COSIGN_VERSION:-v2.6.1}"
COSIGN_BASE_URL="${KONTINUO_INSTALL_COSIGN_BASE_URL:-https://github.com/sigstore/cosign/releases/download/$COSIGN_VERSION}"
OIDC_ISSUER="https://token.actions.githubusercontent.com"

usage() {
  printf '%s\n' 'Usage: install.sh [--version <tag>] [--install-dir <path>] [--no-setup-hint]'
}

log() { printf '%s\n' "$*"; }
die() { printf 'kontinuo install: %s\n' "$*" >&2; exit 1; }
PROGRESS_STEP=0
PROGRESS_TOTAL=8
PROGRESS_WIDTH=20

repeat_char() {
  char="$1"
  count="$2"
  out=""
  while [ "$count" -gt 0 ]; do
    out="$out$char"
    count=$((count - 1))
  done
  printf '%s' "$out"
}

progress_step() {
  [ "${KONTINUO_INSTALL_DRY_RUN:-}" = 1 ] && return
  PROGRESS_STEP=$((PROGRESS_STEP + 1))
  filled=$((PROGRESS_STEP * PROGRESS_WIDTH / PROGRESS_TOTAL))
  empty=$((PROGRESS_WIDTH - filled))
  bar="$(repeat_char '#' "$filled")$(repeat_char '-' "$empty")"
  printf 'kontinuo install [%s] %d/%d %s\n' "$bar" "$PROGRESS_STEP" "$PROGRESS_TOTAL" "$1" >&2
}

TAG=""
INSTALL_DIR=""
SHOW_SETUP_HINT=1
PLATFORM_OS=""
PLATFORM_ARCH=""
ARCHIVE=""
COSIGN_BIN=""
COSIGN_SOURCE=""
COSIGN_TMP_ROOT=""
INSTALL_TARGET=""
COMMAND_PATH=""
COMMAND_DIR=""
NEED_SYMLINK=0
PATH_HINT_NEEDED=0
USE_SUDO_INSTALL=0
USE_SUDO_LINK=0
INSTALL_MODE=""


while [ "$#" -gt 0 ]; do
  case "$1" in
    --version)
      [ "$#" -ge 2 ] || die "--version requires a tag"
      TAG="$2"
      shift 2
      ;;
    --install-dir)
      [ "$#" -ge 2 ] || die "--install-dir requires a path"
      INSTALL_DIR="$2"
      shift 2
      ;;
    --no-setup-hint)
      SHOW_SETUP_HINT=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

os_name() {
  if [ -n "${KONTINUO_INSTALL_TEST_OS:-}" ]; then
    printf '%s' "$KONTINUO_INSTALL_TEST_OS"
    return
  fi
  uname -s
}

arch_name() {
  if [ -n "${KONTINUO_INSTALL_TEST_ARCH:-}" ]; then
    printf '%s' "$KONTINUO_INSTALL_TEST_ARCH"
    return
  fi
  uname -m
}

normalize_platform() {
  raw_os="$(os_name)"
  raw_arch="$(arch_name)"

  case "$raw_os" in
    Darwin|darwin) PLATFORM_OS="darwin" ;;
    Linux|linux) PLATFORM_OS="linux" ;;
    *) die "unsupported platform: $raw_os/$raw_arch" ;;
  esac

  case "$raw_arch" in
    arm64|aarch64) PLATFORM_ARCH="arm64" ;;
    x86_64|amd64) PLATFORM_ARCH="amd64" ;;
    *) die "unsupported platform: $raw_os/$raw_arch" ;;
  esac

  case "$PLATFORM_OS" in
    darwin) ARCHIVE="kontinuo-darwin-$PLATFORM_ARCH.zip" ;;
    linux) ARCHIVE="kontinuo-linux-$PLATFORM_ARCH.tar.gz" ;;
    *) die "unsupported platform: $raw_os/$raw_arch" ;;
  esac
}

path_contains() {
  dir="$1"
  case ":${PATH:-}:" in
    *":$dir:"*) return 0 ;;
    *) return 1 ;;
  esac
}

usr_local_bin() {
  if [ -n "${KONTINUO_INSTALL_TEST_USR_LOCAL_BIN:-}" ]; then
    printf '%s' "$KONTINUO_INSTALL_TEST_USR_LOCAL_BIN"
    return
  fi
  printf '%s' '/usr/local/bin'
}

can_write_or_create_dir() {
  dir="$1"
  if [ -d "$dir" ]; then
    [ -w "$dir" ]
    return
  fi
  parent="$(dirname "$dir")"
  [ -d "$parent" ] && [ -w "$parent" ]
}

is_kontinuo_binary() {
  binary="$1"
  [ -f "$binary" ] || return 1
  output="$("$binary" version 2>/dev/null)" || return 1
  case "$output" in
    kontinuo*) return 0 ;;
    *) return 1 ;;
  esac
}

link_points_to() {
  link="$1"
  target="$2"
  [ -L "$link" ] || return 1
  actual="$(readlink "$link")" || return 1
  [ "$actual" = "$target" ]
}

set_install_target() {
  INSTALL_DIR="$1"
  INSTALL_TARGET="$INSTALL_DIR/kontinuo"
  COMMAND_PATH="$2"
  COMMAND_DIR="$(dirname "$COMMAND_PATH")"
  INSTALL_MODE="$3"
}

fallback_to_user_install() {
  [ -n "${HOME:-}" ] || die "HOME is not set; pass --install-dir"
  set_install_target "$HOME/.local/bin" "$HOME/.local/bin/kontinuo" "user-fallback"
  NEED_SYMLINK=0
  PATH_HINT_NEEDED=1
  USE_SUDO_INSTALL=0
  USE_SUDO_LINK=0
}

choose_macos_install_dir() {
  [ -n "${HOME:-}" ] || die "HOME is not set; pass --install-dir"
  macos_command_dir="$(usr_local_bin)"
  macos_command_path="$macos_command_dir/kontinuo"
  legacy_dir="$HOME/.local/bin"
  legacy_target="$legacy_dir/kontinuo"

  if [ -L "$macos_command_path" ]; then
    if link_points_to "$macos_command_path" "$legacy_target"; then
      set_install_target "$legacy_dir" "$macos_command_path" "macos-legacy-symlink"
      NEED_SYMLINK=1
      return
    fi
    die "refusing to replace existing symlink: $macos_command_path"
  fi

  if [ -e "$macos_command_path" ]; then
    if is_kontinuo_binary "$macos_command_path"; then
      set_install_target "$macos_command_dir" "$macos_command_path" "macos-direct"
      return
    fi
    die "refusing to overwrite existing non-Kontinuo file: $macos_command_path"
  fi

  if [ -x "$legacy_target" ] && is_kontinuo_binary "$legacy_target"; then
    set_install_target "$legacy_dir" "$macos_command_path" "macos-legacy-symlink"
    NEED_SYMLINK=1
    return
  fi

  set_install_target "$macos_command_dir" "$macos_command_path" "macos-direct"
}

choose_linux_install_dir() {
  [ -n "${HOME:-}" ] || die "HOME is not set; pass --install-dir"
  if path_contains "$HOME/.local/bin"; then
    set_install_target "$HOME/.local/bin" "$HOME/.local/bin/kontinuo" "linux-user"
    return
  fi
  if [ -d "$HOME/.local/bin" ]; then
    set_install_target "$HOME/.local/bin" "$HOME/.local/bin/kontinuo" "linux-user"
    PATH_HINT_NEEDED=1
    return
  fi
  if path_contains "$HOME/bin"; then
    set_install_target "$HOME/bin" "$HOME/bin/kontinuo" "linux-user"
    return
  fi
  set_install_target "$HOME/.local/bin" "$HOME/.local/bin/kontinuo" "linux-user"
  PATH_HINT_NEEDED=1
}

choose_install_dir() {
  if [ -n "$INSTALL_DIR" ]; then
    set_install_target "$INSTALL_DIR" "$INSTALL_DIR/kontinuo" "explicit"
    if ! path_contains "$INSTALL_DIR"; then
      PATH_HINT_NEEDED=1
    fi
    return
  fi

  case "$PLATFORM_OS" in
    darwin) choose_macos_install_dir ;;
    linux) choose_linux_install_dir ;;
    *) die "unsupported platform: $PLATFORM_OS/$PLATFORM_ARCH" ;;
  esac
}

confirm_sudo() {
  target="$1"
  [ "${KONTINUO_INSTALL_TEST_NO_SUDO:-}" = 1 ] && return 1
  if [ "${KONTINUO_INSTALL_TEST_ASSUME_SUDO:-}" = 1 ]; then
    command -v sudo >/dev/null 2>&1 || return 1
    return 0
  fi
  command -v sudo >/dev/null 2>&1 || return 1
  [ -r /dev/tty ] && [ -w /dev/tty ] || return 1

  {
    printf '%s\n' "Kontinuo needs permission to place the command at $target."
    printf '%s\n' 'This lets you run `kontinuo` without editing PATH.'
    printf '%s' 'Run this step with sudo? [y/N] '
  } >/dev/tty
  read answer </dev/tty || return 1
  case "$answer" in
    y|Y|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

resolve_install_permissions() {
  if [ "$PLATFORM_OS" != "darwin" ]; then
    return 0
  fi
  if [ "$INSTALL_MODE" = "explicit" ]; then
    return 0
  fi

  if [ "$NEED_SYMLINK" = 1 ]; then
    if [ -L "$COMMAND_PATH" ] && link_points_to "$COMMAND_PATH" "$INSTALL_TARGET"; then
      return
    fi
    if can_write_or_create_dir "$COMMAND_DIR"; then
      return
    fi
    if confirm_sudo "$COMMAND_PATH"; then
      USE_SUDO_LINK=1
      return
    fi
    fallback_to_user_install
    return
  fi

  if can_write_or_create_dir "$INSTALL_DIR"; then
    return
  fi
  if confirm_sudo "$COMMAND_PATH"; then
    USE_SUDO_INSTALL=1
    return
  fi
  fallback_to_user_install
}


sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
    return
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
    return
  fi
  die "missing shasum or sha256sum"
}

download_file() {
  url="$1"
  out="$2"
  case "$url" in
    file://*)
      cp "${url#file://}" "$out"
      ;;
    https://*)
      command -v curl >/dev/null 2>&1 || die "missing curl for HTTPS download"
      curl -fsSL "$url" -o "$out"
      ;;
    *)
      die "refusing unsupported download URL: $url"
      ;;
  esac
}

resolve_tag() {
  if [ -n "$TAG" ]; then
    return
  fi
  if [ -n "${KONTINUO_INSTALL_LATEST_TAG:-}" ]; then
    TAG="$KONTINUO_INSTALL_LATEST_TAG"
    return
  fi
  if [ "${KONTINUO_INSTALL_DRY_RUN:-}" = 1 ]; then
    TAG="latest"
    return
  fi

  latest_json="$(mktemp)"
  download_file "https://api.github.com/repos/$RELEASE_REPO/releases/latest" "$latest_json"
  TAG="$(awk 'match($0, /"tag_name"[[:space:]]*:[[:space:]]*"[^"]+"/) { s = substr($0, RSTART, RLENGTH); sub(/^.*"tag_name"[[:space:]]*:[[:space:]]*"/, "", s); sub(/"$/, "", s); print s; exit }' "$latest_json")"
  rm -f "$latest_json"
  [ -n "$TAG" ] || die "could not resolve latest release tag"
}

is_prerelease_tag() {
  case "$1" in
    *-beta*|*-alpha*|*-rc*) return 0 ;;
    *) return 1 ;;
  esac
}

cosign_asset_name() {
  case "$PLATFORM_OS/$PLATFORM_ARCH" in
    linux/amd64) printf '%s' 'cosign-linux-amd64' ;;
    linux/arm64) printf '%s' 'cosign-linux-arm64' ;;
    darwin/amd64) printf '%s' 'cosign-darwin-amd64' ;;
    darwin/arm64) printf '%s' 'cosign-darwin-arm64' ;;
    *) die "cosign bootstrap unsupported for $PLATFORM_OS/$PLATFORM_ARCH" ;;
  esac
}

expected_cosign_sha() {
  case "$PLATFORM_OS/$PLATFORM_ARCH" in
    linux/amd64) printf '%s' "${KONTINUO_INSTALL_COSIGN_SHA256_LINUX_AMD64:-064954c5d8c7e3b28188eee5b1727b31c411550bc5fefd41aa672d3c761d103a}" ;;
    linux/arm64) printf '%s' "${KONTINUO_INSTALL_COSIGN_SHA256_LINUX_ARM64:-56a16480bdd56ec789abaa65924402f6b92c0041f06885995853c05567b76f34}" ;;
    darwin/amd64) printf '%s' "${KONTINUO_INSTALL_COSIGN_SHA256_DARWIN_AMD64:-f1ed2787cc9648fd3c644fcb279e43f3f55da63b788d69a527aa14ad97ffdca1}" ;;
    darwin/arm64) printf '%s' "${KONTINUO_INSTALL_COSIGN_SHA256_DARWIN_ARM64:-54047052cf46f40a5c3c95a510db276e164ba77e096aea1ca1b733f770359689}" ;;
  esac
}

cleanup_cosign() {
  if [ -n "$COSIGN_TMP_ROOT" ] && [ -d "$COSIGN_TMP_ROOT" ]; then
    rm -rf "$COSIGN_TMP_ROOT"
  fi
}

bootstrap_cosign() {
  expected="$(expected_cosign_sha)"
  [ -n "$expected" ] || die "missing pinned cosign checksum for $PLATFORM_OS/$PLATFORM_ARCH"
  if [ "${KONTINUO_INSTALL_DRY_RUN:-}" = 1 ]; then
    COSIGN_SOURCE="downloaded"
    return
  fi

  COSIGN_TMP_ROOT="$(mktemp -d)"
  COSIGN_BIN="$COSIGN_TMP_ROOT/cosign"
  asset="$(cosign_asset_name)"
  download_file "$COSIGN_BASE_URL/$asset" "$COSIGN_BIN"
  actual="$(sha256_file "$COSIGN_BIN")"
  [ "$actual" = "$expected" ] || die "cosign checksum mismatch"
  chmod +x "$COSIGN_BIN"
  COSIGN_SOURCE="downloaded"
}

ensure_cosign() {
  if command -v cosign >/dev/null 2>&1; then
    COSIGN_BIN="$(command -v cosign)"
    COSIGN_SOURCE="path"
    return
  fi
  bootstrap_cosign
}

release_asset_url() {
  tag="$1"
  asset="$2"
  case "$RELEASE_BASE_URL" in
    file://*) printf 'file://%s/%s/%s' "${RELEASE_BASE_URL#file://}" "$tag" "$asset" ;;
    https://*) printf '%s/download/%s/%s' "$RELEASE_BASE_URL" "$tag" "$asset" ;;
    *) die "refusing unsupported release base URL: $RELEASE_BASE_URL" ;;
  esac
}

verify_checksum() {
  file="$1"
  sums="$2"
  base="$(basename "$file")"
  expected="$(awk -v f="$base" '$2 == f || $2 == "*" f { print $1; exit }' "$sums")"
  [ -n "$expected" ] || die "checksum missing for $base"
  actual="$(sha256_file "$file")"
  [ "$actual" = "$expected" ] || die "checksum mismatch for $base"
}

verify_sigstore() {
  archive="$1"
  bundle="$2"
  identity="https://github.com/pullbase/Kontinuo/.github/workflows/release.yml@refs/tags/$TAG"
  if ! "$COSIGN_BIN" verify-blob "$archive" --bundle "$bundle" --certificate-identity "$identity" --certificate-oidc-issuer "$OIDC_ISSUER"; then
    die "Sigstore verification failed"
  fi
}

check_existing_install_target() {
  target="$1"
  if [ -L "$target" ]; then
    die "refusing to overwrite symlink install target: $target"
  fi
  if [ -e "$target" ] && ! is_kontinuo_binary "$target"; then
    die "refusing to overwrite existing non-Kontinuo file: $target"
  fi
}

copy_candidate_without_sudo() {
  candidate="$1"
  tmp_target="$INSTALL_DIR/.kontinuo.$$"
  mkdir -p "$INSTALL_DIR"
  if cp "$candidate" "$tmp_target" &&
    chmod +x "$tmp_target" &&
    mv "$tmp_target" "$INSTALL_TARGET"; then
    return
  fi
  rm -f "$tmp_target"
  die "failed to install binary to $INSTALL_TARGET"
}

copy_candidate_to_target() {
  candidate="$1"
  check_existing_install_target "$INSTALL_TARGET"
  if [ "$USE_SUDO_INSTALL" = 1 ]; then
    tmp_target="$INSTALL_DIR/.kontinuo.$$"
    if sudo mkdir -p "$INSTALL_DIR" &&
      sudo cp "$candidate" "$tmp_target" &&
      sudo chmod +x "$tmp_target" &&
      sudo mv "$tmp_target" "$INSTALL_TARGET"; then
      return
    fi
    sudo rm -f "$tmp_target" >/dev/null 2>&1 || true
    log "sudo install failed; falling back to $HOME/.local/bin/kontinuo"
    fallback_to_user_install
    check_existing_install_target "$INSTALL_TARGET"
  fi
  copy_candidate_without_sudo "$candidate"
}


ensure_command_symlink() {
  if [ "$NEED_SYMLINK" != 1 ]; then
    return 0
  fi
  if [ -L "$COMMAND_PATH" ]; then
    link_points_to "$COMMAND_PATH" "$INSTALL_TARGET" || die "refusing to replace existing symlink: $COMMAND_PATH"
    return
  fi
  if [ -e "$COMMAND_PATH" ]; then
    die "refusing to overwrite existing non-Kontinuo file: $COMMAND_PATH"
  fi
  if [ "$USE_SUDO_LINK" = 1 ]; then
    if sudo mkdir -p "$COMMAND_DIR" &&
      sudo ln -s "$INSTALL_TARGET" "$COMMAND_PATH"; then
      return
    fi
    log "sudo symlink failed; using $INSTALL_TARGET directly"
    COMMAND_PATH="$INSTALL_TARGET"
    COMMAND_DIR="$INSTALL_DIR"
    NEED_SYMLINK=0
    PATH_HINT_NEEDED=1
    USE_SUDO_LINK=0
    return
  fi
  if mkdir -p "$COMMAND_DIR" &&
    ln -s "$INSTALL_TARGET" "$COMMAND_PATH"; then
    return
  fi
  die "failed to create command symlink at $COMMAND_PATH"
}

install_binary() {
  archive="$1"
  workdir="$2"
  mkdir -p "$workdir/extract"
  case "$archive" in
    *.tar.gz)
      tar -xzf "$archive" -C "$workdir/extract" kontinuo
      ;;
    *.zip)
      command -v unzip >/dev/null 2>&1 || die "missing unzip"
      unzip -q "$archive" kontinuo -d "$workdir/extract"
      ;;
    *)
      die "unsupported archive format: $archive"
      ;;
  esac
  candidate="$workdir/extract/kontinuo"
  [ ! -L "$candidate" ] || die "refusing symlink binary"
  [ -f "$candidate" ] || die "archive did not contain kontinuo binary"
  chmod +x "$candidate"
  "$candidate" version >/dev/null
  copy_candidate_to_target "$candidate"
  ensure_command_symlink
}

path_hint_dir() {
  if [ -n "${HOME:-}" ] && [ "$INSTALL_DIR" = "$HOME/.local/bin" ]; then
    printf '%s' '$HOME/.local/bin'
    return
  fi
  printf '%s' "$INSTALL_DIR"
}

print_path_hint() {
  hint_dir="$(path_hint_dir)"
  log ""
  log '`kontinuo` is not on PATH in this shell.'
  log "Run this now:"
  log "  export PATH=\"$hint_dir:\$PATH\""
  case "${SHELL:-}" in
    */zsh)
      log "To make it permanent for zsh:"
      log "  printf '\\nexport PATH=\"$hint_dir:\$PATH\"\\n' >> ~/.zshrc"
      ;;
    */bash)
      log "To make it permanent for bash:"
      log "  printf '\\nexport PATH=\"$hint_dir:\$PATH\"\\n' >> ~/.bashrc"
      ;;
  esac
}

print_post_install() {
  log "Kontinuo installed:"
  log "  binary:  $INSTALL_TARGET"
  if [ "$NEED_SYMLINK" = 1 ]; then
    log "  command: $COMMAND_PATH -> $INSTALL_TARGET"
  else
    log "  command: $COMMAND_PATH"
  fi
  log ""
  log "PATH check:"
  if resolved="$(command -v kontinuo 2>/dev/null)"; then
    log "  command -v kontinuo: $resolved"
    if [ "$resolved" != "$COMMAND_PATH" ]; then
      log "  warning: PATH resolves a different kontinuo command before $COMMAND_PATH"
    fi
  else
    log "  command -v kontinuo: not found"
    PATH_HINT_NEEDED=1
  fi
  if version_output="$("$COMMAND_PATH" version 2>/dev/null)"; then
    log "  version: $version_output"
  fi
  if [ "$PATH_HINT_NEEDED" = 1 ]; then
    print_path_hint
  fi
}

run_install() {
  workdir="$(mktemp -d)"
  trap 'rm -rf "$workdir"; cleanup_cosign' EXIT INT TERM

  archive_path="$workdir/$ARCHIVE"
  bundle_path="$workdir/$ARCHIVE.sigstore.json"
  sums_path="$workdir/checksums.txt"

  progress_step "Downloading release assets"
  download_file "$(release_asset_url "$TAG" "$ARCHIVE")" "$archive_path"
  download_file "$(release_asset_url "$TAG" "$ARCHIVE.sigstore.json")" "$bundle_path"
  download_file "$(release_asset_url "$TAG" checksums.txt)" "$sums_path"

  progress_step "Verifying release assets"
  verify_checksum "$archive_path" "$sums_path"
  verify_checksum "$bundle_path" "$sums_path"
  verify_sigstore "$archive_path" "$bundle_path"
  progress_step "Installing binary"
  install_binary "$archive_path" "$workdir"

  print_post_install
  if [ "$PLATFORM_OS" = "darwin" ] && is_prerelease_tag "$TAG"; then
    log "Note: beta macOS builds are Sigstore-verified but not Apple-notarized yet."
  fi
  if [ "$SHOW_SETUP_HINT" = 1 ]; then
    log "Next: kontinuo setup --host auto --cwd ."
    log "Tip: in a terminal, setup asks before adding Kontinuo handoff instructions to AGENTS.md or CLAUDE.md; use --instructions yes or --instructions no to choose without a prompt."
    log "Note: normal MCP hosts write checkpoints when the agent follows those instructions or you ask it to call Kontinuo."
  fi
}

main() {
  progress_step "Detecting platform"
  normalize_platform
  progress_step "Choosing install path"
  choose_install_dir
  progress_step "Resolving release"
  resolve_tag
  progress_step "Preparing Sigstore verifier"
  ensure_cosign

  if [ "${KONTINUO_INSTALL_DRY_RUN:-}" = 1 ]; then
    log "tag=$TAG"
    log "archive=$ARCHIVE"
    log "install_dir=$INSTALL_DIR"
    log "install_target=$INSTALL_TARGET"
    log "command_path=$COMMAND_PATH"
    log "install_mode=$INSTALL_MODE"
    log "cosign_source=$COSIGN_SOURCE"
    cleanup_cosign
    exit 0
  fi

  progress_step "Checking install permissions"
  resolve_install_permissions

  run_install
}

main "$@"
