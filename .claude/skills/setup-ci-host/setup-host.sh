#!/usr/bin/env bash
#
# setup-host.sh — prepare a new miles CI host so /data/miles_ci is canonical.
# See SKILL.md (next to this script) for the full operator-facing docs.
#
# Behavior summary:
#   1. df probes mounted filesystems; picks the mount point with most total bytes
#      among non-system fs types.
#   2. If the chosen mount is /data: ensure /data/miles_ci exists as a real dir.
#   3. Else: ensure <chosen>/miles_ci exists, atomically symlink /data/miles_ci.
#   4. mkdir -p /data/miles_ci/{models,datasets,hf_cache}.
#   5. Print summary.
#
# Idempotent. Re-running on a properly-prepared host is a no-op except for the
# summary line. When /data/miles_ci already exists pointing somewhere
# unexpected, prompts the operator interactively (migrate / wipe / abort);
# non-interactive runs (cron, CI) exit non-zero instead of choosing silently.

set -euo pipefail

# --- Preconditions -----------------------------------------------------------

if [[ "$(id -u)" -ne 0 ]] && ! [ -w / ]; then
  echo "ERROR: setup-host.sh needs to write under /; re-run with sudo:" >&2
  echo "  sudo ./setup-host.sh" >&2
  exit 1
fi

# --- Disk probe --------------------------------------------------------------

# Pick the mount point with the most total size, excluding ephemeral / pseudo
# filesystems and well-known system mounts. The leading `df` columns under
# --output=target,size are: target, 1K-blocks (with -B1: bytes).
biggest_mount=$(
  df --output=target,size -B1 -x tmpfs -x devtmpfs -x overlay -x squashfs 2>/dev/null \
    | awk 'NR>1 && $1 !~ /^\/(boot|dev|proc|run|sys)(\/|$)/ {print $2, $1}' \
    | sort -rn \
    | head -1 \
    | awk '{print $2}'
)

if [[ -z "${biggest_mount}" ]]; then
  echo "ERROR: could not detect any non-system mount from df output" >&2
  exit 1
fi

# --- Decide canonical target -------------------------------------------------

# canonical: /data/miles_ci on every host. real_target is the path on the
# biggest disk where the data actually lives.
canonical=/data/miles_ci

if [[ "${biggest_mount}" == "/data" ]]; then
  real_target=/data/miles_ci
else
  real_target="${biggest_mount}/miles_ci"
fi

# --- Establish /data/miles_ci ------------------------------------------------

# Make sure the real target directory exists on the big disk.
mkdir -p "${real_target}"

# Helper: place the symlink ${canonical} -> ${real_target}, atomically.
place_symlink() {
  mkdir -p /data
  local tmp_link
  tmp_link="$(mktemp -u "${canonical}.XXXXXX.tmp")"
  ln -s "${real_target}" "${tmp_link}"
  mv -T "${tmp_link}" "${canonical}"
}

# Helper: prompt the operator only when running interactively. In
# non-interactive contexts (cron, scripts, CI), fall through to the caller's
# bail path — never make destructive choices without a human.
interactive() { [ -t 0 ] && [ -t 1 ]; }

if [[ "${real_target}" == "${canonical}" ]]; then
  # /data IS the big disk; /data/miles_ci is the real path; nothing to symlink.
  :
elif [[ -L "${canonical}" ]]; then
  current_target=$(readlink "${canonical}")
  if [[ "${current_target}" == "${real_target}" ]]; then
    : # already correct symlink
  else
    cat >&2 <<EOF
WARNING: ${canonical} is a symlink to '${current_target}', but the biggest
disk is '${biggest_mount}' (expected target '${real_target}'). Re-pointing
does NOT migrate any data already at '${current_target}'.
EOF
    if ! interactive; then
      echo "Refusing to silently re-point in non-interactive mode." >&2
      echo "Re-run from a terminal to confirm, or fix manually." >&2
      exit 1
    fi
    read -r -p "Re-point ${canonical} -> ${real_target}? [y/N]: " choice </dev/tty
    if [[ "${choice}" =~ ^[yY]$ ]]; then
      rm -f "${canonical}"
      place_symlink
    else
      echo "Aborted by operator." >&2
      exit 1
    fi
  fi
elif [[ -d "${canonical}" ]] && [[ -n "$(ls -A "${canonical}" 2>/dev/null || true)" ]]; then
  cat >&2 <<EOF
WARNING: ${canonical} is a non-empty real directory, but the biggest disk
is '${biggest_mount}' (expected target '${real_target}'). The data here
lives on a smaller filesystem than the canonical CI cache should.

Options:
  [m] migrate: rsync ${canonical}/ -> ${real_target}/, then rm -rf ${canonical}, then symlink
  [w] wipe:    rm -rf ${canonical} (DATA LOST), then symlink to ${real_target}
  [a] abort:   leave as-is and exit non-zero (default)
EOF
  if ! interactive; then
    echo "Refusing to choose silently in non-interactive mode." >&2
    echo "Re-run from a terminal, or migrate/wipe manually." >&2
    exit 1
  fi
  read -r -p "Choice [m/w/a]: " choice </dev/tty
  case "${choice}" in
    m|M)
      command -v rsync >/dev/null || { echo "ERROR: rsync not installed" >&2; exit 1; }
      rsync -aHX --info=progress2 "${canonical}/" "${real_target}/"
      rm -rf "${canonical}"
      ;;
    w|W)
      read -r -p "Type 'yes' to confirm rm -rf ${canonical}: " confirm </dev/tty
      [[ "${confirm}" == "yes" ]] || { echo "Aborted by operator." >&2; exit 1; }
      rm -rf "${canonical}"
      ;;
    *)
      echo "Aborted by operator." >&2
      exit 1
      ;;
  esac
  place_symlink
else
  # /data/miles_ci does not exist (or is an empty dir we can remove).
  [[ -d "${canonical}" ]] && rmdir "${canonical}" 2>/dev/null || true
  place_symlink
fi

# --- Standard subdirectories -------------------------------------------------

mkdir -p \
  "${canonical}/models" \
  "${canonical}/datasets" \
  "${canonical}/hf_cache"

# --- Summary -----------------------------------------------------------------

echo "biggest mount  : ${biggest_mount}"
echo "real target    : ${real_target}"
if [[ -L "${canonical}" ]]; then
  echo "/data/miles_ci : symlink -> $(readlink "${canonical}")"
else
  echo "/data/miles_ci : real directory"
fi
df -h --output=target,size,used,avail "${biggest_mount}" | tail -1 \
  | awk '{printf "disk usage     : %s used / %s total (%s available)\n", $3, $2, $4}'
