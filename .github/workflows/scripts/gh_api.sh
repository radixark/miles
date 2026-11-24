# Source this file. gh_api <gh api args...>: `gh api` with a retry on transient
# errors (HTTP 502, a connection reset from the release CDN), so only a lookup
# that keeps failing fails the caller. Each attempt is buffered so a failed
# attempt's error body never mixes into the output.
gh_api() {
  local out attempt
  for attempt in 1 2 3; do
    out=$(gh api "$@") && { printf '%s\n' "$out"; return 0; }
    [ "$attempt" -lt 3 ] && sleep 10
  done
  return 1
}
