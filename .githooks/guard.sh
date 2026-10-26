#!/bin/sh
# Repository guard. Enforces the commit rules for this repository:
#   1. the configured git identity is the owner identity;
#   2. the local hooks in .githooks are active (commit-msg, pre-commit);
#   3. unpushed commits carry the owner identity and no attribution trailers;
#   4. no secrets in the tracked tree (gitleaks, config in .gitleaks.toml).
# Usage: sh .githooks/guard.sh   (also run by the test command)
# Set GUARD_ALLOW_NO_GITLEAKS=1 only on a host where gitleaks cannot be installed;
# the skip is reported, never silent.
set -u
OWNER_EMAIL="hilaal@gmail.com"
root=$(git rev-parse --show-toplevel 2>/dev/null) || {
  echo "guard: SKIPPED, not inside a git work tree" >&2
  exit 0
}
cd "$root" || exit 1
fail=0
say() { echo "guard: $*" >&2; }

email=$(git config user.email || true)
if [ "$email" != "$OWNER_EMAIL" ]; then
  say "git user.email is '$email', expected $OWNER_EMAIL"
  fail=1
fi

hooks=$(git config core.hooksPath || true)
if [ "$hooks" != ".githooks" ]; then
  say "hooks not active; run: git config core.hooksPath .githooks"
  fail=1
fi

if upstream=$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null); then
  for c in $(git rev-list "$upstream..HEAD"); do
    ae=$(git log -1 --format=%ae "$c"); ce=$(git log -1 --format=%ce "$c")
    if [ "$ae" != "$OWNER_EMAIL" ] || [ "$ce" != "$OWNER_EMAIL" ]; then
      say "unpushed commit $c has author $ae, committer $ce"
      fail=1
    fi
    if git log -1 --format=%B "$c" | sh .githooks/commit-msg /dev/stdin; then :; else
      say "unpushed commit $c carries a refused trailer"
      fail=1
    fi
  done
fi

if command -v gitleaks >/dev/null 2>&1; then
  tmp=$(mktemp -d) || exit 1
  # scan exactly the tracked files, not build output or dependencies
  git ls-files -z | tar --null -T - -cf - 2>/dev/null | tar -xf - -C "$tmp"
  if ! gitleaks dir --no-banner --redact --log-level warn --max-target-megabytes 5 -c "$root/.gitleaks.toml" "$tmp"; then
    say "gitleaks found secrets in tracked files"
    fail=1
  fi
  rm -rf "$tmp"
elif [ "${GUARD_ALLOW_NO_GITLEAKS:-}" = 1 ]; then
  say "gitleaks NOT RUN (GUARD_ALLOW_NO_GITLEAKS=1)"
else
  say "gitleaks not installed: https://github.com/gitleaks/gitleaks/releases (or set GUARD_ALLOW_NO_GITLEAKS=1)"
  fail=1
fi

[ "$fail" = 0 ] && echo "guard: ok"
exit "$fail"
