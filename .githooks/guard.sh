#!/bin/sh
# Repository guard. Enforces the commit rules for this repository:
#   1. the configured git identity is the owner identity;
#   2. the local hooks in .githooks are active (commit-msg, pre-commit);
#   3. unpushed commits carry the owner identity and no attribution trailers;
#   4. no secrets in the tree (gitleaks, config in .gitleaks.toml).
# In a git checkout all four run, and the secret scan covers exactly the tracked files.
# Outside a git checkout (a deploy release directory, an unpacked tarball) 1 to 3 do not
# apply and are skipped with one printed line; the secret scan covers the directory itself,
# with dependencies and build output excluded by .gitleaks.toml.
# gitleaks is required in both cases: when it is missing the guard fails.
# Usage: sh .githooks/guard.sh   (also run by the test command)
set -u
OWNER_EMAIL="hilaal@gmail.com"
say() { echo "guard: $*" >&2; }

# The repository root is the directory holding .githooks, wherever the guard is run from.
root=$(cd "$(dirname "$0")/.." && pwd -P) || exit 1
cd "$root" || exit 1
fail=0

if ! command -v gitleaks >/dev/null 2>&1; then
  say "gitleaks not found on PATH ($PATH); install the latest release from https://github.com/gitleaks/gitleaks/releases into /usr/local/bin"
  exit 1
fi

# A git checkout of this repository: git reports a work tree whose top level is this root.
# A directory nested inside some other repository is not a checkout of this one.
top=$(git rev-parse --show-toplevel 2>/dev/null) && top=$(cd "$top" && pwd -P) || top=""

if [ "$top" = "$root" ]; then
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

  tmp=$(mktemp -d) || exit 1
  # scan exactly the tracked files, not build output or dependencies
  git ls-files -z | tar --null -T - -cf - 2>/dev/null | tar -xf - -C "$tmp"
  if ! gitleaks dir --no-banner --redact --log-level warn --max-target-megabytes 5 -c "$root/.gitleaks.toml" "$tmp"; then
    say "gitleaks found secrets in tracked files"
    fail=1
  fi
  rm -rf "$tmp"
else
  say "not a git checkout: commit identity checks skipped"
  if ! gitleaks dir --no-banner --redact --log-level warn --max-target-megabytes 5 -c "$root/.gitleaks.toml" "$root"; then
    say "gitleaks found secrets in $root"
    fail=1
  fi
fi

[ "$fail" = 0 ] && echo "guard: ok"
exit "$fail"
