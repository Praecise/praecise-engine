#!/bin/sh
# Fails when a commit in BASE..HEAD moves a submodule pointer to a commit that
# does not descend from the previous pointer, or to a commit the submodule's
# remote does not have. A deliberate history rewrite of a submodule (for
# example a rebase onto a new upstream) is allowed only when that commit's
# message carries a line "submodule rewrite: <path>".
#
# Usage: scripts/check-submodule-pointers.sh <base> [<head>]
set -eu

base=${1:?usage: $0 <base> [<head>]}
head=${2:-HEAD}
top=$(git rev-parse --show-toplevel)
cd "$top"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# A blob-less clone of each submodule remote, made once, holds the history
# needed for ancestry checks without the file contents.
mirror() {
    path=$1
    dir="$work/$(printf %s "$path" | tr / _)"
    if [ ! -d "$dir" ]; then
        url=$(git config -f .gitmodules --get "submodule.$path.url" 2>/dev/null || true)
        if [ -z "$url" ]; then
            name=$(git config -f .gitmodules --get-regexp '^submodule\..*\.path$' | awk -v p="$path" '$2 == p { sub(/^submodule\./, "", $1); sub(/\.path$/, "", $1); print $1 }')
            url=$(git config -f .gitmodules --get "submodule.$name.url")
        fi
        git clone --quiet --bare --filter=blob:none "$url" "$dir"
    fi
    printf %s "$dir"
}

status=0
for c in $(git rev-list --reverse "$base..$head"); do
    parent=$(git rev-parse --verify --quiet "$c^1" || true)
    [ -n "$parent" ] || continue
    git diff-tree --no-commit-id -r --raw "$parent" "$c" | awk '$1 == ":160000" && $2 == "160000" { print $3, $4, $6 }' |
    while read -r old new path; do
        repo=$(mirror "$path")
        subject=$(git log -1 --format='%h %s' "$c")
        if ! git -C "$repo" cat-file -e "$new^{commit}" 2>/dev/null; then
            echo "FAIL $subject: $path -> $new is not on the submodule remote"
            exit 1
        fi
        if git -C "$repo" merge-base --is-ancestor "$old" "$new" 2>/dev/null; then
            echo "ok   $subject: $path $(echo "$old" | cut -c1-12) -> $(echo "$new" | cut -c1-12)"
        elif git log -1 --format=%B "$c" | grep -qx "submodule rewrite: $path"; then
            echo "ok   $subject: $path rewritten on purpose"
        else
            echo "FAIL $subject: $path $(echo "$old" | cut -c1-12) -> $(echo "$new" | cut -c1-12) does not descend from the previous pointer"
            exit 1
        fi
    done || status=1
done
exit $status
