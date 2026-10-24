#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_output="$(dirname "$PROJECT_DIR")/ragloom-source.tar.gz"
output_path="${1:-$default_output}"

if [[ "$output_path" != /* ]]; then
    output_path="$(pwd)/$output_path"
fi
output_path="$(readlink -m "$output_path")"

case "$output_path" in
    "$PROJECT_DIR"|"$PROJECT_DIR"/*)
        printf 'Output must be outside the project directory: %s\n' "$output_path" >&2
        exit 2
        ;;
esac

tar --exclude-from="$PROJECT_DIR/.archiveignore" \
    -czf "$output_path" \
    -C "$PROJECT_DIR" .

printf 'Created source archive: %s\n' "$output_path"
printf 'Excluded runtime data, secrets, local dependencies, caches, and build outputs.\n'
