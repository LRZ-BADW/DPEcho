#!/usr/bin/env bash
set -eo pipefail
git config core.hooksPath .githooks
echo "core.hooksPath set to .githooks"
