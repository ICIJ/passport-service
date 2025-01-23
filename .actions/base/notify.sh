#!/bin/bash

set -eu

# shellcheck source=lib.sh
source "$(dirname "$(realpath "$0")")/lib.sh"

notify "$@"
