#!/usr/bin/env bash

set -Eeuo pipefail

TEST_SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "${TEST_SCRIPT_DIR}/.." && pwd)"

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/install.sh"

assert_contract() {
  local toolkit="$1"
  local driver="$2"
  local expected_minimum="$3"

  installer_select_cuda_contract "${toolkit}" \
    || fail "CUDA ${toolkit} was not selected"
  [[ "${minimum_driver_version}" == "${expected_minimum}" ]] \
    || fail "CUDA ${toolkit} selected ${minimum_driver_version}"
  installer_parse_driver_version "${driver}" \
    || fail "Driver ${driver} was not parsed"
  installer_driver_meets_minimum \
    || fail "CUDA ${toolkit} rejected driver ${driver}"
}

assert_below_floor() {
  local toolkit="$1"
  local driver="$2"

  installer_select_cuda_contract "${toolkit}" \
    || fail "CUDA ${toolkit} was not selected"
  installer_parse_driver_version "${driver}" \
    || fail "Driver ${driver} was not parsed"
  if installer_driver_meets_minimum; then
    fail "CUDA ${toolkit} accepted driver ${driver} below its floor"
  fi
}

assert_contract 12.6 560.28.03 560.28.03
assert_contract 12.6 560.35.05 560.28.03
assert_contract 12.8 570.26 570.26
assert_contract 12.8 570.211.01 570.26
assert_below_floor 12.6 560.28.02
assert_below_floor 12.8 570.25

if installer_select_cuda_contract 12.7; then
  fail 'Unsupported CUDA 12.7 selected a driver contract'
fi
for malformed in '' 570 570.26beta 570.26.01.2; do
  if installer_parse_driver_version "${malformed}"; then
    fail "Malformed driver version was accepted: ${malformed:-empty}"
  fi
done

printf 'Installer CUDA preflight tests: PASS\n'
