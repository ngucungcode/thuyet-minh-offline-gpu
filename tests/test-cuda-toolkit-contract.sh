#!/usr/bin/env bash

set -Eeuo pipefail

TEST_SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "${TEST_SCRIPT_DIR}/.." && pwd)"
LLAMA_INSTALLER="${PROJECT_ROOT}/scripts/install-llama-cpp.sh"
LLAMA_RELEASE="$(jq -er '.components.llama_cpp.release' "${PROJECT_ROOT}/native/components.lock.json")"
LLAMA_COMMIT="$(jq -er '.components.llama_cpp.commit' "${PROJECT_ROOT}/native/components.lock.json")"
LLAMA_LINK="/usr/local/lib/llama.cpp"
TEST_ROOT="$(mktemp -d)"
FAKE_NVCC="${TEST_ROOT}/nvcc"

if [[ "${EUID}" -ne 0 ]]; then
  printf 'CUDA toolkit contract test must run as root\n' >&2
  exit 2
fi

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

targets=(
  "/usr/local/lib/llama.cpp-${LLAMA_RELEASE}-cuda12.6-sm86-offline"
  "/usr/local/lib/llama.cpp-${LLAMA_RELEASE}-cuda12.8-sm86-offline"
)
for target in "${targets[@]}"; do
  [[ ! -e "${target}" && ! -L "${target}" ]] \
    || fail "Refusing to touch pre-existing test target ${target}"
done
[[ ! -e "${LLAMA_LINK}" && ! -L "${LLAMA_LINK}" ]] \
  || fail "Refusing to touch pre-existing llama.cpp link"

cleanup() {
  local target
  local link_target
  if [[ -L "${LLAMA_LINK}" ]]; then
    link_target="$(readlink -f -- "${LLAMA_LINK}")"
    case "${link_target}" in
      "${targets[0]}"|"${targets[1]}") rm -f -- "${LLAMA_LINK}" ;;
      *) printf 'Unsafe llama.cpp link target: %s\n' "${link_target}" >&2 ;;
    esac
  fi
  for target in "${targets[@]}"; do
    case "${target}" in
      /usr/local/lib/llama.cpp-*-cuda12.6-sm86-offline|\
      /usr/local/lib/llama.cpp-*-cuda12.8-sm86-offline)
        rm -rf -- "${target}"
        ;;
      *)
        printf 'Unsafe cleanup target: %s\n' "${target}" >&2
        ;;
    esac
  done
  rm -rf -- "${TEST_ROOT}"
}
trap cleanup EXIT

cat >"${FAKE_NVCC}" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
case "${1:-}" in
  --version)
    if [[ -n "${FAKE_NVCC_RELEASE:-}" ]]; then
      printf 'Cuda compilation tools, release %s, V%s.0\n' \
        "${FAKE_NVCC_RELEASE}" "${FAKE_NVCC_RELEASE}"
    else
      printf 'malformed nvcc output\n'
    fi
    ;;
  --list-gpu-arch)
    printf '%s\n' ${FAKE_NVCC_ARCHES:-compute_86}
    ;;
  *) exit 64 ;;
esac
SH
chmod 0755 "${FAKE_NVCC}"

for version in 12.6 12.8; do
  target="/usr/local/lib/llama.cpp-${LLAMA_RELEASE}-cuda${version}-sm86-offline"
  mkdir -p -- "${target}"
  output="$(
    FAKE_NVCC_RELEASE="${version}" FAKE_NVCC_ARCHES=compute_86 \
      CUDACXX="${FAKE_NVCC}" DUB_LLAMA_CUDA_ARCHITECTURES=86 \
      bash "${LLAMA_INSTALLER}" 2>&1
  )" && fail "CUDA ${version} unexpectedly accepted an invalid cached target"
  printf '%s\n' "${output}" | grep -Fq 'receipt khong khop' \
    || fail "CUDA ${version} did not pass version/architecture validation"
  rm -rf -- "${target}"
done

# A verified 12.6 cache must be reused without network/build work and must
# select only the target whose receipt was produced by CUDA 12.6.
target="${targets[0]}"
mkdir -p -- "${target}"
cat >"${target}/llama-server" <<SH
#!/usr/bin/env bash
printf '%s\n' '${LLAMA_COMMIT:0:7}'
SH
cat >"${target}/llama-cli" <<'SH'
#!/usr/bin/env bash
exit 0
SH
chmod 0755 "${target}/llama-server" "${target}/llama-cli"
jq -n \
  --arg release "${LLAMA_RELEASE}" \
  --arg commit "${LLAMA_COMMIT}" \
  --arg cuda_version 12.6 \
  --arg cuda_architectures 86 \
  --arg llama_server_sha256 "$(sha256sum "${target}/llama-server" | awk '{print $1}')" \
  --arg llama_cli_sha256 "$(sha256sum "${target}/llama-cli" | awk '{print $1}')" \
  '{schema_version:1, release:$release, commit:$commit, cuda_version:$cuda_version,
    cuda_architectures:$cuda_architectures,
    binaries:{llama_server_sha256:$llama_server_sha256,
      llama_cli_sha256:$llama_cli_sha256}}' \
  >"${target}/build-receipt.json"
FAKE_NVCC_RELEASE=12.6 FAKE_NVCC_ARCHES=compute_86 \
  CUDACXX="${FAKE_NVCC}" DUB_LLAMA_CUDA_ARCHITECTURES=86 \
  bash "${LLAMA_INSTALLER}"
[[ -L "${LLAMA_LINK}" && "$(readlink -f -- "${LLAMA_LINK}")" == "${target}" ]] \
  || fail "Valid CUDA 12.6 receipt was not reused"
rm -f -- "${LLAMA_LINK}"
rm -rf -- "${target}"

for version in 12.5 12.9 13.0; do
  output="$(
    FAKE_NVCC_RELEASE="${version}" FAKE_NVCC_ARCHES=compute_86 \
      CUDACXX="${FAKE_NVCC}" DUB_LLAMA_CUDA_ARCHITECTURES=86 \
      bash "${LLAMA_INSTALLER}" 2>&1
  )" && fail "Unsupported CUDA ${version} was accepted"
  printf '%s\n' "${output}" | grep -Fq 'khong nam trong ma tran ho tro' \
    || fail "Unsupported CUDA ${version} failed for the wrong reason"
done

output="$(
  FAKE_NVCC_RELEASE='' FAKE_NVCC_ARCHES=compute_86 \
    CUDACXX="${FAKE_NVCC}" DUB_LLAMA_CUDA_ARCHITECTURES=86 \
    bash "${LLAMA_INSTALLER}" 2>&1
)" && fail "Malformed nvcc version output was accepted"
printf '%s\n' "${output}" | grep -Fq 'khong nam trong ma tran ho tro' \
  || fail "Malformed nvcc output failed for the wrong reason"

output="$(
  FAKE_NVCC_RELEASE=12.6 FAKE_NVCC_ARCHES=compute_80 \
    CUDACXX="${FAKE_NVCC}" DUB_LLAMA_CUDA_ARCHITECTURES=86 \
    bash "${LLAMA_INSTALLER}" 2>&1
)" && fail "CUDA 12.6 without compute_86 was accepted"
printf '%s\n' "${output}" | grep -Fq 'nvcc khong the build cho sm_86' \
  || fail "Missing compute_86 failed for the wrong reason"

printf 'CUDA toolkit contract tests: PASS\n'
