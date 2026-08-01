#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
LOCK_PATH="${PROJECT_ROOT}/native/components.lock.json"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Trinh cai llama.cpp phai chay bang root" >&2
  exit 2
fi

LLAMA_RELEASE="$(jq -er '.components.llama_cpp.release' "${LOCK_PATH}")"
LLAMA_COMMIT="$(jq -er '.components.llama_cpp.commit' "${LOCK_PATH}")"
LLAMA_REPOSITORY="$(jq -er '.components.llama_cpp.repository' "${LOCK_PATH}")"
LLAMA_CUDA_ARCH="$(jq -er '.components.llama_cpp.cuda_architectures' "${LOCK_PATH}")"
LLAMA_TARGET="/usr/local/lib/llama.cpp-${LLAMA_RELEASE}-offline"
LLAMA_LINK="/usr/local/lib/llama.cpp"
NVCC_PATH="${CUDACXX:-/usr/local/cuda/bin/nvcc}"

if [[ ! "${LLAMA_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Commit llama.cpp trong components.lock.json khong hop le" >&2
  exit 2
fi
if [[ ! -x "${NVCC_PATH}" ]]; then
  echo "Khong tim thay nvcc tai ${NVCC_PATH}" >&2
  exit 2
fi

if [[ -x "${LLAMA_TARGET}/llama-server" ]]; then
  if "${LLAMA_TARGET}/llama-server" --version 2>&1 | grep -Fq "${LLAMA_COMMIT:0:7}"; then
    ln -sfn "${LLAMA_TARGET}" "${LLAMA_LINK}"
    exit 0
  fi
  echo "Thu muc llama.cpp dich da ton tai nhung khong dung commit da khoa: ${LLAMA_TARGET}" >&2
  exit 2
fi

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf -- "${TEMP_DIR}"' EXIT
SOURCE_DIR="${TEMP_DIR}/source"
BUILD_DIR="${TEMP_DIR}/build"
STAGE_DIR="${TEMP_DIR}/runtime"

git init -q "${SOURCE_DIR}"
git -C "${SOURCE_DIR}" remote add origin "${LLAMA_REPOSITORY}"
git -C "${SOURCE_DIR}" fetch --depth 1 origin "${LLAMA_COMMIT}"
git -C "${SOURCE_DIR}" checkout -q --detach FETCH_HEAD
if [[ "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" != "${LLAMA_COMMIT}" ]]; then
  echo "Source llama.cpp khong dung commit da khoa" >&2
  exit 2
fi

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
  -DCMAKE_CUDA_COMPILER="${NVCC_PATH}" \
  -DCMAKE_CUDA_ARCHITECTURES="${LLAMA_CUDA_ARCH}" \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON
cmake --build "${BUILD_DIR}" --target llama-server llama-cli --parallel "${DUB_BUILD_JOBS:-4}"

install -d -m 0755 "${STAGE_DIR}"
cp -a "${BUILD_DIR}/bin/." "${STAGE_DIR}/"
if ! LD_LIBRARY_PATH="${STAGE_DIR}" "${STAGE_DIR}/llama-server" --version 2>&1 \
    | grep -Fq "${LLAMA_COMMIT:0:7}"; then
  echo "Binary llama-server vua build khong dung commit da khoa" >&2
  exit 2
fi
chown -R root:root "${STAGE_DIR}"
chmod -R a-w "${STAGE_DIR}"
mv "${STAGE_DIR}" "${LLAMA_TARGET}"
ln -sfn "${LLAMA_TARGET}" "${LLAMA_LINK}"
