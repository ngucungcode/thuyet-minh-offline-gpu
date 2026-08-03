#!/usr/bin/env bash

set -Eeuo pipefail

main() {
INSTALLER_VERSION="0.3.0"
DEFAULT_REPOSITORY_URL="https://github.com/ngucungcode/thuyet-minh-offline-gpu.git"
REPOSITORY_URL="${DUB_REPOSITORY_URL:-${DEFAULT_REPOSITORY_URL}}"
SOURCE_REF="v${INSTALLER_VERSION}"
INSTALL_DIR=""
DATA_DIR=""
DATA_DIR_EXPLICIT=false
MODEL_PROFILE="auto"
START_STACK=true
AUTOSTART_MODE="auto"
ACCEPTANCE_MODE="basic"
ASSUME_YES=false
DRY_RUN=false
MIGRATE_EXISTING=false
UPGRADE_EXISTING=false
COMPATIBLE_UPGRADE_FROM="0.2.0 0.2.1 0.2.2 0.2.3 0.2.4"
MIGRATED_RUNTIME_REUSABLE=false
MIGRATION_BACKUP_PATH=""
MIGRATION_ACTIVE=false
SYSTEMD_ENABLE_PENDING=false

# BASH_SOURCE is absent when the installer is streamed through `bash -s`.
SCRIPT_SOURCE="${BASH_SOURCE[0]-}"
SCRIPT_PATH=""
if [[ -n "${SCRIPT_SOURCE}" ]]; then
  SCRIPT_PATH="$(readlink -f -- "${SCRIPT_SOURCE}" 2>/dev/null || true)"
fi

log() {
  printf '[thuyet-minh] %s\n' "$*"
}

die() {
  printf '[thuyet-minh] Lỗi: %s\n' "$*" >&2
  exit 2
}

installer_stack_healthy() {
  local status_output
  local program
  status_output="$("${PROJECT_ROOT}/scripts/native-stack.sh" status 2>/dev/null)" \
    || return 1
  for program in api prowlarr qbittorrent worker; do
    printf '%s\n' "${status_output}" \
      | grep -Eq "^${program}[[:space:]]+RUNNING[[:space:]]" || return 1
  done
  ! printf '%s\n' "${status_output}" \
    | grep -Eq ' (BACKOFF|EXITED|FATAL|STOPPED|UNKNOWN) '
}

wait_for_installer_stack() {
  local attempt
  for attempt in {1..30}; do
    if installer_stack_healthy; then
      return 0
    fi
    sleep 1
  done
  return 1
}

installer_exit_handler() {
  local installer_status=$?
  trap - EXIT INT TERM HUP
  if [[ "${MIGRATION_ACTIVE}" == true ]]; then
    MIGRATION_ACTIVE=false
    MIGRATION_SIGNAL_ARMED=false
    log "Installer lỗi sau khi đổi source; đang rollback source, dữ liệu và stack cũ"
    if ! migration_rollback_switch \
      "${INSTALL_DIR}" "${MIGRATION_BACKUP_PATH}" \
      "${MIGRATION_FAILED_SOURCE_PATH}" "${MIGRATION_OLD_STACK_MODE}" \
      true "${MIGRATION_MOVED_ITEMS[@]}"; then
      installer_status=1
    else
      migration_clear_journal || installer_status=1
    fi
  fi
  exit "${installer_status}"
}

arm_source_transaction() {
  MIGRATION_ACTIVE=true
  trap installer_exit_handler EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM HUP
}

usage() {
  cat <<'EOF'
Cài Thuyết Minh Offline GPU trên Ubuntu 22.04 + NVIDIA GPU.

Cách dùng:
  sudo ./install.sh [tùy chọn]
  curl -fsSL URL/install.sh | sudo bash -s -- [tùy chọn]

Tùy chọn:
  --repo URL                 Repository Git (dùng khi chạy qua curl)
  --ref REF                  Branch, tag hoặc commit; mặc định tag của installer
  --install-dir PATH         Mặc định /workspace/thuyet-minh-offline hoặc /opt/...
  --data-dir PATH            Mặc định <install-dir>/var
  --profile PROFILE          auto, maximum, balanced, minimal hoặc none
  --start | --no-start       Khởi động stack sau cài; mặc định --start
  --autostart MODE           auto, systemd, provider hoặc none
  --acceptance MODE          basic, full hoặc none; mặc định basic
  --migrate-existing         Nâng cấp deployment cũ hợp lệ; giữ dữ liệu và tạo backup
  --upgrade-existing         Nâng cấp Git deployment từ release tương thích; có rollback
  --yes                      Xác nhận tải model và chạy không tương tác
  --dry-run                  Chỉ kiểm tra và in kế hoạch
  --help                     Hiển thị trợ giúp
EOF
}

while (($#)); do
  case "$1" in
    --repo)
      [[ $# -ge 2 ]] || die "--repo cần một URL"
      REPOSITORY_URL="$2"
      shift 2
      ;;
    --ref)
      [[ $# -ge 2 ]] || die "--ref cần một giá trị"
      SOURCE_REF="$2"
      shift 2
      ;;
    --install-dir)
      [[ $# -ge 2 ]] || die "--install-dir cần một đường dẫn"
      INSTALL_DIR="$2"
      shift 2
      ;;
    --data-dir)
      [[ $# -ge 2 ]] || die "--data-dir cần một đường dẫn"
      DATA_DIR="$2"
      DATA_DIR_EXPLICIT=true
      shift 2
      ;;
    --profile)
      [[ $# -ge 2 ]] || die "--profile cần một giá trị"
      MODEL_PROFILE="$2"
      shift 2
      ;;
    --start)
      START_STACK=true
      shift
      ;;
    --no-start)
      START_STACK=false
      shift
      ;;
    --autostart)
      [[ $# -ge 2 ]] || die "--autostart cần một giá trị"
      AUTOSTART_MODE="$2"
      shift 2
      ;;
    --acceptance)
      [[ $# -ge 2 ]] || die "--acceptance cần một giá trị"
      ACCEPTANCE_MODE="$2"
      shift 2
      ;;
    --migrate-existing)
      MIGRATE_EXISTING=true
      shift
      ;;
    --upgrade-existing)
      UPGRADE_EXISTING=true
      shift
      ;;
    --yes)
      ASSUME_YES=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "Tùy chọn không hợp lệ: $1"
      ;;
  esac
done

case "${MODEL_PROFILE}" in
  auto|maximum|balanced|minimal|none) ;;
  *) die "Profile phải là auto, maximum, balanced, minimal hoặc none" ;;
esac
case "${AUTOSTART_MODE}" in
  auto|systemd|provider|none) ;;
  *) die "Autostart phải là auto, systemd, provider hoặc none" ;;
esac
case "${ACCEPTANCE_MODE}" in
  basic|full|none) ;;
  *) die "Acceptance phải là basic, full hoặc none" ;;
esac

if [[ -v DUB_NATIVE_ROOT ]]; then
  die "Không truyền DUB_NATIVE_ROOT qua environment; hãy dùng --data-dir"
fi

[[ "${EUID}" -eq 0 ]] || die "Trình cài phải chạy bằng root (sudo)"

SCRIPT_ROOT=""
if [[ -n "${SCRIPT_PATH}" && -f "${SCRIPT_PATH}" ]]; then
  candidate_root="$(CDPATH= cd -- "$(dirname -- "${SCRIPT_PATH}")" && pwd)"
  if [[ -f "${candidate_root}/pyproject.toml" && -f "${candidate_root}/config/models.lock.json" ]]; then
    SCRIPT_ROOT="${candidate_root}"
  fi
fi

if [[ -z "${INSTALL_DIR}" ]]; then
  if [[ -d /workspace ]]; then
    INSTALL_DIR="/workspace/thuyet-minh-offline"
  else
    INSTALL_DIR="/opt/thuyet-minh-offline"
  fi
fi
if [[ -n "${SCRIPT_ROOT}" ]]; then
  INSTALL_DIR="${SCRIPT_ROOT}"
fi
if [[ "${MIGRATE_EXISTING}" == true && -n "${SCRIPT_ROOT}" ]]; then
  die "--migrate-existing chỉ dùng với installer chạy qua curl; không chạy từ source đang hoạt động"
fi
if [[ "${UPGRADE_EXISTING}" == true && -n "${SCRIPT_ROOT}" ]]; then
  die "--upgrade-existing chỉ dùng với installer chạy qua curl; không chạy từ source đang hoạt động"
fi
if [[ "${MIGRATE_EXISTING}" == true && "${UPGRADE_EXISTING}" == true ]]; then
  die "Không dùng đồng thời --migrate-existing và --upgrade-existing"
fi
if [[ -z "${DATA_DIR}" ]]; then
  DATA_DIR="${INSTALL_DIR}/var"
fi

for selected_path in "${INSTALL_DIR}" "${DATA_DIR}"; do
  [[ "${selected_path}" == /* ]] || die "Đường dẫn phải là tuyệt đối: ${selected_path}"
  [[ "${selected_path}" != *$'\n'* && "${selected_path}" != *$'\r'* ]] \
    || die "Đường dẫn chứa ký tự xuống dòng"
done
INSTALL_DIR="$(readlink -m -- "${INSTALL_DIR}")"
DATA_DIR="$(readlink -m -- "${DATA_DIR}")"
for selected_path in "${INSTALL_DIR}" "${DATA_DIR}"; do
  [[ "${selected_path}" != "/" ]] || die "Không được dùng thư mục gốc /"
done

pending_migration_journal="${INSTALL_DIR}.migration-state.json"
if [[ -e "${pending_migration_journal}" || -L "${pending_migration_journal}" ]]; then
  die "Phát hiện migration dang dở tại ${pending_migration_journal}. Installer dừng fail-closed; không xóa journal hoặc source/backup trước khi phục hồi"
fi

if [[ ! -r /etc/os-release ]]; then
  die "Không đọc được /etc/os-release"
fi
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "22.04" ]] \
  || die "Hiện chỉ hỗ trợ Ubuntu 22.04; máy này là ${PRETTY_NAME:-không xác định}"
[[ "$(uname -m)" == "x86_64" ]] || die "Chỉ hỗ trợ kiến trúc x86_64"

ram_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
[[ "${ram_kib}" =~ ^[0-9]+$ && "${ram_kib}" -ge 16777216 ]] \
  || die "Cần tối thiểu 16 GiB RAM"
command -v python3 >/dev/null || die "Không tìm thấy Python 3"
python3 - <<'PY'
import sys
if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(f"Cần Python 3.11 hoặc 3.12, hiện có {sys.version.split()[0]}")
PY
command -v nvidia-smi >/dev/null || die "Không tìm thấy NVIDIA driver/nvidia-smi"
[[ -x /usr/local/cuda/bin/nvcc ]] || die "Không tìm thấy CUDA toolkit tại /usr/local/cuda"

gpu_report="$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits)" \
  || die "Không đọc được thông tin GPU"
gpu_name="$(printf '%s\n' "${gpu_report}" | head -n 1 | cut -d, -f1 | xargs)"
vram_mib="$(printf '%s\n' "${gpu_report}" | awk -F, 'NR == 1 {gsub(/ /, "", $2); print int($2)}')"
compute_cap="$(printf '%s\n' "${gpu_report}" | awk -F, 'NR == 1 {gsub(/ /, "", $3); print $3}')"
[[ "${vram_mib}" =~ ^[0-9]+$ ]] || die "VRAM GPU không hợp lệ"
[[ "${compute_cap}" == "8.6" ]] \
  || die "Bản native hiện khóa CUDA sm_86; GPU có compute capability ${compute_cap}"
python3 - <<'PY'
try:
    import torch
except Exception as exc:
    raise SystemExit(f"Không import được PyTorch GPU từ image nhà cung cấp: {exc}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch không thấy CUDA GPU")
PY

if [[ "${MODEL_PROFILE}" == "auto" ]]; then
  if ((vram_mib >= 22528)); then
    MODEL_PROFILE="maximum"
  elif ((vram_mib >= 8192)); then
    MODEL_PROFILE="balanced"
  else
    MODEL_PROFILE="minimal"
  fi
fi
if [[ "${MODEL_PROFILE}" == "maximum" && "${vram_mib}" -lt 22528 ]]; then
  die "Profile maximum cần ít nhất 22 GiB VRAM; hiện có ${vram_mib} MiB"
fi
if [[ "${MODEL_PROFILE}" != "none" && "${vram_mib}" -lt 6144 ]]; then
  die "Pipeline TIGER-DnR cần tối thiểu 6 GiB VRAM"
fi

case "${MODEL_PROFILE}" in
  maximum) required_disk_gib=55 ;;
  balanced) required_disk_gib=35 ;;
  minimal) required_disk_gib=25 ;;
  none) required_disk_gib=20 ;;
esac
probe_path="${INSTALL_DIR}"
while [[ ! -e "${probe_path}" ]]; do
  probe_path="$(dirname -- "${probe_path}")"
done
free_kib="$(df -Pk -- "${probe_path}" | awk 'NR == 2 {print $4}')"
required_kib=$((required_disk_gib * 1024 * 1024))
[[ "${free_kib}" =~ ^[0-9]+$ && "${free_kib}" -ge "${required_kib}" ]] \
  || die "Profile ${MODEL_PROFILE} cần ít nhất ${required_disk_gib} GiB trống"

log "Installer ${INSTALLER_VERSION}"
log "GPU: ${gpu_name}, ${vram_mib} MiB VRAM, sm_${compute_cap/./}"
log "Source: ${REPOSITORY_URL}@${SOURCE_REF}"
log "Cài tại: ${INSTALL_DIR}"
log "Dữ liệu: ${DATA_DIR}"
log "Profile: ${MODEL_PROFILE}; acceptance: ${ACCEPTANCE_MODE}"

if [[ "${DRY_RUN}" == true ]]; then
  log "Dry-run đạt; chưa thay đổi hệ thống"
  exit 0
fi

if [[ "${MODEL_PROFILE}" != "none" && "${ASSUME_YES}" != true ]]; then
  if exec {prompt_fd}<>/dev/tty 2>/dev/null; then
    if ! read -r -u "${prompt_fd}" -p \
      "Tiếp tục tải model và cài runtime? [y/N] " answer; then
      exec {prompt_fd}>&-
      die "Không đọc được xác nhận; chạy không tương tác cần cờ --yes"
    fi
    exec {prompt_fd}>&-
    [[ "${answer}" =~ ^[Yy]$ ]] || die "Đã hủy cài đặt"
  else
    die "Chạy không tương tác cần cờ --yes"
  fi
fi

command -v flock >/dev/null || die "Không tìm thấy flock"
install -d -m 0755 /run/lock
installer_lock_path=/run/lock/thuyet-minh-offline-install.lock
exec 9>"${installer_lock_path}"
if ! flock -n 9; then
  lock_owner=""
  if command -v lslocks >/dev/null; then
    lock_owner="$(
      lslocks --noheadings --raw --output PID,COMMAND,PATH 2>/dev/null \
        | awk -v path="${installer_lock_path}" \
          '$3 == path {print "PID " $1 " (" $2 ")"; exit}' \
        || true
    )"
  fi
  if [[ -n "${lock_owner}" ]]; then
    die "Một trình cài khác đang chạy hoặc tiến trình cũ đang giữ khóa: ${lock_owner}. Không xóa file lock"
  fi
  die "Một trình cài khác đang chạy hoặc tiến trình cũ đang giữ khóa. Không xóa file lock"
fi

if [[ -z "${SCRIPT_ROOT}" ]]; then
  if ! command -v git >/dev/null || ! command -v curl >/dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y ca-certificates curl git
  fi
  is_git_worktree() {
    git -C "$1" rev-parse --is-inside-work-tree >/dev/null 2>&1
  }
  fetch_ref_into() {
    local destination="$1"
    git init -q -- "${destination}" || return
    git -C "${destination}" remote add origin "${REPOSITORY_URL}" || return
    git -C "${destination}" fetch --depth 1 origin "${SOURCE_REF}" || return
    git -C "${destination}" checkout -q --detach FETCH_HEAD || return
  }

  if [[ ! -e "${INSTALL_DIR}" ]]; then
    fetch_ref_into "${INSTALL_DIR}" \
      || die "Không thể lấy source ${REPOSITORY_URL}@${SOURCE_REF}"
  elif is_git_worktree "${INSTALL_DIR}"; then
    current_origin="$(git -C "${INSTALL_DIR}" remote get-url origin)"
    current_commit="$(git -C "${INSTALL_DIR}" rev-parse HEAD)"
    normalize_url() { printf '%s' "$1" | sed -E 's#\.git$##'; }
    [[ "$(normalize_url "${current_origin}")" == "$(normalize_url "${REPOSITORY_URL}")" ]] \
      || die "Repository hiện có không đúng origin: ${current_origin}"
    [[ -z "$(git -C "${INSTALL_DIR}" status --porcelain)" ]] \
      || die "Repository có thay đổi cục bộ; không tự ghi đè"
    git -C "${INSTALL_DIR}" fetch --depth 1 origin "${SOURCE_REF}"
    target_commit="$(git -C "${INSTALL_DIR}" rev-parse FETCH_HEAD)"
    if [[ "${current_commit}" == "${target_commit}" ]]; then
      git -C "${INSTALL_DIR}" checkout --detach FETCH_HEAD
    else
      [[ "${UPGRADE_EXISTING}" == true ]] \
        || die "Không tự nâng cấp in-place giữa hai release; chạy lại với --upgrade-existing"
      install_parent="$(dirname -- "${INSTALL_DIR}")"
      install_name="$(basename -- "${INSTALL_DIR}")"
      staging_dir="$(mktemp -d "${install_parent}/.${install_name}.upgrade.XXXXXXXX")"
      chmod 0700 "${staging_dir}"
      if ! fetch_ref_into "${staging_dir}"; then
        rm -rf -- "${staging_dir}"
        die "Không thể lấy source nâng cấp ${REPOSITORY_URL}@${SOURCE_REF}"
      fi
      upgrade_helper="${staging_dir}/installer/migrate-legacy.sh"
      if [[ ! -f "${upgrade_helper}" || -L "${upgrade_helper}" ]] \
        || ! git -C "${staging_dir}" diff --quiet \
        || [[ -n "$(git -C "${staging_dir}" status --porcelain)" ]] \
        || ! git -C "${staging_dir}" fsck --no-dangling >/dev/null \
        || ! git -C "${staging_dir}" ls-files --error-unmatch \
          installer/migrate-legacy.sh >/dev/null; then
        rm -rf -- "${staging_dir}"
        die "Source nâng cấp không có transaction helper hợp lệ"
      fi
      # shellcheck disable=SC1090
      source "${upgrade_helper}"
      if ! migrate_git_release_upgrade \
        "${INSTALL_DIR}" "${staging_dir}" "${DATA_DIR}" \
        "${DATA_DIR_EXPLICIT}" "${INSTALLER_VERSION}" \
        "${COMPATIBLE_UPGRADE_FROM}"; then
        if [[ -d "${staging_dir}" ]]; then
          rm -rf -- "${staging_dir}"
        fi
        die "Nâng cấp Git deployment thất bại; source hiện tại chưa bị thay đổi hoặc đã được phục hồi"
      fi
      DATA_DIR="${MIGRATION_EFFECTIVE_DATA_DIR}"
      arm_source_transaction
      if [[ "${MIGRATED_RUNTIME_REUSABLE}" != true ]]; then
        die "Runtime release cũ không tương thích source mới"
      fi
      log "Đã chuyển source atomically; backup release cũ: ${MIGRATION_BACKUP_PATH}"
    fi
  elif [[ -n "$(find "${INSTALL_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    if [[ "${MIGRATE_EXISTING}" != true ]]; then
      die "Thư mục cài là deployment cũ không có Git. Chạy lại với --migrate-existing để giữ .env.native, model và dữ liệu trong một bản nâng cấp có backup"
    fi

    install_parent="$(dirname -- "${INSTALL_DIR}")"
    install_name="$(basename -- "${INSTALL_DIR}")"
    staging_dir="$(mktemp -d "${install_parent}/.${install_name}.incoming.XXXXXXXX")"
    chmod 0700 "${staging_dir}"
    if ! fetch_ref_into "${staging_dir}"; then
      rm -rf -- "${staging_dir}"
      die "Không thể lấy source mới; deployment cũ chưa bị thay đổi"
    fi
    migration_helper="${staging_dir}/installer/migrate-legacy.sh"
    if [[ ! -f "${migration_helper}" || -L "${migration_helper}" ]] \
      || ! git -C "${staging_dir}" diff --quiet \
      || [[ -n "$(git -C "${staging_dir}" status --porcelain)" ]] \
      || ! git -C "${staging_dir}" fsck --no-dangling >/dev/null \
      || ! git -C "${staging_dir}" ls-files --error-unmatch \
        installer/migrate-legacy.sh >/dev/null; then
      rm -rf -- "${staging_dir}"
      die "Source mới không có bộ nâng cấp legacy hợp lệ; deployment cũ chưa bị thay đổi"
    fi
    # shellcheck disable=SC1090
    source "${migration_helper}"
    if ! migrate_legacy_install \
      "${INSTALL_DIR}" "${staging_dir}" "${DATA_DIR}" "${DATA_DIR_EXPLICIT}"; then
      if [[ -d "${staging_dir}" ]]; then
        rm -rf -- "${staging_dir}"
      fi
      die "Nâng cấp deployment cũ thất bại; xem thông báo rollback ở trên"
    fi
    DATA_DIR="${MIGRATION_EFFECTIVE_DATA_DIR}"
    arm_source_transaction
    if [[ "${MIGRATED_RUNTIME_REUSABLE}" != true ]]; then
      die "Runtime legacy khác source mới; đã từ chối bootstrap trong transaction migration"
    fi
    log "Đã giữ nguyên dữ liệu; backup source cũ: ${MIGRATION_BACKUP_PATH}"
  else
    fetch_ref_into "${INSTALL_DIR}" \
      || die "Không thể lấy source ${REPOSITORY_URL}@${SOURCE_REF}"
  fi
fi

PROJECT_ROOT="${INSTALL_DIR}"
[[ -f "${PROJECT_ROOT}/pyproject.toml" && -x "${PROJECT_ROOT}/scripts/native-bootstrap.sh" ]] \
  || die "Source không đầy đủ tại ${PROJECT_ROOT}"

ENV_FILE="${PROJECT_ROOT}/.env.native"
if [[ ! -e "${ENV_FILE}" ]]; then
  temporary_env="$(mktemp "${PROJECT_ROOT}/.env.native.XXXXXX")"
  cp "${PROJECT_ROOT}/.env.native.example" "${temporary_env}"
  printf '\nDUB_NATIVE_ROOT=%s\n' "${DATA_DIR}" >>"${temporary_env}"
  install -m 0600 -o root -g root "${temporary_env}" "${ENV_FILE}"
  rm -f -- "${temporary_env}"
else
  configured_data="$(sed -n 's/^DUB_NATIVE_ROOT=//p' "${ENV_FILE}" | tail -n 1)"
  if [[ -n "${configured_data}" && "${DATA_DIR_EXPLICIT}" != true \
    && -n "${MIGRATION_BACKUP_PATH}" ]]; then
    [[ "${configured_data}" == /* ]] \
      || die "DUB_NATIVE_ROOT trong .env.native phải là đường dẫn tuyệt đối"
    DATA_DIR="${configured_data}"
  fi
  if [[ -n "${configured_data}" && "${configured_data}" != "${DATA_DIR}" ]]; then
    die ".env.native đã dùng DUB_NATIVE_ROOT=${configured_data}; không tự ghi đè"
  fi
  chmod 0600 "${ENV_FILE}"
fi

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/native-common.sh"
data_root_preparer="${PROJECT_ROOT}/installer/prepare-data-root.sh"
[[ -x "${data_root_preparer}" ]] \
  || die "Source thiếu helper chuẩn bị data root"
data_root_state="$("${data_root_preparer}" "${DUB_NATIVE_ROOT}")" \
  || die "Không thể chuẩn bị data root ${DUB_NATIVE_ROOT}"
[[ "${data_root_state}" == "existing" || "${data_root_state}" == "created" ]] \
  || die "Helper data root trả trạng thái không hợp lệ"
state_file="${DUB_NATIVE_ROOT}/install-state.json"
fingerprint="$(
  cd -- "${PROJECT_ROOT}"
  {
    sha256sum \
      pyproject.toml \
      .env.native.example \
      config/models.lock.json \
      scripts/native-bootstrap.sh \
      scripts/native-common.sh \
      scripts/install-llama-cpp.sh \
      scripts/native-stack.sh \
      scripts/vieneu-offline.py
    find native -maxdepth 1 -type f -print0 \
      | sort -z \
      | xargs -0 sha256sum
  } | sha256sum | awk '{print $1}'
)"

bootstrap_required=true
if [[ "${MIGRATED_RUNTIME_REUSABLE}" == true && -x "${DUB_VENV_DIR}/bin/python" ]]; then
  if "${DUB_VENV_DIR}/bin/python" -c 'import dub_server.cli' >/dev/null 2>&1; then
    bootstrap_required=false
    log "Runtime legacy khớp source mới; bỏ qua bootstrap nặng"
  elif [[ "${MIGRATION_ACTIVE}" == true ]]; then
    die "Runtime legacy không import được CLI mới; không bootstrap trong transaction migration"
  fi
elif [[ -x "${DUB_VENV_DIR}/bin/python" && -r "${state_file}" ]] \
  && command -v jq >/dev/null \
  && [[ "$(jq -r '.bootstrap_fingerprint // ""' "${state_file}")" == "${fingerprint}" ]]; then
  if "${PROJECT_ROOT}/scripts/native-preflight.sh" >/dev/null 2>&1; then
    bootstrap_required=false
  fi
fi
if [[ "${bootstrap_required}" == true ]]; then
  log "Bootstrap runtime native và chạy unit test"
  "${PROJECT_ROOT}/scripts/native-bootstrap.sh"
else
  log "Runtime không đổi; bỏ qua bootstrap nặng"
fi

id "${DUB_NATIVE_USER}" >/dev/null 2>&1 \
  || die "Không tồn tại runtime user ${DUB_NATIVE_USER}"
runtime_group="$(id -gn "${DUB_NATIVE_USER}")" \
  || die "Không xác định được primary group của ${DUB_NATIVE_USER}"
command -v runuser >/dev/null \
  || die "Không tìm thấy runuser"
if [[ "${data_root_state}" == "created" ]]; then
  install -d -m 0750 -o root -g "${runtime_group}" -- "${DUB_NATIVE_ROOT}"
fi
runuser -u "${DUB_NATIVE_USER}" -- test -x "${DUB_NATIVE_ROOT}" \
  || die "Runtime user ${DUB_NATIVE_USER} không thể truy cập ${DUB_NATIVE_ROOT}"

case "${MODEL_PROFILE}" in
  maximum)
    model_ids=(
      asr-faster-whisper-large-v3-turbo
      mt-gemma4-31b-q4
      separation-tiger-dnr
      tts-neucodec-onnx-int8
      tts-vieneu-v2
    )
    ;;
  balanced)
    model_ids=(
      asr-faster-whisper-small
      mt-gemma4-e2b-q4
      separation-tiger-dnr
      tts-neucodec-onnx-int8
      tts-vieneu-v2
    )
    ;;
  minimal)
    model_ids=(
      asr-faster-whisper-small
      mt-gemma4-e2b-q4
      separation-tiger-dnr
      tts-piper-vi-vais1000-medium
    )
    ;;
  none)
    model_ids=()
    ;;
esac

for model_id in "${model_ids[@]}"; do
  log "Cài và xác minh model ${model_id}"
  "${PROJECT_ROOT}/scripts/native-model.sh" install "${model_id}"
  "${PROJECT_ROOT}/scripts/native-model.sh" verify "${model_id}"
done

wrapper_target="${PROJECT_ROOT}/scripts/dub-wrapper.sh"
wrapper_link="/usr/local/bin/dub"
if [[ -e "${wrapper_link}" || -L "${wrapper_link}" ]]; then
  current_wrapper="$(readlink -f -- "${wrapper_link}" 2>/dev/null || true)"
  [[ "${current_wrapper}" == "$(readlink -f -- "${wrapper_target}")" ]] \
    || die "${wrapper_link} đã do chương trình khác quản lý"
else
  ln -s -- "${wrapper_target}" "${wrapper_link}"
fi

if [[ "${MIGRATION_ACTIVE}" == true ]]; then
  case "${MIGRATION_OLD_STACK_MODE}" in
    systemd|systemd-stopped)
      if [[ "${AUTOSTART_MODE}" == "auto" ]]; then
        AUTOSTART_MODE="systemd"
      elif [[ "${AUTOSTART_MODE}" != "systemd" ]]; then
        die "Migration không tự đổi control plane systemd; hãy giữ --autostart systemd"
      fi
      ;;
    native)
      if [[ "${AUTOSTART_MODE}" == "auto" ]]; then
        AUTOSTART_MODE="provider"
      elif [[ "${AUTOSTART_MODE}" == "systemd" ]]; then
        die "Migration không tự đổi control plane native sang systemd"
      fi
      ;;
    none)
      if [[ "${AUTOSTART_MODE}" == "auto" ]]; then
        AUTOSTART_MODE="provider"
      elif [[ "${AUTOSTART_MODE}" == "systemd" ]]; then
        die "Migration không tạo systemd control plane mới trong transaction"
      fi
      ;;
  esac
fi

if [[ "${AUTOSTART_MODE}" == "auto" ]]; then
  if [[ -d /run/systemd/system ]] && command -v systemctl >/dev/null; then
    AUTOSTART_MODE="systemd"
  else
    AUTOSTART_MODE="provider"
  fi
fi
if [[ "${AUTOSTART_MODE}" == "systemd" ]]; then
  [[ -d /run/systemd/system ]] || die "Máy này không chạy systemd"
  if [[ "${MIGRATION_ACTIVE}" == true \
    && "${MIGRATION_OLD_STACK_MODE}" == systemd* ]]; then
    log "Giữ nguyên systemd unit và trạng thái enable hiện có"
  else
    unit_path="/etc/systemd/system/thuyet-minh-offline.service"
    temporary_unit="$(mktemp /etc/systemd/system/.thuyet-minh-offline.XXXXXX)"
    cat >"${temporary_unit}" <<EOF
[Unit]
Description=Thuyet Minh Offline GPU
After=network-online.target

[Service]
Type=simple
ExecStart=${PROJECT_ROOT}/scripts/native-stack.sh foreground
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    chmod 0644 "${temporary_unit}"
    mv -f -- "${temporary_unit}" "${unit_path}"
    systemctl daemon-reload
    SYSTEMD_ENABLE_PENDING=true
  fi
elif [[ "${AUTOSTART_MODE}" == "provider" ]]; then
  log "Startup command cho nhà cung cấp: ${PROJECT_ROOT}/scripts/native-stack.sh foreground"
fi

if [[ "${START_STACK}" == true ]]; then
  if [[ "${AUTOSTART_MODE}" == "systemd" ]]; then
    if ! systemctl start thuyet-minh-offline.service; then
      systemctl stop thuyet-minh-offline.service >/dev/null 2>&1 || true
      die "Không thể khởi động systemd service"
    fi
    if ! wait_for_installer_stack; then
      systemctl stop thuyet-minh-offline.service >/dev/null 2>&1 || true
      die "Systemd service đã start nhưng stack chưa khỏe"
    fi
  else
    "${PROJECT_ROOT}/scripts/native-stack.sh" start 9>&-
  fi
  if [[ -s "${DUB_PROWLARR_API_KEY_FILE}" && -s "${DUB_QBITTORRENT_PASSWORD_FILE}" ]]; then
    "${PROJECT_ROOT}/scripts/native-init-services.sh"
  else
    "${PROJECT_ROOT}/scripts/native-init-services.sh" --rotate-secrets
  fi

  case "${ACCEPTANCE_MODE}" in
    basic)
      "${PROJECT_ROOT}/scripts/native-acceptance.sh"
      ;;
    full)
      "${PROJECT_ROOT}/scripts/native-acceptance.sh"
      "${PROJECT_ROOT}/scripts/native-phase2-acceptance.sh"
      "${PROJECT_ROOT}/scripts/native-phase3-acceptance.sh"
      "${PROJECT_ROOT}/scripts/native-phase4-acceptance.sh"
      ;;
    none) ;;
  esac
fi

if [[ "${SYSTEMD_ENABLE_PENDING}" == true ]]; then
  if ! systemctl enable thuyet-minh-offline.service; then
    systemctl stop thuyet-minh-offline.service >/dev/null 2>&1 || true
    die "Không thể enable systemd service"
  fi
fi

install -d -m 0750 -o "${DUB_NATIVE_USER}" -g "${DUB_NATIVE_USER}" \
  "${DUB_NATIVE_ROOT}/reports"
"${DUB_VENV_DIR}/bin/python" "${PROJECT_ROOT}/scripts/generate-sbom.py" \
  --models-lock "${PROJECT_ROOT}/config/models.lock.json" \
  --native-lock "${PROJECT_ROOT}/native/components.lock.json" \
  --output "${DUB_NATIVE_ROOT}/reports/sbom.cdx.json"

commit="source-archive"
if [[ -d "${PROJECT_ROOT}/.git" ]]; then
  commit="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
fi
models_json="$(printf '%s\n' "${model_ids[@]}" | jq -R . | jq -s .)"
temporary_state="$(mktemp "${DUB_NATIVE_ROOT}/.install-state.XXXXXX")"
jq -n \
  --arg installer_version "${INSTALLER_VERSION}" \
  --arg commit "${commit}" \
  --arg profile "${MODEL_PROFILE}" \
  --arg fingerprint "${fingerprint}" \
  --arg installed_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg migration_backup "${MIGRATION_BACKUP_PATH}" \
  --argjson models "${models_json}" \
  '{schema_version:1, installer_version:$installer_version, commit:$commit,
    profile:$profile, models:$models, bootstrap_fingerprint:$fingerprint,
    installed_at:$installed_at,
    migration_backup:(if $migration_backup == "" then null else $migration_backup end)}' >"${temporary_state}"
install -m 0640 -o "${DUB_NATIVE_USER}" -g "${DUB_NATIVE_USER}" \
  "${temporary_state}" "${state_file}"
rm -f -- "${temporary_state}"

if [[ "${MIGRATION_ACTIVE}" == true ]]; then
  migration_verify_data_root_identity \
    || die "Metadata data root đã thay đổi trong transaction migration"
  migration_write_journal committed "${MIGRATION_MOVED_ITEMS[@]}"
  migration_clear_journal
  MIGRATION_ACTIVE=false
  MIGRATION_SIGNAL_ARMED=false
  trap - EXIT INT TERM HUP
fi

log "Cài đặt hoàn tất"
log "Kiểm tra: dub doctor && dub stack status"
log "Bắt đầu: dub search \"Tên phim\" --year 2024"
}

main "$@"
