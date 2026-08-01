#!/usr/bin/env bash

# This file is sourced by install.sh after the requested Git ref has been
# fetched into a private staging directory. It intentionally does not source
# configuration from the legacy deployment before validating the source tree.

MIGRATION_BACKUP_PATH=""
MIGRATION_FAILED_SOURCE_PATH=""
MIGRATION_EFFECTIVE_DATA_DIR=""
MIGRATION_OLD_STACK_MODE="none"
MIGRATION_MOVED_ITEMS=()
MIGRATED_RUNTIME_REUSABLE=false
MIGRATION_SIGNAL_ARMED=false
MIGRATION_SIGNAL_TARGET=""
MIGRATION_SIGNAL_BACKUP=""
MIGRATION_SIGNAL_FAILED=""
MIGRATION_SIGNAL_STACK_MODE="none"
MIGRATION_SIGNAL_EXPECTED_ITEMS=()
MIGRATION_JOURNAL_PATH=""

migration_log() {
  printf '[thuyet-minh] %s\n' "$*"
}

migration_error() {
  printf '[thuyet-minh] Lỗi migration: %s\n' "$*" >&2
}

migration_path_exists() {
  [[ -e "$1" || -L "$1" ]]
}

migration_write_journal() {
  local phase="$1"
  shift
  python3 - \
    "${MIGRATION_JOURNAL_PATH}" "${phase}" \
    "${MIGRATION_SIGNAL_TARGET}" "${MIGRATION_SIGNAL_BACKUP}" \
    "${MIGRATION_SIGNAL_FAILED}" "${MIGRATION_EFFECTIVE_DATA_DIR}" \
    "${MIGRATION_SIGNAL_STACK_MODE}" "$@" <<'PY'
import json
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
document = {
    "schema_version": 1,
    "phase": sys.argv[2],
    "target": sys.argv[3],
    "backup": sys.argv[4],
    "failed_new": sys.argv[5],
    "data_root": sys.argv[6],
    "stack_mode": sys.argv[7],
    "moved_items": sys.argv[8:],
}
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
payload = (json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n").encode()
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
finally:
    if temporary.exists():
        temporary.unlink()
PY
}

migration_clear_journal() {
  [[ -n "${MIGRATION_JOURNAL_PATH}" ]] || return 0
  rm -f -- "${MIGRATION_JOURNAL_PATH}" || return 1
  python3 - "$(dirname -- "${MIGRATION_JOURNAL_PATH}")" <<'PY'
import os
from pathlib import Path
import sys

directory = os.open(Path(sys.argv[1]), os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
}

migration_validate_project_tree() {
  local project_root="$1"
  python3 - "${project_root}" <<'PY'
import json
import os
from pathlib import Path
import sys
import tomllib

root = Path(sys.argv[1])
if not root.is_absolute() or root.is_symlink() or not root.is_dir():
    raise SystemExit("project root không phải thư mục tuyệt đối, thực")

required_files = (
    "pyproject.toml",
    ".env.native.example",
    "LICENSE",
    "config/models.lock.json",
    "native/components.lock.json",
    "native/supervisord.conf",
    "src/dub_server/__init__.py",
    "src/dub_server/cli.py",
    "scripts/native-common.sh",
    "scripts/native-bootstrap.sh",
    "scripts/install-llama-cpp.sh",
    "scripts/native-stack.sh",
    "scripts/vieneu-offline.py",
)
for relative in required_files:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"thiếu marker source hợp lệ: {relative}")

with (root / "pyproject.toml").open("rb") as handle:
    project = tomllib.load(handle).get("project", {})
if project.get("name") != "thuyet-minh-offline-gpu":
    raise SystemExit("project.name không khớp")
if project.get("scripts", {}).get("dub") != "dub_server.cli:app":
    raise SystemExit("entrypoint dub không khớp")

with (root / "config/models.lock.json").open(encoding="utf-8") as handle:
    models_document = json.load(handle)
model_ids = {
    item.get("id") for item in models_document.get("models", [])
    if isinstance(item, dict)
}

required_models = {
    "asr-faster-whisper-small",
    "mt-gemma4-e2b-q4",
    "separation-tiger-dnr",
    "tts-piper-vi-vais1000-medium",
}
if models_document.get("schema_version") != 1 or not required_models <= model_ids:
    raise SystemExit("models.lock.json không phải catalog của dự án")

with (root / "native/components.lock.json").open(encoding="utf-8") as handle:
    components_document = json.load(handle)
components = components_document.get("components", {})
if components_document.get("schema_version") != 1 or not {
    "llama_cpp", "prowlarr", "qbittorrent_nox"
} <= set(components):
    raise SystemExit("components.lock.json không phải catalog của dự án")

for relative in ("scripts/native-bootstrap.sh", "scripts/native-stack.sh"):
    if not os.access(root / relative, os.X_OK):
        raise SystemExit(f"script không executable: {relative}")
PY
}

migration_read_configured_data_root() {
  local env_path="$1"
  python3 - "${env_path}" <<'PY'
from pathlib import Path
import os
import re
import sys

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(0)

matches = []
for raw_line in path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("DUB_NATIVE_ROOT="):
        matches.append(line.removeprefix("DUB_NATIVE_ROOT="))

if len(matches) > 1:
    raise SystemExit(".env.native khai báo DUB_NATIVE_ROOT nhiều lần")
if not matches:
    raise SystemExit(0)

value = matches[0]
if not re.fullmatch(r"/[A-Za-z0-9._/+:-]+", value):
    raise SystemExit("DUB_NATIVE_ROOT phải là đường dẫn tuyệt đối literal")
if os.path.normpath(value) != value:
    raise SystemExit("DUB_NATIVE_ROOT chứa thành phần đường dẫn không chuẩn")
print(value)
PY
}

migration_runtime_fingerprint() {
  local project_root="$1"
  (
    cd -- "${project_root}" || exit 1
    {
      sha256sum \
        pyproject.toml \
        .env.native.example \
        config/models.lock.json \
        scripts/native-bootstrap.sh \
        scripts/native-common.sh \
        scripts/install-llama-cpp.sh \
        scripts/vieneu-offline.py
      find native -maxdepth 1 -type f -print0 \
        | sort -z \
        | xargs -0 sha256sum
    } | sha256sum | awk '{print $1}'
  )
}

migration_restart_stack() {
  local project_root="$1"
  local stack_mode="$2"
  case "${stack_mode}" in
    systemd) systemctl start thuyet-minh-offline.service ;;
    native) "${project_root}/scripts/native-stack.sh" start ;;
    none) return 0 ;;
    *) return 1 ;;
  esac
}

migration_abort_on_signal() {
  local signal_status="$1"
  local inferred_items=()
  local item
  local recovery_ok=true
  trap - INT TERM HUP

  if [[ "${MIGRATION_SIGNAL_ARMED}" == true ]]; then
    MIGRATION_SIGNAL_ARMED=false
    migration_error "Nhận tín hiệu dừng; đang phục hồi transaction migration"
    if migration_path_exists "${MIGRATION_SIGNAL_BACKUP}"; then
      if migration_path_exists "${MIGRATION_SIGNAL_TARGET}"; then
        for item in "${MIGRATION_SIGNAL_EXPECTED_ITEMS[@]}"; do
          if migration_path_exists "${MIGRATION_SIGNAL_TARGET}/${item}" \
            && ! migration_path_exists "${MIGRATION_SIGNAL_BACKUP}/${item}"; then
            inferred_items+=("${item}")
          fi
        done
        migration_rollback_switch \
          "${MIGRATION_SIGNAL_TARGET}" "${MIGRATION_SIGNAL_BACKUP}" \
          "${MIGRATION_SIGNAL_FAILED}" "${MIGRATION_SIGNAL_STACK_MODE}" \
          true "${inferred_items[@]}" || recovery_ok=false
      else
        mv -- "${MIGRATION_SIGNAL_BACKUP}" "${MIGRATION_SIGNAL_TARGET}" \
          || recovery_ok=false
        migration_restart_stack \
          "${MIGRATION_SIGNAL_TARGET}" "${MIGRATION_SIGNAL_STACK_MODE}" \
          || recovery_ok=false
      fi
    else
      migration_restart_stack \
        "${MIGRATION_SIGNAL_TARGET}" "${MIGRATION_SIGNAL_STACK_MODE}" \
        || recovery_ok=false
    fi
    if [[ "${recovery_ok}" == true ]]; then
      migration_clear_journal || true
    fi
  fi
  exit "${signal_status}"
}

migration_rollback_switch() {
  local legacy_root="$1"
  local backup_root="$2"
  local failed_root="$3"
  local old_stack_mode="$4"
  local new_source_active="$5"
  shift 5
  local moved_items=("$@")
  local index
  local item
  local rollback_ok=true

  if [[ "${new_source_active}" == true \
    && -x "${legacy_root}/scripts/native-stack.sh" ]]; then
    "${legacy_root}/scripts/native-stack.sh" stop >/dev/null 2>&1 || {
      migration_error "Không thể dừng stack source mới trước rollback"
      return 1
    }
  fi

  for ((index=${#moved_items[@]} - 1; index >= 0; index--)); do
    item="${moved_items[index]}"
    if migration_path_exists "${legacy_root}/${item}" \
      && ! migration_path_exists "${backup_root}/${item}"; then
      mv -- "${legacy_root}/${item}" "${backup_root}/${item}" || rollback_ok=false
    else
      rollback_ok=false
    fi
  done

  if [[ "${rollback_ok}" != true ]]; then
    migration_error "Không thể tự rollback dữ liệu; không di chuyển thêm. Source mới: ${legacy_root}; backup: ${backup_root}"
    return 1
  fi

  if [[ "${new_source_active}" == true ]]; then
    if migration_path_exists "${failed_root}" || ! mv -- "${legacy_root}" "${failed_root}"; then
      migration_error "Không thể cất source mới để rollback; backup còn tại ${backup_root}"
      return 1
    fi
  fi
  if ! mv -- "${backup_root}" "${legacy_root}"; then
    migration_error "Không thể phục hồi source cũ từ ${backup_root}"
    return 1
  fi
  migration_restart_stack "${legacy_root}" "${old_stack_mode}" \
    || migration_error "Source cũ đã phục hồi nhưng stack cần được khởi động thủ công"
  migration_error "Đã rollback source và dữ liệu cũ; source mới lỗi được giữ tại ${failed_root}"
  return 0
}

migrate_legacy_install() {
  local legacy_root="${1%/}"
  local staged_root="${2%/}"
  local requested_data_dir="${3:-${legacy_root}/var}"
  local data_dir_explicit="${4:-false}"
  local legacy_parent
  local staged_parent
  local configured_data_dir=""
  local effective_data_dir
  local backup_root
  local failed_root
  local timestamp
  local legacy_fingerprint
  local staged_fingerprint
  local stack_mode="none"
  local stack_status_running=false
  local stop_output=""
  local database_path
  local active_jobs
  local new_source_active=false
  local item
  local persistent_items=(.venv-native)
  local moved_items=()

  MIGRATION_BACKUP_PATH=""
  MIGRATION_FAILED_SOURCE_PATH=""
  MIGRATION_EFFECTIVE_DATA_DIR=""
  MIGRATION_OLD_STACK_MODE="none"
  MIGRATION_MOVED_ITEMS=()
  MIGRATED_RUNTIME_REUSABLE=false
  MIGRATION_JOURNAL_PATH="${legacy_root}.migration-state.json"

  [[ "${legacy_root}" == /* && "${staged_root}" == /* \
    && "${requested_data_dir}" == /* ]] || {
    migration_error "Đường dẫn migration và dữ liệu phải là tuyệt đối"
    return 1
  }
  case "${legacy_root}" in
    /|/workspace|/opt)
      migration_error "Từ chối migration thư mục hệ thống rộng: ${legacy_root}"
      return 1
      ;;
  esac
  [[ -d "${legacy_root}" && ! -L "${legacy_root}" \
    && "$(readlink -f -- "${legacy_root}")" == "${legacy_root}" ]] || {
    migration_error "Deployment cũ không phải đường dẫn thư mục chuẩn, thực"
    return 1
  }
  [[ -d "${staged_root}" && ! -L "${staged_root}" \
    && "$(readlink -f -- "${staged_root}")" == "${staged_root}" ]] || {
    migration_error "Source staging không phải đường dẫn thư mục chuẩn, thực"
    return 1
  }
  legacy_parent="$(readlink -f -- "$(dirname -- "${legacy_root}")")"
  staged_parent="$(readlink -f -- "$(dirname -- "${staged_root}")")"
  [[ "${legacy_parent}" == "${staged_parent}" \
    && "$(stat -c '%d' -- "${legacy_root}")" == "$(stat -c '%d' -- "${staged_root}")" ]] || {
    migration_error "Source staging phải ở cùng filesystem cha để đổi tên atomic"
    return 1
  }
  if migration_path_exists "${legacy_root}/.git" \
    || git -C "${legacy_root}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    migration_error "Deployment đích đã là Git worktree; hãy dùng luồng update thông thường"
    return 1
  fi
  migration_validate_project_tree "${legacy_root}" || {
    migration_error "Không nhận diện được deployment cũ"
    return 1
  }
  migration_validate_project_tree "${staged_root}" || {
    migration_error "Source mới không vượt qua kiểm tra marker"
    return 1
  }

  for item in .env.native var .venv-native; do
    if migration_path_exists "${staged_root}/${item}"; then
      migration_error "Source mới chứa đường dẫn persistent ngoài dự kiến: ${item}"
      return 1
    fi
  done
  if migration_path_exists "${legacy_root}/.env.native"; then
    [[ -f "${legacy_root}/.env.native" && ! -L "${legacy_root}/.env.native" ]] || {
      migration_error ".env.native phải là regular file, không phải symlink"
      return 1
    }
    [[ "$(stat -c '%s' -- "${legacy_root}/.env.native")" -le 1048576 ]] || {
      migration_error ".env.native lớn bất thường"
      return 1
    }
    configured_data_dir="$(migration_read_configured_data_root "${legacy_root}/.env.native")" || {
      migration_error "Không thể đọc DUB_NATIVE_ROOT an toàn từ .env.native"
      return 1
    }
    cp --preserve=mode,ownership,timestamps -- \
      "${legacy_root}/.env.native" "${staged_root}/.env.native" || return 1
    chmod 0600 "${staged_root}/.env.native" || return 1
    cmp -s -- "${legacy_root}/.env.native" "${staged_root}/.env.native" || {
      migration_error "Không sao chép nguyên vẹn .env.native"
      return 1
    }
  fi

  if [[ -n "${configured_data_dir}" ]]; then
    if [[ "${data_dir_explicit}" == true \
      && "${requested_data_dir}" != "${configured_data_dir}" ]]; then
      migration_error "--data-dir không khớp DUB_NATIVE_ROOT hiện có"
      return 1
    fi
    effective_data_dir="${configured_data_dir}"
  else
    effective_data_dir="${requested_data_dir}"
    if [[ "${data_dir_explicit}" == true \
      && -f "${staged_root}/.env.native" ]]; then
      printf '\nDUB_NATIVE_ROOT=%s\n' "${effective_data_dir}" \
        >>"${staged_root}/.env.native" || return 1
    fi
  fi
  if [[ "${effective_data_dir}" == "${legacy_root}/var" ]]; then
    persistent_items=(var .venv-native)
  elif [[ "${effective_data_dir}" == "${legacy_root}" \
    || "${effective_data_dir}" == "${legacy_root}/"* ]]; then
    migration_error "Chỉ tự động chuyển data nội bộ tại ${legacy_root}/var; hãy dùng data path ngoài source"
    return 1
  fi

  if command -v mountpoint >/dev/null; then
    if mountpoint -q -- "${legacy_root}"; then
      migration_error "Deployment root là mountpoint; không tự động rename"
      return 1
    fi
    for item in "${persistent_items[@]}"; do
      if [[ -d "${legacy_root}/${item}" && ! -L "${legacy_root}/${item}" ]] \
        && mountpoint -q -- "${legacy_root}/${item}"; then
        migration_error "${legacy_root}/${item} là mountpoint; không tự động migration"
        return 1
      fi
    done
  fi

  chmod --reference="${legacy_root}" -- "${staged_root}" || {
    migration_error "Không thể giữ quyền truy cập của project root"
    return 1
  }
  if [[ "$(stat -c '%u:%g' -- "${legacy_root}")" \
    != "$(stat -c '%u:%g' -- "${staged_root}")" ]]; then
    chown --reference="${legacy_root}" -- "${staged_root}" || {
      migration_error "Không thể giữ owner của project root"
      return 1
    }
  fi

  legacy_fingerprint="$(migration_runtime_fingerprint "${legacy_root}")" || {
    migration_error "Không tạo được fingerprint source cũ"
    return 1
  }
  staged_fingerprint="$(migration_runtime_fingerprint "${staged_root}")" || {
    migration_error "Không tạo được fingerprint source mới"
    return 1
  }
  if [[ "${legacy_fingerprint}" == "${staged_fingerprint}" \
    && -x "${legacy_root}/.venv-native/bin/python" ]]; then
    MIGRATED_RUNTIME_REUSABLE=true
  fi
  if migration_path_exists "${MIGRATION_JOURNAL_PATH}"; then
    migration_error "Đã có journal migration dang dở: ${MIGRATION_JOURNAL_PATH}"
    return 1
  fi

  if command -v systemctl >/dev/null \
    && systemctl is-active --quiet thuyet-minh-offline.service; then
    stack_mode="systemd"
    migration_log "Dừng systemd service cũ trước khi đổi source"
    systemctl stop thuyet-minh-offline.service || {
      migration_error "Không thể dừng systemd service cũ"
      return 1
    }
  else
    if "${legacy_root}/scripts/native-stack.sh" status >/dev/null 2>&1; then
      stack_status_running=true
    fi
    stop_output="$("${legacy_root}/scripts/native-stack.sh" stop 2>&1)" || {
      migration_error "Không thể chứng minh stack cũ đã dừng sạch"
      return 1
    }
    [[ -z "${stop_output}" ]] || printf '%s\n' "${stop_output}"
    if [[ "${stack_status_running}" == true \
      || "${stop_output}" != *"Stack native chưa chạy"* ]]; then
      stack_mode="native"
    fi
  fi

  database_path="${effective_data_dir}/state/jobs.sqlite3"
  if [[ -f "${database_path}" ]]; then
    active_jobs="$(sqlite3 -readonly "${database_path}" \
      "SELECT count(*) FROM jobs WHERE active_slot = 1 OR status = 'cancelling';")" || {
      migration_restart_stack "${legacy_root}" "${stack_mode}" || true
      migration_error "Không kiểm tra được job đang hoạt động; chưa thay đổi source"
      return 1
    }
    if [[ ! "${active_jobs}" =~ ^[0-9]+$ || "${active_jobs}" -ne 0 ]]; then
      migration_restart_stack "${legacy_root}" "${stack_mode}" || true
      migration_error "Còn job đang hoạt động; đã khởi động lại stack cũ và dừng migration"
      return 1
    fi
  fi

  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  backup_root="${legacy_root}.legacy-backup-${timestamp}-$$"
  failed_root="${legacy_root}.failed-migration-${timestamp}-$$"
  if migration_path_exists "${backup_root}" || migration_path_exists "${failed_root}"; then
    migration_error "Đường dẫn backup đã tồn tại; chưa thay đổi source"
    migration_restart_stack "${legacy_root}" "${stack_mode}" || true
    return 1
  fi

  MIGRATION_SIGNAL_TARGET="${legacy_root}"
  MIGRATION_SIGNAL_BACKUP="${backup_root}"
  MIGRATION_SIGNAL_FAILED="${failed_root}"
  MIGRATION_SIGNAL_STACK_MODE="${stack_mode}"
  MIGRATION_SIGNAL_EXPECTED_ITEMS=("${persistent_items[@]}")
  MIGRATION_EFFECTIVE_DATA_DIR="${effective_data_dir}"
  MIGRATION_SIGNAL_ARMED=true
  trap 'migration_abort_on_signal 130' INT
  trap 'migration_abort_on_signal 143' TERM HUP
  if ! migration_write_journal prepared; then
    MIGRATION_SIGNAL_ARMED=false
    trap - INT TERM HUP
    migration_clear_journal || true
    migration_restart_stack "${legacy_root}" "${stack_mode}" || true
    migration_error "Không thể ghi journal migration; chưa thay đổi source"
    return 1
  fi

  if ! mv -- "${legacy_root}" "${backup_root}"; then
    MIGRATION_SIGNAL_ARMED=false
    trap - INT TERM HUP
    migration_error "Không thể tạo backup source cũ"
    migration_clear_journal || true
    migration_restart_stack "${legacy_root}" "${stack_mode}" || true
    return 1
  fi
  if ! migration_write_journal old_moved; then
    local old_restore_ok=true
    mv -- "${backup_root}" "${legacy_root}" || old_restore_ok=false
    MIGRATION_SIGNAL_ARMED=false
    trap - INT TERM HUP
    if [[ "${old_restore_ok}" == true ]]; then
      migration_clear_journal || true
      migration_restart_stack "${legacy_root}" "${stack_mode}" || true
    fi
    migration_error "Không cập nhật được journal sau backup; đã rollback"
    return 1
  fi
  if ! mv -- "${staged_root}" "${legacy_root}"; then
    migration_error "Không thể kích hoạt source mới; đang rollback"
    mv -- "${backup_root}" "${legacy_root}" || {
      migration_error "Cần phục hồi thủ công từ ${backup_root}"
      return 1
    }
    migration_restart_stack "${legacy_root}" "${stack_mode}" || true
    MIGRATION_SIGNAL_ARMED=false
    trap - INT TERM HUP
    migration_clear_journal || true
    return 1
  fi
  new_source_active=true
  if ! migration_write_journal new_active; then
    if migration_rollback_switch \
      "${legacy_root}" "${backup_root}" "${failed_root}" \
      "${stack_mode}" true; then
      migration_clear_journal || true
    fi
    MIGRATION_SIGNAL_ARMED=false
    trap - INT TERM HUP
    migration_error "Không cập nhật được journal sau source swap; đã rollback"
    return 1
  fi

  for item in "${persistent_items[@]}"; do
    if migration_path_exists "${backup_root}/${item}"; then
      if migration_path_exists "${legacy_root}/${item}" \
        || ! mv -- "${backup_root}/${item}" "${legacy_root}/${item}"; then
        migration_error "Không thể gắn lại ${item}; đang rollback"
        if migration_rollback_switch \
          "${legacy_root}" "${backup_root}" "${failed_root}" \
          "${stack_mode}" "${new_source_active}" "${moved_items[@]}"; then
          migration_clear_journal || true
        fi
        MIGRATION_SIGNAL_ARMED=false
        trap - INT TERM HUP
        return 1
      fi
      moved_items+=("${item}")
      if ! migration_write_journal persistent_moved "${moved_items[@]}"; then
        if migration_rollback_switch \
          "${legacy_root}" "${backup_root}" "${failed_root}" \
          "${stack_mode}" true "${moved_items[@]}"; then
          migration_clear_journal || true
        fi
        MIGRATION_SIGNAL_ARMED=false
        trap - INT TERM HUP
        migration_error "Không cập nhật được journal dữ liệu; đã rollback"
        return 1
      fi
    fi
  done

  MIGRATION_BACKUP_PATH="${backup_root}"
  MIGRATION_FAILED_SOURCE_PATH="${failed_root}"
  MIGRATION_OLD_STACK_MODE="${stack_mode}"
  MIGRATION_MOVED_ITEMS=("${moved_items[@]}")
  migration_log "Migration source đã chuẩn bị; backup không bị tự động xóa: ${backup_root}"
  return 0
}
