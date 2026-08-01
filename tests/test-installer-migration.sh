#!/usr/bin/env bash

set -Eeuo pipefail

TEST_SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${TEST_SCRIPT_DIR}/../installer/migrate-legacy.sh"

TEST_ROOT="$(mktemp -d)"
trap 'rm -rf -- "${TEST_ROOT}"' EXIT
export MIGRATION_TEST_STACK_STATE="${TEST_ROOT}/stack.running"
export MIGRATION_TEST_EVENTS="${TEST_ROOT}/stack.events"

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

assert_file() {
  [[ -f "$1" ]] || fail "Thiếu file $1"
}

assert_absent() {
  [[ ! -e "$1" && ! -L "$1" ]] || fail "Đường dẫn vẫn tồn tại: $1"
}

create_project_tree() {
  local root="$1"
  local source_label="$2"
  mkdir -p \
    "${root}/config" \
    "${root}/native" \
    "${root}/scripts" \
    "${root}/src/dub_server"
  printf '%s\n' \
    '[build-system]' \
    'requires = ["setuptools"]' \
    'build-backend = "setuptools.build_meta"' \
    '' \
    '[project]' \
    'name = "thuyet-minh-offline-gpu"' \
    'version = "0.1.0"' \
    '' \
    '[project.scripts]' \
    'dub = "dub_server.cli:app"' \
    >"${root}/pyproject.toml"
  printf 'GPL-3.0-or-later\n' >"${root}/LICENSE"
  printf '# native environment fixture\n' >"${root}/.env.native.example"
  printf '%s\n' \
    '{"schema_version":1,"models":[' \
    '{"id":"asr-faster-whisper-small"},' \
    '{"id":"mt-gemma4-e2b-q4"},' \
    '{"id":"separation-tiger-dnr"},' \
    '{"id":"tts-piper-vi-vais1000-medium"}' \
    ']}' >"${root}/config/models.lock.json"
  printf '%s\n' \
    '{"schema_version":1,"components":{' \
    '"llama_cpp":{},"prowlarr":{},"qbittorrent_nox":{}' \
    '}}' >"${root}/native/components.lock.json"
  printf '[supervisord]\nnodaemon=true\n' >"${root}/native/supervisord.conf"
  printf '__version__ = "0.1.0"\n' >"${root}/src/dub_server/__init__.py"
  printf 'app = object()\n' >"${root}/src/dub_server/cli.py"
  printf '# shared runtime fixture\n' >"${root}/scripts/native-common.sh"
  printf '#!/usr/bin/env bash\nexit 0\n' >"${root}/scripts/native-bootstrap.sh"
  printf '#!/usr/bin/env bash\nexit 0\n' >"${root}/scripts/install-llama-cpp.sh"
  printf 'print("fixture")\n' >"${root}/scripts/vieneu-offline.py"
  cat >"${root}/scripts/native-stack.sh" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
case "${1:-status}" in
  status)
    [[ -f "${MIGRATION_TEST_STACK_STATE:?}" ]]
    ;;
  stop)
    if [[ -f "${MIGRATION_TEST_STACK_STATE:?}" ]]; then
      printf 'stop\n' >>"${MIGRATION_TEST_EVENTS:?}"
      rm -f -- "${MIGRATION_TEST_STACK_STATE}"
    else
      printf 'Stack native chưa chạy\n'
    fi
    ;;
  start)
    printf 'start\n' >>"${MIGRATION_TEST_EVENTS:?}"
    touch "${MIGRATION_TEST_STACK_STATE}"
    ;;
  *) exit 2 ;;
esac
SH
  chmod +x \
    "${root}/scripts/native-bootstrap.sh" \
    "${root}/scripts/install-llama-cpp.sh" \
    "${root}/scripts/native-stack.sh"
  printf '%s\n' "${source_label}" >"${root}/SOURCE_MARKER"
}

legacy_root="${TEST_ROOT}/legacy"
staged_root="${TEST_ROOT}/staged"
create_project_tree "${legacy_root}" old
create_project_tree "${staged_root}" new
printf 'DUB_NATIVE_ROOT=%s/var\n' "${legacy_root}" >"${legacy_root}/.env.native"
chmod 0600 "${legacy_root}/.env.native"
cp -- "${legacy_root}/.env.native" "${TEST_ROOT}/env.before"
mkdir -p "${legacy_root}/var/models" "${legacy_root}/.venv-native/bin"
printf 'model-bytes\n' >"${legacy_root}/var/models/sentinel"
printf '#!/usr/bin/env bash\nexit 0\n' >"${legacy_root}/.venv-native/bin/python"
chmod +x "${legacy_root}/.venv-native/bin/python"
model_inode_before="$(stat -c '%i' -- "${legacy_root}/var/models/sentinel")"
touch "${MIGRATION_TEST_STACK_STATE}"

migrate_legacy_install "${legacy_root}" "${staged_root}"

[[ "${MIGRATED_RUNTIME_REUSABLE}" == true ]] || fail "Runtime tương thích không được tái sử dụng"
[[ -n "${MIGRATION_BACKUP_PATH}" ]] || fail "Không trả đường dẫn backup"
assert_file "${MIGRATION_JOURNAL_PATH}"
grep -q '"phase": "persistent_moved"' "${MIGRATION_JOURNAL_PATH}" \
  || fail "Journal không ghi phase persistent_moved"
assert_file "${legacy_root}/SOURCE_MARKER"
[[ "$(<"${legacy_root}/SOURCE_MARKER")" == new ]] || fail "Source mới chưa active"
assert_file "${MIGRATION_BACKUP_PATH}/SOURCE_MARKER"
[[ "$(<"${MIGRATION_BACKUP_PATH}/SOURCE_MARKER")" == old ]] || fail "Backup source cũ sai"
cmp -s -- "${TEST_ROOT}/env.before" "${legacy_root}/.env.native" \
  || fail ".env.native trong source mới bị đổi nội dung"
cmp -s -- "${TEST_ROOT}/env.before" "${MIGRATION_BACKUP_PATH}/.env.native" \
  || fail ".env.native trong backup bị đổi nội dung"
assert_file "${legacy_root}/var/models/sentinel"
assert_file "${legacy_root}/.venv-native/bin/python"
assert_absent "${MIGRATION_BACKUP_PATH}/var"
assert_absent "${MIGRATION_BACKUP_PATH}/.venv-native"
[[ "$(stat -c '%i' -- "${legacy_root}/var/models/sentinel")" == "${model_inode_before}" ]] \
  || fail "Migration đã copy thay vì rename dữ liệu"
[[ ! -e "${MIGRATION_TEST_STACK_STATE}" ]] || fail "Stack cũ chưa dừng"
[[ "$(<"${MIGRATION_TEST_EVENTS}")" == stop ]] || fail "Stack stop không đúng một lần"

failed_source_path="${MIGRATION_FAILED_SOURCE_PATH}"
migration_rollback_switch \
  "${legacy_root}" "${MIGRATION_BACKUP_PATH}" "${failed_source_path}" \
  "${MIGRATION_OLD_STACK_MODE}" true "${MIGRATION_MOVED_ITEMS[@]}"
migration_clear_journal
assert_absent "${legacy_root}.migration-state.json"
[[ "$(<"${legacy_root}/SOURCE_MARKER")" == old ]] || fail "Rollback chưa phục hồi source cũ"
[[ "$(<"${failed_source_path}/SOURCE_MARKER")" == new ]] || fail "Rollback làm mất source mới"
assert_file "${legacy_root}/var/models/sentinel"
assert_file "${legacy_root}/.venv-native/bin/python"
[[ "$(stat -c '%i' -- "${legacy_root}/var/models/sentinel")" == "${model_inode_before}" ]] \
  || fail "Rollback làm đổi inode dữ liệu"
[[ -e "${MIGRATION_TEST_STACK_STATE}" ]] || fail "Rollback chưa khởi động lại stack cũ"
[[ "$(wc -l <"${MIGRATION_TEST_EVENTS}")" -eq 2 ]] || fail "Stack lifecycle rollback không đúng"
[[ "$(tail -n 1 "${MIGRATION_TEST_EVENTS}")" == start ]] || fail "Stack cũ chưa được start"

signal_root="${TEST_ROOT}/signal-legacy"
signal_stage="${TEST_ROOT}/signal-stage"
create_project_tree "${signal_root}" old
create_project_tree "${signal_stage}" new
printf 'DUB_NATIVE_ROOT=%s/var\n' "${signal_root}" >"${signal_root}/.env.native"
mkdir -p "${signal_root}/var/models" "${signal_root}/.venv-native/bin"
printf 'signal-model\n' >"${signal_root}/var/models/sentinel"
printf '#!/usr/bin/env bash\nexit 0\n' >"${signal_root}/.venv-native/bin/python"
chmod +x "${signal_root}/.venv-native/bin/python"
export MIGRATION_TEST_STACK_STATE="${TEST_ROOT}/signal-stack.running"
export MIGRATION_TEST_EVENTS="${TEST_ROOT}/signal-stack.events"
touch "${MIGRATION_TEST_STACK_STATE}"
if (
  migrate_legacy_install "${signal_root}" "${signal_stage}"
  migration_abort_on_signal 143
); then
  fail "Signal handler không trả mã lỗi"
else
  signal_status=$?
fi
[[ "${signal_status}" -eq 143 ]] || fail "Signal handler trả sai exit code"
[[ "$(<"${signal_root}/SOURCE_MARKER")" == old ]] || fail "Signal rollback chưa phục hồi source"
assert_file "${signal_root}/var/models/sentinel"
assert_file "${signal_root}/.venv-native/bin/python"
assert_absent "${signal_root}.migration-state.json"
[[ -e "${MIGRATION_TEST_STACK_STATE}" ]] || fail "Signal rollback chưa restart stack"

fake_root="${TEST_ROOT}/fake"
fake_stage="${TEST_ROOT}/fake-stage"
mkdir -p "${fake_root}"
printf 'not-this-project\n' >"${fake_root}/pyproject.toml"
create_project_tree "${fake_stage}" new
if migrate_legacy_install "${fake_root}" "${fake_stage}"; then
  fail "Target giả đã được chấp nhận"
fi
assert_file "${fake_root}/pyproject.toml"
assert_file "${fake_stage}/SOURCE_MARKER"

printf 'Installer migration tests: PASS\n'
