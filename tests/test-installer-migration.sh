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
    "${root}/installer" \
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
  printf '#!/usr/bin/env bash\nexit 0\n' >"${root}/installer/prepare-data-root.sh"
  for script_name in \
    dub-wrapper.sh native-acceptance.sh native-init-services.sh native-model.sh \
    native-phase2-acceptance.sh native-phase3-acceptance.sh \
    native-phase4-acceptance.sh native-preflight.sh; do
    printf '#!/usr/bin/env bash\nexit 0\n' >"${root}/scripts/${script_name}"
  done
  for python_name in \
    generate-sbom.py native-phase2-acceptance.py native-phase3-acceptance.py \
    native-qbittorrent-smoke.py phase4_acceptance.py; do
    printf 'print("fixture")\n' >"${root}/scripts/${python_name}"
  done
  cat >"${root}/scripts/native-stack.sh" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
case "${1:-status}" in
  status)
    if [[ ! -f "${MIGRATION_TEST_STACK_STATE:?}" ]]; then
      printf 'Stack native chưa chạy\n'
      exit 1
    fi
    if [[ "${MIGRATION_TEST_UNHEALTHY_STATUS:-false}" == true ]]; then
      printf '%-32s FATAL     fixture unhealthy\n' api
      printf '%-32s RUNNING   fixture\n' prowlarr qbittorrent worker
      exit 3
    elif [[ "${MIGRATION_TEST_STATUS_ERROR:-false}" == true ]]; then
      printf 'Stack native chưa chạy\n'
      exit 6
    else
      printf '%-32s RUNNING   fixture\n' api prowlarr qbittorrent worker
    fi
    ;;
  stop)
    if [[ -f "${MIGRATION_TEST_STACK_STATE:?}" ]]; then
      if [[ -n "${MIGRATION_EXPECTED_JOURNAL:-}" ]]; then
        [[ -f "${MIGRATION_EXPECTED_JOURNAL}" ]] \
          || { printf 'journal missing before stop\n' >&2; exit 3; }
      fi
      printf 'stop\n' >>"${MIGRATION_TEST_EVENTS:?}"
      rm -f -- "${MIGRATION_TEST_STACK_STATE}"
      [[ "${MIGRATION_TEST_FAIL_STOP_AFTER_STATE:-false}" != true ]] || exit 4
    else
      printf 'Stack native chưa chạy\n'
    fi
    ;;
  start)
    [[ "${MIGRATION_TEST_FAIL_START:-false}" != true ]] || exit 5
    printf 'start\n' >>"${MIGRATION_TEST_EVENTS:?}"
    touch "${MIGRATION_TEST_STACK_STATE}"
    ;;
  *) exit 2 ;;
esac
SH
  chmod +x \
    "${root}/installer/prepare-data-root.sh" \
    "${root}/scripts/dub-wrapper.sh" \
    "${root}/scripts/native-acceptance.sh" \
    "${root}/scripts/native-bootstrap.sh" \
    "${root}/scripts/native-init-services.sh" \
    "${root}/scripts/native-model.sh" \
    "${root}/scripts/native-phase2-acceptance.sh" \
    "${root}/scripts/native-phase3-acceptance.sh" \
    "${root}/scripts/native-phase4-acceptance.sh" \
    "${root}/scripts/native-preflight.sh" \
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
chmod 0755 "${legacy_root}/var"
printf 'model-bytes\n' >"${legacy_root}/var/models/sentinel"
printf '#!/usr/bin/env bash\nexit 0\n' >"${legacy_root}/.venv-native/bin/python"
chmod +x "${legacy_root}/.venv-native/bin/python"
model_inode_before="$(stat -c '%i' -- "${legacy_root}/var/models/sentinel")"
data_root_identity_before="$(stat -c '%d:%i:%u:%g:%a' -- "${legacy_root}/var")"
touch "${MIGRATION_TEST_STACK_STATE}"
export MIGRATION_EXPECTED_JOURNAL="${legacy_root}.migration-state.json"

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
[[ "$(stat -c '%d:%i:%u:%g:%a' -- "${legacy_root}/var")" \
  == "${data_root_identity_before}" ]] \
  || fail "Migration làm đổi metadata data root"
[[ ! -e "${MIGRATION_TEST_STACK_STATE}" ]] || fail "Stack cũ chưa dừng"
[[ "$(<"${MIGRATION_TEST_EVENTS}")" == stop ]] || fail "Stack stop không đúng một lần"

failed_source_path="${MIGRATION_FAILED_SOURCE_PATH}"
chmod 0700 "${legacy_root}/var"
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
[[ "$(stat -c '%d:%i:%u:%g:%a' -- "${legacy_root}/var")" \
  == "${data_root_identity_before}" ]] \
  || fail "Rollback không phục hồi metadata data root"
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
export MIGRATION_EXPECTED_JOURNAL="${signal_root}.migration-state.json"
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

restart_fail_root="${TEST_ROOT}/restart-fail-legacy"
restart_fail_stage="${TEST_ROOT}/restart-fail-stage"
create_project_tree "${restart_fail_root}" old
create_project_tree "${restart_fail_stage}" new
printf 'DUB_NATIVE_ROOT=%s/var\n' "${restart_fail_root}" \
  >"${restart_fail_root}/.env.native"
mkdir -p "${restart_fail_root}/var/models" "${restart_fail_root}/.venv-native/bin"
printf 'restart-fail-model\n' >"${restart_fail_root}/var/models/sentinel"
printf '#!/usr/bin/env bash\nexit 0\n' \
  >"${restart_fail_root}/.venv-native/bin/python"
chmod +x "${restart_fail_root}/.venv-native/bin/python"
restart_fail_identity="$(stat -c '%d:%i:%u:%g:%a' -- "${restart_fail_root}/var")"
export MIGRATION_TEST_STACK_STATE="${TEST_ROOT}/restart-fail-stack.running"
export MIGRATION_TEST_EVENTS="${TEST_ROOT}/restart-fail-stack.events"
export MIGRATION_EXPECTED_JOURNAL="${restart_fail_root}.migration-state.json"
touch "${MIGRATION_TEST_STACK_STATE}"
migrate_legacy_install "${restart_fail_root}" "${restart_fail_stage}"
chmod 0700 "${restart_fail_root}/var"
export MIGRATION_TEST_FAIL_START=true
if migration_rollback_switch \
  "${restart_fail_root}" "${MIGRATION_BACKUP_PATH}" \
  "${MIGRATION_FAILED_SOURCE_PATH}" "${MIGRATION_OLD_STACK_MODE}" \
  true "${MIGRATION_MOVED_ITEMS[@]}"; then
  fail "Rollback báo thành công dù stack không restart được"
fi
unset MIGRATION_TEST_FAIL_START
[[ "$(<"${restart_fail_root}/SOURCE_MARKER")" == old ]] \
  || fail "Rollback lỗi restart chưa phục hồi source cũ"
assert_file "${restart_fail_root}/var/models/sentinel"
[[ "$(stat -c '%d:%i:%u:%g:%a' -- "${restart_fail_root}/var")" \
  == "${restart_fail_identity}" ]] \
  || fail "Rollback lỗi restart chưa phục hồi metadata data root"
assert_file "${restart_fail_root}.migration-state.json"
grep -q '"phase": "rollback_incomplete"' \
  "${restart_fail_root}.migration-state.json" \
  || fail "Journal không ghi rollback_incomplete"
"${restart_fail_root}/scripts/native-stack.sh" start >/dev/null
migration_clear_journal

stop_fail_root="${TEST_ROOT}/stop-fail-legacy"
stop_fail_stage="${TEST_ROOT}/stop-fail-stage"
create_project_tree "${stop_fail_root}" old
create_project_tree "${stop_fail_stage}" new
printf 'DUB_NATIVE_ROOT=%s/var\n' "${stop_fail_root}" \
  >"${stop_fail_root}/.env.native"
mkdir -p "${stop_fail_root}/var/models" "${stop_fail_root}/.venv-native/bin"
printf 'stop-fail-model\n' >"${stop_fail_root}/var/models/sentinel"
printf '#!/usr/bin/env bash\nexit 0\n' >"${stop_fail_root}/.venv-native/bin/python"
chmod +x "${stop_fail_root}/.venv-native/bin/python"
export MIGRATION_TEST_STACK_STATE="${TEST_ROOT}/stop-fail-stack.running"
export MIGRATION_TEST_EVENTS="${TEST_ROOT}/stop-fail-stack.events"
export MIGRATION_EXPECTED_JOURNAL="${stop_fail_root}.migration-state.json"
export MIGRATION_TEST_FAIL_STOP_AFTER_STATE=true
touch "${MIGRATION_TEST_STACK_STATE}"
if migrate_legacy_install "${stop_fail_root}" "${stop_fail_stage}"; then
  fail "Migration tiếp tục sau khi stop stack trả lỗi"
fi
unset MIGRATION_TEST_FAIL_STOP_AFTER_STATE
[[ "$(<"${stop_fail_root}/SOURCE_MARKER")" == old ]] \
  || fail "Stop failure đã đổi source"
[[ "$(<"${stop_fail_stage}/SOURCE_MARKER")" == new ]] \
  || fail "Stop failure đã đổi staging source"
assert_absent "${stop_fail_root}.migration-state.json"
[[ -e "${MIGRATION_TEST_STACK_STATE}" ]] \
  || fail "Stop failure không phục hồi stack cũ"

unhealthy_root="${TEST_ROOT}/unhealthy-legacy"
unhealthy_stage="${TEST_ROOT}/unhealthy-stage"
create_project_tree "${unhealthy_root}" old
create_project_tree "${unhealthy_stage}" new
printf 'DUB_NATIVE_ROOT=%s/var\n' "${unhealthy_root}" \
  >"${unhealthy_root}/.env.native"
mkdir -p "${unhealthy_root}/var/models" "${unhealthy_root}/.venv-native/bin"
mkdir -p "${unhealthy_root}/var/run"
printf '%s\n' "$$" >"${unhealthy_root}/var/run/supervisord.pid"
printf '#!/usr/bin/env bash\nexit 0\n' >"${unhealthy_root}/.venv-native/bin/python"
chmod +x "${unhealthy_root}/.venv-native/bin/python"
export MIGRATION_TEST_STACK_STATE="${TEST_ROOT}/unhealthy-stack.running"
export MIGRATION_TEST_EVENTS="${TEST_ROOT}/unhealthy-stack.events"
export MIGRATION_EXPECTED_JOURNAL="${unhealthy_root}.migration-state.json"
export MIGRATION_TEST_UNHEALTHY_STATUS=true
touch "${MIGRATION_TEST_STACK_STATE}"
if migrate_legacy_install "${unhealthy_root}" "${unhealthy_stage}"; then
  fail "Migration chấp nhận stack cũ không khỏe"
fi
unset MIGRATION_TEST_UNHEALTHY_STATUS
[[ "$(<"${unhealthy_root}/SOURCE_MARKER")" == old ]] \
  || fail "Preflight stack không khỏe đã đổi source"
assert_absent "${unhealthy_root}.migration-state.json"
assert_absent "${MIGRATION_TEST_EVENTS}"
[[ -e "${MIGRATION_TEST_STACK_STATE}" ]] \
  || fail "Preflight stack không khỏe đã dừng supervisor"
unhealthy_status_stage="${TEST_ROOT}/unhealthy-status-stage"
create_project_tree "${unhealthy_status_stage}" new
export MIGRATION_TEST_STATUS_ERROR=true
if migrate_legacy_install "${unhealthy_root}" "${unhealthy_status_stage}"; then
  fail "Migration coi lỗi status là stack đã dừng"
fi
unset MIGRATION_TEST_STATUS_ERROR
assert_absent "${unhealthy_root}.migration-state.json"
assert_absent "${MIGRATION_TEST_EVENTS}"
[[ -e "${MIGRATION_TEST_STACK_STATE}" ]] \
  || fail "Lỗi status đã dừng supervisor"

stopped_root="${TEST_ROOT}/stopped-legacy"
stopped_stage="${TEST_ROOT}/stopped-stage"
create_project_tree "${stopped_root}" old
create_project_tree "${stopped_stage}" new
printf '# DUB_NATIVE_ROOT intentionally uses the default\n' \
  >"${stopped_root}/.env.native"
mkdir -p "${stopped_root}/var/models" "${stopped_root}/.venv-native/bin"
printf '#!/usr/bin/env bash\nexit 0\n' >"${stopped_root}/.venv-native/bin/python"
chmod +x "${stopped_root}/.venv-native/bin/python"
export MIGRATION_TEST_STACK_STATE="${TEST_ROOT}/stopped-stack.running"
export MIGRATION_TEST_EVENTS="${TEST_ROOT}/stopped-stack.events"
export MIGRATION_EXPECTED_JOURNAL="${stopped_root}.migration-state.json"
migrate_legacy_install "${stopped_root}" "${stopped_stage}"
stopped_failed_source="${MIGRATION_FAILED_SOURCE_PATH}"
migration_rollback_switch \
  "${stopped_root}" "${MIGRATION_BACKUP_PATH}" "${stopped_failed_source}" \
  "${MIGRATION_OLD_STACK_MODE}" true "${MIGRATION_MOVED_ITEMS[@]}"
migration_clear_journal
assert_absent "${MIGRATION_TEST_STACK_STATE}"
assert_absent "${MIGRATION_TEST_EVENTS}"

systemd_root="${TEST_ROOT}/systemd-legacy"
systemd_stage="${TEST_ROOT}/systemd-stage"
create_project_tree "${systemd_root}" old
create_project_tree "${systemd_stage}" new
printf '# DUB_NATIVE_ROOT intentionally uses the default\n' \
  >"${systemd_root}/.env.native"
mkdir -p "${systemd_root}/var/models" "${systemd_root}/.venv-native/bin"
printf '#!/usr/bin/env bash\nexit 0\n' >"${systemd_root}/.venv-native/bin/python"
chmod +x "${systemd_root}/.venv-native/bin/python"
fake_system_bin="${TEST_ROOT}/fake-system-bin"
mkdir -p "${fake_system_bin}"
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -Eeuo pipefail' \
  'case "$1" in' \
  '  is-active) [[ -f "${MIGRATION_TEST_SYSTEMD_ACTIVE:?}" ]] ;;' \
  '  cat) [[ -f "${MIGRATION_TEST_SYSTEMD_UNIT:?}" ]] ;;' \
  '  stop)' \
  '    [[ -f "${MIGRATION_EXPECTED_JOURNAL:?}" ]] || exit 7' \
  '    printf "systemd-stop\\n" >>"${MIGRATION_TEST_EVENTS:?}"' \
  '    rm -f -- "${MIGRATION_TEST_SYSTEMD_ACTIVE}" "${MIGRATION_TEST_STACK_STATE:?}"' \
  '    ;;' \
  '  start)' \
  '    printf "systemd-start\\n" >>"${MIGRATION_TEST_EVENTS:?}"' \
  '    touch "${MIGRATION_TEST_SYSTEMD_ACTIVE:?}" "${MIGRATION_TEST_STACK_STATE:?}"' \
  '    ;;' \
  '  *) exit 8 ;;' \
  'esac' >"${fake_system_bin}/systemctl"
chmod +x "${fake_system_bin}/systemctl"
export MIGRATION_TEST_STACK_STATE="${TEST_ROOT}/systemd-stack.running"
export MIGRATION_TEST_EVENTS="${TEST_ROOT}/systemd-stack.events"
export MIGRATION_TEST_SYSTEMD_ACTIVE="${TEST_ROOT}/systemd-active"
export MIGRATION_TEST_SYSTEMD_UNIT="${TEST_ROOT}/systemd-unit"
export MIGRATION_EXPECTED_JOURNAL="${systemd_root}.migration-state.json"
touch "${MIGRATION_TEST_STACK_STATE}" "${MIGRATION_TEST_SYSTEMD_ACTIVE}" \
  "${MIGRATION_TEST_SYSTEMD_UNIT}"
original_path="${PATH}"
PATH="${fake_system_bin}:${PATH}"
export PATH
migrate_legacy_install "${systemd_root}" "${systemd_stage}"
[[ "${MIGRATION_OLD_STACK_MODE}" == systemd ]] \
  || fail "Migration không giữ control plane systemd"
assert_absent "${MIGRATION_TEST_STACK_STATE}"
assert_absent "${MIGRATION_TEST_SYSTEMD_ACTIVE}"
systemd_failed_source="${MIGRATION_FAILED_SOURCE_PATH}"
migration_rollback_switch \
  "${systemd_root}" "${MIGRATION_BACKUP_PATH}" "${systemd_failed_source}" \
  "${MIGRATION_OLD_STACK_MODE}" true "${MIGRATION_MOVED_ITEMS[@]}"
migration_clear_journal
[[ -e "${MIGRATION_TEST_STACK_STATE}" \
  && -e "${MIGRATION_TEST_SYSTEMD_ACTIVE}" ]] \
  || fail "Rollback không phục hồi systemd stack"
[[ "$(tail -n 1 "${MIGRATION_TEST_EVENTS}")" == systemd-start ]] \
  || fail "Rollback không restart bằng systemd"

stopped_systemd_root="${TEST_ROOT}/stopped-systemd-legacy"
stopped_systemd_stage="${TEST_ROOT}/stopped-systemd-stage"
create_project_tree "${stopped_systemd_root}" old
create_project_tree "${stopped_systemd_stage}" new
printf '# DUB_NATIVE_ROOT intentionally uses the default\n' \
  >"${stopped_systemd_root}/.env.native"
mkdir -p \
  "${stopped_systemd_root}/var/models" \
  "${stopped_systemd_root}/.venv-native/bin"
printf '#!/usr/bin/env bash\nexit 0\n' \
  >"${stopped_systemd_root}/.venv-native/bin/python"
chmod +x "${stopped_systemd_root}/.venv-native/bin/python"
export MIGRATION_TEST_STACK_STATE="${TEST_ROOT}/stopped-systemd-stack.running"
export MIGRATION_TEST_EVENTS="${TEST_ROOT}/stopped-systemd-stack.events"
export MIGRATION_TEST_SYSTEMD_ACTIVE="${TEST_ROOT}/stopped-systemd-active"
export MIGRATION_TEST_SYSTEMD_UNIT="${TEST_ROOT}/stopped-systemd-unit"
export MIGRATION_EXPECTED_JOURNAL="${stopped_systemd_root}.migration-state.json"
touch "${MIGRATION_TEST_SYSTEMD_UNIT}"
migrate_legacy_install "${stopped_systemd_root}" "${stopped_systemd_stage}"
[[ "${MIGRATION_OLD_STACK_MODE}" == systemd-stopped ]] \
  || fail "Migration không nhận diện systemd unit đang dừng"
stopped_systemd_failed_source="${MIGRATION_FAILED_SOURCE_PATH}"
migration_rollback_switch \
  "${stopped_systemd_root}" "${MIGRATION_BACKUP_PATH}" \
  "${stopped_systemd_failed_source}" "${MIGRATION_OLD_STACK_MODE}" \
  true "${MIGRATION_MOVED_ITEMS[@]}"
migration_clear_journal
assert_absent "${MIGRATION_TEST_STACK_STATE}"
assert_absent "${MIGRATION_TEST_SYSTEMD_ACTIVE}"
[[ -e "${MIGRATION_TEST_SYSTEMD_UNIT}" ]] \
  || fail "Rollback làm mất systemd unit đang dừng"
PATH="${original_path}"
export PATH

export_env_root="${TEST_ROOT}/export-env-legacy"
export_env_stage="${TEST_ROOT}/export-env-stage"
create_project_tree "${export_env_root}" old
create_project_tree "${export_env_stage}" new
printf 'export DUB_NATIVE_ROOT=%s/var\n' "${export_env_root}" \
  >"${export_env_root}/.env.native"
mkdir -p "${export_env_root}/var/models" "${export_env_root}/.venv-native/bin"
printf '#!/usr/bin/env bash\nexit 0\n' >"${export_env_root}/.venv-native/bin/python"
chmod +x "${export_env_root}/.venv-native/bin/python"
export MIGRATION_TEST_STACK_STATE="${TEST_ROOT}/export-env-stack.running"
export MIGRATION_TEST_EVENTS="${TEST_ROOT}/export-env-stack.events"
export MIGRATION_EXPECTED_JOURNAL="${export_env_root}.migration-state.json"
if migrate_legacy_install "${export_env_root}" "${export_env_stage}"; then
  fail "Migration chấp nhận export DUB_NATIVE_ROOT không cùng semantics"
fi
[[ "$(<"${export_env_root}/SOURCE_MARKER")" == old ]] \
  || fail "DUB_NATIVE_ROOT không hợp lệ đã đổi source"
assert_absent "${export_env_root}.migration-state.json"

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
