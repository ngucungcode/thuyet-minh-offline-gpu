#!/usr/bin/env bash

set -Eeuo pipefail

TEST_SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${TEST_SCRIPT_DIR}/../installer/migrate-legacy.sh"

TEST_ROOT="$(mktemp -d)"
trap 'rm -rf -- "${TEST_ROOT}"' EXIT

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

assert_absent() {
  [[ ! -e "$1" && ! -L "$1" ]] || fail "Đường dẫn vẫn tồn tại: $1"
}

create_git_project() {
  local root="$1"
  local version="$2"
  local source_label="$3"
  local dependency="${4:-httpx==0.27.0}"

  mkdir -p \
    "${root}/config" \
    "${root}/installer" \
    "${root}/native" \
    "${root}/scripts" \
    "${root}/src/dub_server"
  printf '%s\n' \
    '.env*' \
    '!.env.native.example' \
    '.venv-native/' \
    'var/' >"${root}/.gitignore"
  printf '%s\n' \
    '[build-system]' \
    'requires = ["setuptools"]' \
    'build-backend = "setuptools.build_meta"' \
    '' \
    '[project]' \
    'name = "thuyet-minh-offline-gpu"' \
    "version = \"${version}\"" \
    "dependencies = [\"${dependency}\"]" \
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
  printf '__version__ = "%s"\n' "${version}" >"${root}/src/dub_server/__init__.py"
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
    printf '%-32s RUNNING   fixture\n' api prowlarr qbittorrent worker
    ;;
  stop)
    if [[ -f "${MIGRATION_TEST_STACK_STATE:?}" ]]; then
      [[ -f "${MIGRATION_EXPECTED_JOURNAL:?}" ]] \
        || { printf 'journal missing before stop\n' >&2; exit 3; }
      printf 'stop\n' >>"${MIGRATION_TEST_EVENTS:?}"
      rm -f -- "${MIGRATION_TEST_STACK_STATE}"
    else
      printf 'Stack native chưa chạy\n'
    fi
    ;;
  start)
    if [[ "${MIGRATION_TEST_REQUIRE_CLOSED_FD9:-false}" == true \
      && -e /proc/$$/fd/9 ]]; then
      printf 'installer lock FD 9 leaked into rollback stack start\n' >&2
      exit 7
    fi
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

  git init -q -- "${root}"
  git -C "${root}" config user.name fixture
  git -C "${root}" config user.email fixture@example.invalid
  git -C "${root}" add .
  git -C "${root}" commit -qm "fixture ${source_label}"
}

prepare_installed_release() {
  local root="$1"
  local version="$2"
  local sentinel="$3"
  local commit

  printf 'DUB_NATIVE_ROOT=%s/var\n' "${root}" >"${root}/.env.native"
  mkdir -p "${root}/var/models" "${root}/var/state" "${root}/.venv-native/bin"
  printf '%s\n' "${sentinel}" >"${root}/var/models/sentinel"
  printf '#!/usr/bin/env bash\nexit 0\n' >"${root}/.venv-native/bin/python"
  chmod +x "${root}/.venv-native/bin/python"
  commit="$(git -C "${root}" rev-parse HEAD)"
  printf '{"schema_version":1,"installer_version":"%s","commit":"%s"}\n' \
    "${version}" "${commit}" >"${root}/var/install-state.json"
}

# A runtime built by releases through v0.3.2 is sm_86. Moving the deployment
# to a different GPU architecture must fail before source or stack mutation.
gpu_swap_root="${TEST_ROOT}/gpu-swap-current"
gpu_swap_stage="${TEST_ROOT}/gpu-swap-stage"
create_git_project "${gpu_swap_root}" 0.3.2 gpu-old
create_git_project "${gpu_swap_stage}" 0.3.3 gpu-new
prepare_installed_release "${gpu_swap_root}" 0.3.2 gpu-model
if migrate_git_release_upgrade \
  "${gpu_swap_root}" "${gpu_swap_stage}" "${gpu_swap_root}/var" false \
  0.3.3 "0.3.2" sm_80; then
  fail "Upgrade tái sử dụng nhầm runtime sm_86 trên GPU sm_80"
fi
[[ "$(<"${gpu_swap_root}/SOURCE_MARKER")" == gpu-old ]] \
  || fail "GPU architecture mismatch đã đổi source hiện tại"
assert_absent "${gpu_swap_root}.migration-state.json"

# Version-only and application-source changes are compatible. The source swap
# must preserve the persistent runtime and support a complete rollback.
upgrade_root="${TEST_ROOT}/upgrade-current"
upgrade_stage="${TEST_ROOT}/upgrade-stage"
create_git_project "${upgrade_root}" 0.3.2 old
create_git_project "${upgrade_stage}" 0.3.3 new
# v0.3.2 used one ambiguous sm_86 field. The target release separates the
# supported matrix from the default/actual native build without changing the
# already-installed sm_86 runtime.
printf '%s\n' \
  '{"schema_version":1,"components":{' \
  '"llama_cpp":{"release":"fixture","cuda_architectures":"86"},' \
  '"prowlarr":{},"qbittorrent_nox":{}' \
  '}}' >"${upgrade_root}/native/components.lock.json"
printf '%s\n' \
  '{"schema_version":1,"components":{' \
  '"llama_cpp":{"release":"fixture","cuda_supported_architectures":[70,75,80,86,89,90],"cuda_default_build_architecture":86},' \
  '"prowlarr":{},"qbittorrent_nox":{}' \
  '}}' >"${upgrade_stage}/native/components.lock.json"
git -C "${upgrade_root}" add native/components.lock.json
git -C "${upgrade_root}" commit --amend --no-edit -q
git -C "${upgrade_stage}" add native/components.lock.json
git -C "${upgrade_stage}" commit --amend --no-edit -q
# The target release changes application behavior around the persistent
# runtime without changing its locked dependencies.
# This must not force an in-place venv rebuild, but the full fingerprint still
# needs to record that the release contents changed.
printf '\n# provider stack control changed in the target release\n' \
  >>"${upgrade_stage}/scripts/native-stack.sh"
printf '\n# cold-install bootstrap procedure changed in the target release\n' \
  >>"${upgrade_stage}/scripts/native-bootstrap.sh"
printf '\n# architecture-aware cold-install build procedure changed\n' \
  >>"${upgrade_stage}/scripts/install-llama-cpp.sh"
git -C "${upgrade_stage}" add \
  scripts/install-llama-cpp.sh scripts/native-bootstrap.sh scripts/native-stack.sh
git -C "${upgrade_stage}" commit --amend --no-edit -q
prepare_installed_release "${upgrade_root}" 0.3.2 model-bytes
job_checkpoint="${upgrade_root}/var/data/jobs/job-upgrade/checkpoint.json"
mkdir -p "$(dirname -- "${job_checkpoint}")"
printf '%s\n' \
  '{"job_id":"job-upgrade","stage":"phase4","status":"failed"}' \
  >"${job_checkpoint}"
model_inode_before="$(stat -c '%i' -- "${upgrade_root}/var/models/sentinel")"
job_inode_before="$(stat -c '%i' -- "${job_checkpoint}")"
job_hash_before="$(sha256sum -- "${job_checkpoint}" | awk '{print $1}')"
old_commit="$(git -C "${upgrade_root}" rev-parse HEAD)"
new_commit="$(git -C "${upgrade_stage}" rev-parse HEAD)"
old_runtime_fingerprint="$(migration_runtime_fingerprint "${upgrade_root}")"
new_runtime_fingerprint="$(migration_runtime_fingerprint "${upgrade_stage}")"
old_compatibility_fingerprint="$(
  migration_runtime_compatibility_fingerprint "${upgrade_root}"
)"
new_compatibility_fingerprint="$(
  migration_runtime_compatibility_fingerprint "${upgrade_stage}"
)"
[[ "${old_runtime_fingerprint}" != "${new_runtime_fingerprint}" ]] \
  || fail "Full runtime fingerprint không phản ánh source release mới"
[[ "${old_compatibility_fingerprint}" == "${new_compatibility_fingerprint}" ]] \
  || fail "Metadata CUDA/bootstrap bị coi nhầm là runtime không tương thích"
export MIGRATION_TEST_STACK_STATE="${TEST_ROOT}/upgrade-stack.running"
export MIGRATION_TEST_EVENTS="${TEST_ROOT}/upgrade-stack.events"
export MIGRATION_EXPECTED_JOURNAL="${upgrade_root}.migration-state.json"
touch "${MIGRATION_TEST_STACK_STATE}"

migrate_git_release_upgrade \
  "${upgrade_root}" "${upgrade_stage}" "${upgrade_root}/var" false \
  0.3.3 "0.3.2"

[[ "$(<"${upgrade_root}/SOURCE_MARKER")" == new ]] \
  || fail "Source release mới chưa active"
[[ "$(git -C "${upgrade_root}" rev-parse HEAD)" == "${new_commit}" ]] \
  || fail "Commit release mới không đúng"
[[ "$(git -C "${MIGRATION_BACKUP_PATH}" rev-parse HEAD)" == "${old_commit}" ]] \
  || fail "Backup không giữ đúng commit release cũ"
[[ -f "${upgrade_root}/var/models/sentinel" \
  && -x "${upgrade_root}/.venv-native/bin/python" ]] \
  || fail "Dữ liệu hoặc runtime không được chuyển sang source mới"
[[ "$(stat -c '%i' -- "${upgrade_root}/var/models/sentinel")" \
  == "${model_inode_before}" ]] \
  || fail "Upgrade đã copy thay vì rename data root"
[[ -f "${job_checkpoint}" \
  && "$(stat -c '%i' -- "${job_checkpoint}")" == "${job_inode_before}" \
  && "$(sha256sum -- "${job_checkpoint}" | awk '{print $1}')" \
    == "${job_hash_before}" ]] \
  || fail "Upgrade làm mất hoặc thay đổi checkpoint job trong var"
[[ ! -e "${MIGRATION_TEST_STACK_STATE}" ]] \
  || fail "Stack cũ chưa dừng trước source swap"

failed_source="${MIGRATION_FAILED_SOURCE_PATH}"
backup_source="${MIGRATION_BACKUP_PATH}"
migration_rollback_switch \
  "${upgrade_root}" "${backup_source}" "${failed_source}" \
  "${MIGRATION_OLD_STACK_MODE}" true "${MIGRATION_MOVED_ITEMS[@]}"
migration_clear_journal
MIGRATION_SIGNAL_ARMED=false
trap - INT TERM HUP
[[ "$(<"${upgrade_root}/SOURCE_MARKER")" == old ]] \
  || fail "Rollback chưa phục hồi source release cũ"
[[ "$(git -C "${upgrade_root}" rev-parse HEAD)" == "${old_commit}" ]] \
  || fail "Rollback chưa phục hồi commit release cũ"
[[ "$(<"${failed_source}/SOURCE_MARKER")" == new ]] \
  || fail "Rollback làm mất source release mới bị lỗi"
[[ "$(stat -c '%i' -- "${upgrade_root}/var/models/sentinel")" \
  == "${model_inode_before}" ]] \
  || fail "Rollback làm đổi data inode"
job_checkpoint="${upgrade_root}/var/data/jobs/job-upgrade/checkpoint.json"
[[ -f "${job_checkpoint}" \
  && "$(stat -c '%i' -- "${job_checkpoint}")" == "${job_inode_before}" \
  && "$(sha256sum -- "${job_checkpoint}" | awk '{print $1}')" \
    == "${job_hash_before}" ]] \
  || fail "Rollback làm mất hoặc thay đổi checkpoint job trong var"
[[ -e "${MIGRATION_TEST_STACK_STATE}" ]] \
  || fail "Rollback chưa khởi động lại stack cũ"
[[ "$(tr '\n' ' ' <"${MIGRATION_TEST_EVENTS}")" == "stop start " ]] \
  || fail "Lifecycle rollback không đúng"
assert_absent "${upgrade_root}.migration-state.json"

# A release not listed by the installer must be rejected before stack stop.
compat_root="${TEST_ROOT}/compat-current"
compat_stage="${TEST_ROOT}/compat-stage"
create_git_project "${compat_root}" 0.2.0 old
create_git_project "${compat_stage}" 0.2.1 new
prepare_installed_release "${compat_root}" 0.2.0 compat-model
export MIGRATION_TEST_STACK_STATE="${TEST_ROOT}/compat-stack.running"
export MIGRATION_TEST_EVENTS="${TEST_ROOT}/compat-stack.events"
export MIGRATION_EXPECTED_JOURNAL="${compat_root}.migration-state.json"
touch "${MIGRATION_TEST_STACK_STATE}"
if migrate_git_release_upgrade \
  "${compat_root}" "${compat_stage}" "${compat_root}/var" false \
  0.2.1 "0.1.9"; then
  fail "Upgrade chấp nhận release không có trong compatibility gate"
fi
[[ "$(<"${compat_root}/SOURCE_MARKER")" == old ]] \
  || fail "Compatibility failure đã đổi source"
[[ -e "${MIGRATION_TEST_STACK_STATE}" ]] \
  || fail "Compatibility failure đã dừng stack"
assert_absent "${MIGRATION_TEST_EVENTS}"
assert_absent "${compat_root}.migration-state.json"

# Dependency changes require a new runtime and must fail before transaction.
abi_root="${TEST_ROOT}/abi-current"
abi_stage="${TEST_ROOT}/abi-stage"
create_git_project "${abi_root}" 0.2.0 old httpx==0.27.0
create_git_project "${abi_stage}" 0.2.1 new httpx==0.28.0
prepare_installed_release "${abi_root}" 0.2.0 abi-model
export MIGRATION_TEST_STACK_STATE="${TEST_ROOT}/abi-stack.running"
export MIGRATION_TEST_EVENTS="${TEST_ROOT}/abi-stack.events"
export MIGRATION_EXPECTED_JOURNAL="${abi_root}.migration-state.json"
touch "${MIGRATION_TEST_STACK_STATE}"
if migrate_git_release_upgrade \
  "${abi_root}" "${abi_stage}" "${abi_root}/var" false \
  0.2.1 "0.2.0"; then
  fail "Upgrade chấp nhận runtime ABI không tương thích"
fi
[[ "$(<"${abi_root}/SOURCE_MARKER")" == old ]] \
  || fail "ABI failure đã đổi source"
[[ -e "${MIGRATION_TEST_STACK_STATE}" ]] \
  || fail "ABI failure đã dừng stack"
assert_absent "${MIGRATION_TEST_EVENTS}"
assert_absent "${abi_root}.migration-state.json"

# install-state must bind the declared release to the exact current commit.
state_root="${TEST_ROOT}/state-current"
state_stage="${TEST_ROOT}/state-stage"
create_git_project "${state_root}" 0.2.0 old
create_git_project "${state_stage}" 0.2.1 new
prepare_installed_release "${state_root}" 0.2.0 state-model
printf '{"schema_version":1,"installer_version":"0.2.0","commit":"%040d"}\n' \
  0 >"${state_root}/var/install-state.json"
export MIGRATION_TEST_STACK_STATE="${TEST_ROOT}/state-stack.running"
export MIGRATION_TEST_EVENTS="${TEST_ROOT}/state-stack.events"
export MIGRATION_EXPECTED_JOURNAL="${state_root}.migration-state.json"
touch "${MIGRATION_TEST_STACK_STATE}"
if migrate_git_release_upgrade \
  "${state_root}" "${state_stage}" "${state_root}/var" false \
  0.2.1 "0.2.0"; then
  fail "Upgrade chấp nhận install-state không khớp commit"
fi
[[ -e "${MIGRATION_TEST_STACK_STATE}" ]] \
  || fail "State mismatch đã dừng stack"
[[ "$(<"${state_root}/SOURCE_MARKER")" == old ]] \
  || fail "State mismatch đã đổi source"
assert_absent "${MIGRATION_TEST_EVENTS}"
assert_absent "${state_root}.migration-state.json"

# Rollback can invoke an older native-stack script that does not close the
# installer descriptor itself. The migration boundary must close only the child
# copy while the parent continues to own the lock.
fd_root="${TEST_ROOT}/fd-boundary"
create_git_project "${fd_root}" 0.2.2 old
export MIGRATION_TEST_STACK_STATE="${TEST_ROOT}/fd-boundary-stack.running"
export MIGRATION_TEST_EVENTS="${TEST_ROOT}/fd-boundary-stack.events"
export MIGRATION_EXPECTED_JOURNAL="${fd_root}.migration-state.json"
export MIGRATION_TEST_REQUIRE_CLOSED_FD9=true
fd_lock="${TEST_ROOT}/fd-boundary.lock"
exec 9>"${fd_lock}"
flock -n 9 || fail "KhÃ´ng thá»ƒ táº¡o installer lock cho rollback fixture"
migration_restart_stack "${fd_root}" native \
  || fail "Rollback stack start khÃ´ng Ä‘Ã³ng installer FD trong child"
if flock -n "${fd_lock}" true; then
  fail "Rollback stack start Ä‘Ã£ lÃ m parent máº¥t installer lock"
fi
exec 9>&-
flock -n "${fd_lock}" true \
  || fail "Installer lock khÃ´ng Ä‘Æ°á»£c nháº£ sau khi parent Ä‘Ã³ng FD"
unset MIGRATION_TEST_REQUIRE_CLOSED_FD9

# A nested directory must never be accepted as the deployment worktree root.
if migration_validate_git_tree "${state_root}/src"; then
  fail "Git validator chấp nhận thư mục con thay cho worktree root"
fi

# Fingerprint parsing failures must propagate even when called in an if guard.
fingerprint_root="${TEST_ROOT}/fingerprint-invalid"
create_git_project "${fingerprint_root}" 0.2.0 invalid
printf 'not valid toml = [\n' >"${fingerprint_root}/pyproject.toml"
if migration_runtime_compatibility_fingerprint \
  "${fingerprint_root}" >/dev/null 2>&1; then
  fail "Compatibility fingerprint che giấu lỗi parse pyproject.toml"
fi

printf 'Installer Git upgrade tests: PASS\n'
