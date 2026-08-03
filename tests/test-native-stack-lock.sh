#!/usr/bin/env bash

set -Eeuo pipefail

TEST_SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "${TEST_SCRIPT_DIR}/.." && pwd)"
TEST_ROOT="$(mktemp -d)"
daemon_pid=""

cleanup() {
  if [[ -n "${daemon_pid}" ]]; then
    kill "${daemon_pid}" >/dev/null 2>&1 || true
    for _ in {1..50}; do
      kill -0 "${daemon_pid}" >/dev/null 2>&1 || break
      sleep 0.02
    done
    kill -9 "${daemon_pid}" >/dev/null 2>&1 || true
  fi
  rm -rf -- "${TEST_ROOT}"
}
trap cleanup EXIT

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

runtime_root="${TEST_ROOT}/runtime"
venv_dir="${TEST_ROOT}/venv"
pid_file="${TEST_ROOT}/supervisord.pid"
lock_file="${TEST_ROOT}/installer.lock"
mkdir -p "${venv_dir}/bin" "${runtime_root}"

cat >"${venv_dir}/bin/supervisord" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
printf '%s\n' "$$" >"${LOCK_TEST_DAEMON_PID_FILE:?}"
exec sleep 300
SH
cat >"${venv_dir}/bin/supervisorctl" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
action="${*: -1}"
pid_file="${LOCK_TEST_DAEMON_PID_FILE:?}"
[[ -s "${pid_file}" ]] || exit 1
daemon_pid="$(<"${pid_file}")"
[[ "${daemon_pid}" =~ ^[0-9]+$ ]] || exit 1
kill -0 "${daemon_pid}" 2>/dev/null || exit 1
case "${action}" in
  pid)
    printf '%s\n' "${daemon_pid}"
    ;;
  status)
    printf '%-32s RUNNING   pid %s\n' api "${daemon_pid}"
    printf '%-32s RUNNING   pid %s\n' prowlarr "${daemon_pid}"
    printf '%-32s RUNNING   pid %s\n' qbittorrent "${daemon_pid}"
    printf '%-32s RUNNING   pid %s\n' worker "${daemon_pid}"
    ;;
  *)
    exit 2
    ;;
esac
SH
cat >"${TEST_ROOT}/prowlarr" <<'SH'
#!/usr/bin/env bash
exit 0
SH
chmod +x \
  "${venv_dir}/bin/supervisord" \
  "${venv_dir}/bin/supervisorctl" \
  "${TEST_ROOT}/prowlarr"

exec 9>"${lock_file}"
flock -n 9 || fail "Không thể tạo installer lock cho fixture"
if flock -n "${lock_file}" true; then
  fail "Installer cạnh tranh đã lấy được lock đang được giữ"
fi
LOCK_TEST_DAEMON_PID_FILE="${pid_file}" \
DUB_NATIVE_ENV_FILE="${TEST_ROOT}/missing.env" \
DUB_NATIVE_ROOT="${runtime_root}" \
DUB_VENV_DIR="${venv_dir}" \
DUB_PROWLARR_BIN="${TEST_ROOT}/prowlarr" \
DUB_NATIVE_USER="$(id -un)" \
  "${PROJECT_ROOT}/scripts/native-stack.sh" start \
  >"${TEST_ROOT}/stack.stdout" 2>"${TEST_ROOT}/stack.stderr"

[[ -s "${pid_file}" ]] || fail "Fake supervisord không ghi PID"
daemon_pid="$(<"${pid_file}")"
[[ "${daemon_pid}" =~ ^[0-9]+$ ]] || fail "PID fake supervisord không hợp lệ"
kill -0 "${daemon_pid}" 2>/dev/null || fail "Fake supervisord chưa chạy"

# Simulate the installer exiting while the provider-mode daemon remains alive.
exec 9>&-
flock -n "${lock_file}" true \
  || fail "Supervisord đã kế thừa installer lock FD 9"

printf 'PASS: provider daemon does not inherit installer lock\n'
