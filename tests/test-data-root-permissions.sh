#!/usr/bin/env bash

set -Eeuo pipefail

TEST_SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PREPARER="${TEST_SCRIPT_DIR}/../installer/prepare-data-root.sh"
INSTALLER="${TEST_SCRIPT_DIR}/../install.sh"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf -- "${TEST_ROOT}"' EXIT
chmod 0755 "${TEST_ROOT}"

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

existing_root="${TEST_ROOT}/existing"
mkdir -m 0755 "${existing_root}"
printf 'sentinel\n' >"${existing_root}/sentinel"
if [[ "${EUID}" -eq 0 ]] && id nobody >/dev/null 2>&1; then
  service_group="$(id -gn nobody)"
  mkdir -m 0750 "${existing_root}/service-data"
  chown nobody:"${service_group}" "${existing_root}/service-data"
  printf 'service-sentinel\n' >"${existing_root}/service-data/sentinel"
  chown nobody:"${service_group}" "${existing_root}/service-data/sentinel"
fi
existing_identity="$(stat -c '%d:%i:%u:%g:%a' -- "${existing_root}")"
[[ "$("${PREPARER}" "${existing_root}")" == existing ]] \
  || fail "Helper không nhận diện data root hiện hữu"
[[ "$(stat -c '%d:%i:%u:%g:%a' -- "${existing_root}")" \
  == "${existing_identity}" ]] \
  || fail "Helper làm đổi metadata data root hiện hữu"
[[ "$(<"${existing_root}/sentinel")" == sentinel ]] \
  || fail "Helper làm đổi nội dung data root hiện hữu"
if [[ "${EUID}" -eq 0 ]] && id nobody >/dev/null 2>&1; then
  runuser -u nobody -- test -x "${existing_root}" \
    || fail "Service user không traverse được data root hiện hữu"
  runuser -u nobody -- test -r "${existing_root}/service-data/sentinel" \
    || fail "Service user không đọc được thư mục con của mình"
fi

created_root="${TEST_ROOT}/created"
[[ "$("${PREPARER}" "${created_root}")" == created ]] \
  || fail "Helper không báo data root mới"
[[ -d "${created_root}" && ! -L "${created_root}" ]] \
  || fail "Data root mới không phải thư mục thực"
[[ "$(stat -c '%a' -- "${created_root}")" == 755 ]] \
  || fail "Data root mới không có mode bootstrap 0755"
test -x "${created_root}" || fail "Data root mới không traversable"

symlink_root="${TEST_ROOT}/symlink"
ln -s "${existing_root}" "${symlink_root}"
if "${PREPARER}" "${symlink_root}" >/dev/null 2>&1; then
  fail "Helper chấp nhận symlink data root"
fi

file_root="${TEST_ROOT}/regular-file"
printf 'not-a-directory\n' >"${file_root}"
if "${PREPARER}" "${file_root}" >/dev/null 2>&1; then
  fail "Helper chấp nhận regular file làm data root"
fi

for root_alias in '///' '/./' '/tmp/..'; do
  if "${PREPARER}" "${root_alias}" >/dev/null 2>&1; then
    fail "Helper chấp nhận alias filesystem root: ${root_alias}"
  fi
done

grep -Fq 'data_root_state="$("${data_root_preparer}" "${DUB_NATIVE_ROOT}")"' \
  "${INSTALLER}" || fail "Installer không gọi data root helper"
if grep -Fq 'install -d -m 0750 "${DUB_NATIVE_ROOT}"' "${INSTALLER}"; then
  fail "Installer tái xuất hiện lệnh đổi mode data root hiện hữu"
fi
grep -Fq '[[ "${data_root_state}" == "created" ]]' "${INSTALLER}" \
  || fail "Installer không giới hạn bước siết quyền vào data root mới"
grep -Fq '[[ -v DUB_NATIVE_ROOT ]]' "${INSTALLER}" \
  || fail "Installer không từ chối DUB_NATIVE_ROOT kế thừa từ environment"
grep -Fq 'systemctl start thuyet-minh-offline.service' "${INSTALLER}" \
  || fail "Installer không khởi động systemd control plane"
grep -Fq 'Giữ nguyên systemd unit và trạng thái enable hiện có' "${INSTALLER}" \
  || fail "Installer không bảo toàn systemd unit hiện hữu"

printf 'Data root permission tests: PASS\n'
