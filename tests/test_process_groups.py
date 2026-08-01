from __future__ import annotations

import signal

import dub_server.process_groups as process_groups


class FakeProcess:
    def __init__(self, pid: int | None = None) -> None:
        if pid is not None:
            self.pid = pid
        self.returncode = None
        self.terminated = 0
        self.killed = 0

    def terminate(self) -> None:
        self.terminated += 1

    def kill(self) -> None:
        self.killed += 1


def test_spawn_options_create_an_owned_process_group() -> None:
    assert process_groups.process_group_spawn_options("posix") == {
        "start_new_session": True
    }
    assert process_groups.process_group_spawn_options("nt")["creationflags"]


def test_posix_signal_targets_the_whole_owned_group(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        process_groups,
        "_signal_posix_group",
        lambda pid, group_signal: calls.append((pid, group_signal)),
    )
    process = FakeProcess(pid=4321)

    process_groups.signal_process_group(process, force=False, platform_name="posix")
    process_groups.signal_process_group(process, force=True, platform_name="posix")

    assert calls == [
        (4321, getattr(signal, "SIGTERM", 15)),
        (4321, getattr(signal, "SIGKILL", 9)),
    ]
    assert process.terminated == 0
    assert process.killed == 0


def test_injected_handle_without_pid_uses_direct_fallback() -> None:
    process = FakeProcess()

    process_groups.signal_process_group(process, force=False, platform_name="posix")
    process_groups.signal_process_group(process, force=True, platform_name="posix")

    assert process.terminated == 1
    assert process.killed == 1


def test_windows_force_kill_uses_tree_and_falls_back(monkeypatch) -> None:
    process = FakeProcess(pid=9876)
    monkeypatch.setattr(process_groups, "_force_kill_windows_tree", lambda _pid: False)

    process_groups.signal_process_group(process, force=True, platform_name="nt")

    assert process.killed == 1
