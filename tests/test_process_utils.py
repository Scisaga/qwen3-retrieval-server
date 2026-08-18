import signal

import process_utils


class FakeProcess:
    pid = 12345

    def poll(self):
        return 1


def test_terminate_process_group_targets_group_even_if_parent_exited(monkeypatch):
    calls = []

    def fake_killpg(process_group_id, sig):
        calls.append((process_group_id, sig))
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(process_utils.os, "killpg", fake_killpg)

    process_utils.terminate_process_group(FakeProcess())

    assert calls == [(12345, signal.SIGTERM), (12345, 0)]
