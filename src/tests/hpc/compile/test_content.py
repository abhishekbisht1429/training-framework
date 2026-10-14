import io
import os
import sqlite3
import threading
from contextlib import closing

import pytest

from training_framework.hpc.__main__ import HashProgress
from training_framework.hpc.compile.content import HashCache, content_summary


@pytest.fixture
def cache(tmp_path):
    return HashCache(tmp_path / "cache" / "hashes.sqlite3")


def make_tree(root):
    (root / "sub").mkdir(parents=True)
    (root / "a.bin").write_bytes(b"alpha")
    (root / "sub" / "b.bin").write_bytes(b"beta")
    return root


def test_the_key_depends_on_content_not_on_location_or_order(tmp_path, cache):
    first = content_summary(str(make_tree(tmp_path / "one")), cache)
    second = content_summary(str(make_tree(tmp_path / "two")), cache)
    assert first.key == second.key
    assert (first.files, first.size) == (2, 9)


def test_a_content_change_preserving_size_and_mtime_invalidates(tmp_path, cache):
    root = make_tree(tmp_path / "data")
    before = content_summary(str(root), cache)
    target = root / "a.bin"
    stat = os.stat(target)
    target.write_bytes(b"ALPHA")                                 # same size
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))    # same mtime
    after = content_summary(str(root), cache)
    assert after.key != before.key


def test_a_replaced_file_with_a_new_inode_is_rehashed(tmp_path, cache):
    root = make_tree(tmp_path / "data")
    before = content_summary(str(root), cache)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"other")
    os.replace(replacement, root / "a.bin")
    assert content_summary(str(root), cache).key != before.key


def test_unchanged_files_come_from_the_cache(tmp_path, cache, monkeypatch):
    root = make_tree(tmp_path / "data")
    first = content_summary(str(root), cache)
    opened = []
    real_open = open

    def tracking_open(path, *args, **kwargs):
        opened.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    assert content_summary(str(root), cache).key == first.key
    assert not [path for path in opened if str(path).endswith(".bin")]


def test_a_single_file_has_a_key(tmp_path, cache):
    path = tmp_path / "model.pt"
    path.write_bytes(b"weights")
    assert content_summary(str(path), cache).files == 1


def test_symbolic_links_to_directories_are_refused(tmp_path, cache):
    root = make_tree(tmp_path / "data")
    (root / "link").symlink_to(root / "sub", target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link to a directory"):
        content_summary(str(root), cache)


def test_concurrent_cache_writers_stay_valid(tmp_path):
    root = make_tree(tmp_path / "data")
    path = tmp_path / "cache" / "hashes.sqlite3"
    keys, errors = [], []

    def work():
        try:
            keys.append(content_summary(str(root), HashCache(path)).key)
        except Exception as error:  # surfaced by the assertion below
            errors.append(error)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(set(keys)) == 1


def make_files(root, count):
    root.mkdir(parents=True)
    for index in range(count):
        (root / f"f{index:03d}.bin").write_bytes(bytes([index % 256]) * 10)
    return root


def test_progress_reports_every_file_against_the_total(tmp_path, cache):
    root = make_files(tmp_path / "data", 5)
    calls = []
    content_summary(str(root), cache, lambda *args: calls.append(args))
    assert [call[1:3] for call in calls] == [(done, 5) for done in range(1, 6)]
    assert calls[-1] == (str(root), 5, 5, 50)


def test_a_dataset_uses_one_connection_however_many_files(tmp_path, cache, monkeypatch):
    root = make_files(tmp_path / "data", 40)
    connections = []
    real_connect = HashCache._connect
    monkeypatch.setattr(HashCache, "_connect", lambda self: connections.append(1) or real_connect(self))
    content_summary(str(root), cache)
    content_summary(str(root), cache)          # all cached
    assert len(connections) == 2


def test_an_interrupted_run_keeps_what_it_hashed(tmp_path, cache):
    root = make_files(tmp_path / "data", 10)

    def interrupt(root_path, done, total, size):
        if done == 4:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        content_summary(str(root), cache, interrupt)
    with closing(sqlite3.connect(cache.path)) as connection:
        (rows,) = connection.execute("SELECT COUNT(*) FROM hashes").fetchone()
    assert rows == 4


class FakeStream(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self.tty = tty

    def isatty(self):
        return self.tty


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_hash_progress_is_quiet_for_quick_datasets():
    stream, clock = FakeStream(tty=False), FakeClock()
    report = HashProgress(stream, delay=1.0, clock=clock)
    for done in range(1, 4):
        report("/data/small", done, 3, done * 10)
    assert stream.getvalue() == ""


def test_hash_progress_logs_lines_without_a_terminal():
    stream, clock = FakeStream(tty=False), FakeClock()
    report = HashProgress(stream, delay=1.0, log_interval=10.0, clock=clock)
    for done in range(1, 101):
        clock.now = done * 0.5                 # the first file at 0.5s, the last at 50s
        report("/data/imagenet", done, 100, done * 1024)
    lines = stream.getvalue().splitlines()
    assert lines[0].startswith("[hpc] hashing /data/imagenet (upload source): 3/100 files")   # after the 1s delay
    assert 4 <= len(lines) <= 7
    assert lines[-1].endswith("100/100 files, 100.0 KiB, done in 50s")


def test_hash_progress_rewrites_one_line_on_a_terminal():
    stream, clock = FakeStream(tty=True), FakeClock()
    report = HashProgress(stream, delay=0.0, clock=clock)
    for done in range(1, 4):
        clock.now = done                       # timed from the first file
        report("/data/x", done, 3, done)
    text = stream.getvalue()
    assert text.count("\r\033[K") == 3 and text.endswith("done in 2s\n")


def test_a_second_run_reports_checking_not_hashing(tmp_path):
    from pathlib import Path

    from training_framework.hpc.__main__ import WorkstationFacts

    root = make_files(tmp_path / "data", 3)
    outputs = []
    for _ in range(2):
        stream = FakeStream(tty=False)
        facts = WorkstationFacts(tmp_path, cache_path=Path(tmp_path / "cache.sqlite3"),
                                 progress=HashProgress(stream, delay=0.0, log_interval=0.0))
        facts.content(str(root))
        outputs.append(stream.getvalue().splitlines()[-1])
    assert "3/3 files" in outputs[0] and "(3 hashed, 0 unchanged since the last run)" in outputs[0]
    assert outputs[1].startswith(f"[hpc] checking {root}")
    assert "(0 hashed, 3 unchanged since the last run)" in outputs[1]
