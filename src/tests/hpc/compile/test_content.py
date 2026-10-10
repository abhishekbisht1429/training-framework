import os
import threading

import pytest

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
