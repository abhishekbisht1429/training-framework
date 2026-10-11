"""Artifact lease and ready marker (building block B3).

A small primitive shared by the environment cache, the data uploader, and
the data preparer; it knows nothing about what it guards. Standard library
only, Python 3.8 syntax (the bootstrap bundles it). It depends on a
filesystem (`mkdir_atomic`, `read`, `write`, `delete`) and an owner probe
(`is_alive(owner) -> True | False | None`).
"""

import json
import time

MARKER_SUFFIX = ".tf-hpc-ready"
LOCK_SUFFIX = ".tf-hpc-lock"


class LeaseError(Exception):
    """The lease cannot be taken (an owner that cannot be checked from here)."""


class KeyMismatch(Exception):
    """A ready artifact at this location has a different key."""


def read_marker(filesystem, location):
    data = filesystem.read(location + MARKER_SUFFIX)
    if data is None:
        return None
    return json.loads(data.decode())["key"]


def check_marker(filesystem, location, key):
    """True when `location` is ready with `key`; raises KeyMismatch when it
    is ready with another key (a ready artifact is never modified)."""
    found = read_marker(filesystem, location)
    if found is None:
        return False
    if found != key:
        raise KeyMismatch(
            "{} is ready with key {} but this batch needs key {}; use a new "
            "location (or remove the old artifact deliberately)".format(location, found, key)
        )
    return True


def write_marker(filesystem, location, key):
    filesystem.write(location + MARKER_SUFFIX, json.dumps({"key": key}).encode())


class Lease:
    def __init__(self, filesystem, probe, location, owner, poll=5.0, sleep=time.sleep,
                 notify=None):
        self.filesystem = filesystem
        self.probe = probe
        self.location = location
        self.lock = location + LOCK_SUFFIX
        self.owner = owner
        self.poll = poll
        self.sleep = sleep
        self.took_over = False
        self.held = False
        self.notify = notify or (lambda message: None)

    def _holder(self):
        data = self.filesystem.read(self.lock + "/owner.json")
        return json.loads(data.decode()) if data else None

    def acquire(self):
        """Take the lease, waiting while a live owner holds it. A dead owner's
        lease is taken over (`took_over` is set); an owner this host cannot
        check is never taken over. No timers."""
        parent = self.lock.rsplit("/", 1)[0]
        if parent:
            self.filesystem.makedirs(parent)   # a fresh root has no directory yet
        announced = None
        while True:
            if self.filesystem.mkdir_atomic(self.lock):
                self.filesystem.write(self.lock + "/owner.json",
                                      json.dumps(self.owner).encode())
                self.held = True
                return
            holder = self._holder()
            if holder is None:
                # Created but owner not yet written: its creator is alive.
                self.sleep(self.poll)
                continue
            alive = self.probe.is_alive(holder)
            if alive is None:
                raise LeaseError(
                    "{} is held by {} which cannot be checked from here; if "
                    "that owner is gone, remove {} by hand".format(
                        self.location, json.dumps(holder, sort_keys=True), self.lock)
                )
            if alive:
                if holder != announced:
                    self.notify("waiting for the lease on {} held by {}".format(
                        self.location, json.dumps(holder, sort_keys=True)))
                    announced = holder
                self.sleep(self.poll)
                continue
            self.notify("taking over the lease from {}, which is no longer running".format(
                json.dumps(holder, sort_keys=True)))
            self.filesystem.delete(self.lock)
            self.took_over = True

    def release(self):
        if self.held:
            self.filesystem.delete(self.lock)
            self.held = False
