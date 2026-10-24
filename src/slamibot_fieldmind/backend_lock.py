"""Nonblocking OS lock: only one backend may own a record directory."""

import os


class BackendLock:
    def __init__(self, root):
        self.handle = (root / ".backend.lock").open("a+b")
        if self.handle.seek(0, 2) == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.handle.close()
            raise RuntimeError("another backend owns this record directory") from error

    def close(self):
        self.handle.close()
