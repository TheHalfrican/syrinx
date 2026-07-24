"""Minimal torch stand-in — the worker needs no_grad() and cuda.is_available()."""


class no_grad:  # noqa: N801 — matches torch's own lowercase name
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Cuda:
    @staticmethod
    def is_available():
        return False


cuda = _Cuda()
