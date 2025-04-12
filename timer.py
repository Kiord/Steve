import timeit

class FrameTimer:
    def __init__(self):
        self._last = timeit.default_timer()

    def get_dt(self):
        now = timeit.default_timer()
        dt = now - self._last
        self._last = now
        return dt