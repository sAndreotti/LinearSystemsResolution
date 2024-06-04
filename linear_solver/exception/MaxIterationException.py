class MaxIterationException(Exception):
    def __init__(self, message, iter=None):
        super().__init__(message)
        self._iter = iter

    def get_iter(self):
        return self._iter

