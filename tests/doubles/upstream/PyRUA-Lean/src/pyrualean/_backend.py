"""Exceptions with the upstream constructor shapes (TEST DOUBLE)."""


class ToolError(Exception):
    def __init__(self, tool, message):
        super().__init__(f'{tool}: {message}')
        self.tool = tool
        self.message = message


class EpisodeFinished(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason
