class TickHandleError(Exception):
    pass


class OpenOrderError(TickHandleError):
    pass


class CloseOrderError(TickHandleError):
    pass